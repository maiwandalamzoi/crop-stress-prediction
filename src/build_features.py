"""
Join Sentinel-2 vegetation indices with Open-Meteo weather, define a
vegetation-stress proxy label, and build a lead-time prediction feature set.

Label definition ("stress" proxy, not a lab-confirmed disease diagnosis):
    For each location and each 16-day period-of-year bucket, compute a
    leave-one-year-out NDVI baseline (mean, std) across the other years of
    data for that same point and time of year. A period is labeled
    stress = 1 if its NDVI z-score against that baseline is <= -1.0
    (roughly the bottom ~16% of the point's own seasonal distribution).
    This is the same anomaly-based approach operational vegetation
    monitoring systems (e.g. USDA VegDRI, FEWS NET) use in the absence of
    ground-truth field diagnoses.

Prediction framing (avoids same-period leakage):
    Features describing period t-1 (previous satellite composite + weather
    accumulated over that prior 16-day window, plus the t-2 -> t-1 NDVI
    trend) are used to predict period t's stress label. That's a ~2-4 week
    lead time, matching an early-warning use case.

Usage:
    python src/build_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.csv"

PERIOD_DAYS = 16
MAX_LAG_GAP_DAYS = 20  # if the previous row is further back than this, treat as a break
Z_STRESS_THRESHOLD = -1.0
MIN_BASELINE_YEARS = 3


def load_satellite():
    df = pd.read_csv(RAW_DIR / "sentinel2_timeseries.csv", parse_dates=["period_start", "period_end"])
    df["year"] = df["period_start"].dt.year
    df["doy"] = df["period_start"].dt.dayofyear
    df["period_of_year"] = (df["doy"] - 1) // PERIOD_DAYS  # 0..22, resets each calendar year
    df = df.sort_values(["name", "period_start"]).reset_index(drop=True)
    return df


def load_weather():
    df = pd.read_csv(RAW_DIR / "weather_timeseries.csv", parse_dates=["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["period_of_year"] = (df["doy"] - 1) // PERIOD_DAYS
    agg = df.groupby(["name", "country", "year", "period_of_year"]).agg(
        tmax_mean=("temperature_2m_max", "mean"),
        tmin_mean=("temperature_2m_min", "mean"),
        precip_sum=("precipitation_sum", "sum"),
        humidity_mean=("relative_humidity_2m_mean", "mean"),
        et0_sum=("et0_fao_evapotranspiration", "sum"),
        heat_days=("temperature_2m_max", lambda s: int((s > 35).sum())),
        dry_days=("precipitation_sum", lambda s: int((s < 1.0).sum())),
        frost_days=("temperature_2m_min", lambda s: int((s < 0).sum())),
    ).reset_index()
    return agg


def add_stress_labels(df):
    """Leave-one-year-out NDVI z-score anomaly per (name, period_of_year)."""
    z_scores = np.full(len(df), np.nan)
    labels = np.full(len(df), np.nan)

    for (name, poy), grp in df.groupby(["name", "period_of_year"]):
        years = grp["year"].values
        ndvi = grp["NDVI"].values
        idx = grp.index.values
        for i, y in enumerate(years):
            others = ndvi[years != y]
            if len(others) < MIN_BASELINE_YEARS:
                continue
            mu, sigma = others.mean(), others.std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                continue
            z = (ndvi[i] - mu) / sigma
            z_scores[idx[i]] = z
            labels[idx[i]] = 1 if z <= Z_STRESS_THRESHOLD else 0

    df["ndvi_zscore"] = z_scores
    df["stress"] = labels
    return df


def add_lag_features(df):
    df = df.sort_values(["name", "period_start"]).reset_index(drop=True)
    for col in ["NDVI", "EVI", "SAVI", "NDMI"]:
        df[f"{col}_lag1"] = df.groupby("name")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("name")[col].shift(2)
    df["prev_period_start"] = df.groupby("name")["period_start"].shift(1)
    df["gap_days"] = (df["period_start"] - df["prev_period_start"]).dt.days
    df["ndvi_trend"] = df["NDVI_lag1"] - df["NDVI_lag2"]
    df["evi_trend"] = df["EVI_lag1"] - df["EVI_lag2"]
    return df


def main():
    sat = load_satellite()
    sat = add_stress_labels(sat)
    sat = add_lag_features(sat)

    weather = load_weather()
    # Weather features are for the PRIOR period (lag1), matched by that
    # period's own (name, year, period_of_year).
    sat["lag1_year"] = sat.groupby("name")["year"].shift(1)
    sat["lag1_poy"] = sat.groupby("name")["period_of_year"].shift(1)

    merged = sat.merge(
        weather,
        left_on=["name", "country", "lag1_year", "lag1_poy"],
        right_on=["name", "country", "year", "period_of_year"],
        how="left",
        suffixes=("", "_w"),
    )

    # Cyclical month-of-year encoding for the CURRENT (target) period.
    merged["month"] = merged["period_start"].dt.month
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

    feature_cols = [
        "NDVI_lag1", "EVI_lag1", "SAVI_lag1", "NDMI_lag1",
        "ndvi_trend", "evi_trend",
        "tmax_mean", "tmin_mean", "precip_sum", "humidity_mean", "et0_sum",
        "heat_days", "dry_days", "frost_days",
        "lat", "lon", "month_sin", "month_cos",
        "country", "climate_zone",
    ]
    keep_cols = ["name", "country", "region", "climate_zone", "period_start", "year",
                 "period_of_year", "ndvi_zscore", "stress", "gap_days"] + feature_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # de-dupe, preserve order

    out = merged[keep_cols].copy()

    before = len(out)
    out = out.dropna(subset=["stress", "NDVI_lag1", "tmax_mean"])
    out = out[out["gap_days"] <= MAX_LAG_GAP_DAYS]
    after = len(out)
    print(f"Rows before filtering: {before}, after dropping missing lag/label/weather "
          f"and gaps > {MAX_LAG_GAP_DAYS}d: {after}")

    out["stress"] = out["stress"].astype(int)
    print("Stress label distribution:")
    print(out["stress"].value_counts(normalize=True).rename("share"))
    print("\nRows per country:")
    print(out["country"].value_counts())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows x {out.shape[1]} cols to {OUT_PATH}")


if __name__ == "__main__":
    main()
