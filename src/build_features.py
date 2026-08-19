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


def leave_one_year_out_zscore(df, value_col, group_cols=("name", "period_of_year"),
                               year_col="year", min_years=MIN_BASELINE_YEARS):
    """
    Generic leave-one-year-out z-score: for each row, compare its value against
    the mean/std of the *other* years at the same (group_cols) bucket. Used for
    both the NDVI stress label and the weather anomaly features below, so a
    weather reading is judged against that same location's own climatology,
    not a fixed threshold that would conflate Afghanistan's climate with the
    Netherlands' or New Zealand's.
    """
    z = np.full(len(df), np.nan)
    for _, grp in df.groupby(list(group_cols)):
        years = grp[year_col].values
        vals = grp[value_col].values
        idx = grp.index.values
        for i, y in enumerate(years):
            others = vals[years != y]
            if len(others) < min_years:
                continue
            mu, sigma = others.mean(), others.std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                continue
            z[idx[i]] = (vals[i] - mu) / sigma
    return z


def add_stress_labels(df):
    """Leave-one-year-out NDVI z-score anomaly per (name, period_of_year)."""
    z_scores = leave_one_year_out_zscore(df, "NDVI")
    df["ndvi_zscore"] = z_scores
    df["stress"] = np.where(np.isnan(z_scores), np.nan, (z_scores <= Z_STRESS_THRESHOLD).astype(float))
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

    # Stress persistence: was the PREVIOUS period already flagged as stressed?
    # Known at prediction time (it's t-1's label, computed from t-1 data only),
    # and a strong prior given that drought/heat stress tends to be autocorrelated.
    df["stress_lag1"] = df.groupby("name")["stress"].shift(1)

    # Rolling NDVI/EVI stats over the last up-to-3 valid PRIOR periods (excludes
    # the current period itself -- shift(1) first, then roll over the past).
    grp = df.groupby("name")
    df["ndvi_roll_mean3"] = grp["NDVI"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    df["ndvi_roll_std3"] = grp["NDVI"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    df["evi_roll_mean3"] = grp["EVI"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    return df


def main():
    sat = load_satellite()
    sat = add_stress_labels(sat)
    sat = add_lag_features(sat)

    weather = load_weather()
    # Weather anomaly relative to this location's own climatology at this
    # time of year -- e.g. a 30C day means something very different in
    # Nimroz (Afghanistan) than in Groningen (Netherlands); the anomaly
    # normalizes for that the same way the NDVI stress label does.
    weather["tmax_anom"] = leave_one_year_out_zscore(weather, "tmax_mean")
    weather["precip_anom"] = leave_one_year_out_zscore(weather, "precip_sum")

    # Weather features are for the PRIOR period (lag1) and the one before
    # that (lag2), matched by that period's own (name, year, period_of_year).
    sat["lag1_year"] = sat.groupby("name")["year"].shift(1)
    sat["lag1_poy"] = sat.groupby("name")["period_of_year"].shift(1)
    sat["lag2_year"] = sat.groupby("name")["year"].shift(2)
    sat["lag2_poy"] = sat.groupby("name")["period_of_year"].shift(2)

    merged = sat.merge(
        weather,
        left_on=["name", "country", "lag1_year", "lag1_poy"],
        right_on=["name", "country", "year", "period_of_year"],
        how="left",
        suffixes=("", "_w"),
    )
    merged = merged.merge(
        weather[["name", "country", "year", "period_of_year",
                  "precip_sum", "heat_days", "dry_days", "et0_sum"]],
        left_on=["name", "country", "lag2_year", "lag2_poy"],
        right_on=["name", "country", "year", "period_of_year"],
        how="left",
        suffixes=("", "_lag2"),
    )

    # Cumulative weather over the last TWO periods (~4 weeks) -- a single dry,
    # hot 16-day window is noisier evidence of building drought/heat stress
    # than two consecutive ones, so this should be a more stable signal than
    # the single-prior-period weather features alone.
    merged["precip_sum_2p"] = merged["precip_sum"] + merged["precip_sum_lag2"]
    merged["heat_days_2p"] = merged["heat_days"] + merged["heat_days_lag2"]
    merged["dry_days_2p"] = merged["dry_days"] + merged["dry_days_lag2"]
    merged["et0_sum_2p"] = merged["et0_sum"] + merged["et0_sum_lag2"]

    # Cyclical month-of-year encoding for the CURRENT (target) period.
    merged["month"] = merged["period_start"].dt.month
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

    feature_cols = [
        "NDVI_lag1", "EVI_lag1", "SAVI_lag1", "NDMI_lag1",
        "ndvi_trend", "evi_trend",
        "ndvi_roll_mean3", "ndvi_roll_std3", "evi_roll_mean3",
        "stress_lag1",
        "tmax_mean", "tmin_mean", "precip_sum", "humidity_mean", "et0_sum",
        "heat_days", "dry_days", "frost_days",
        "tmax_anom", "precip_anom",
        "precip_sum_2p", "heat_days_2p", "dry_days_2p", "et0_sum_2p",
        "lat", "lon", "month_sin", "month_cos",
        "country", "climate_zone",
    ]
    keep_cols = ["name", "country", "region", "climate_zone", "period_start", "year",
                 "period_of_year", "ndvi_zscore", "stress", "gap_days"] + feature_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # de-dupe, preserve order

    out = merged[keep_cols].copy()

    # Required (dropped if missing): the label itself, and the core t-1
    # satellite/weather signal every row must have. Optional signals with
    # more missingness (2-period cumulative weather, rolling stats,
    # stress_lag1) are allowed to be NaN and imputed at train time instead
    # of dropping rows over them -- they need 2-3 valid prior periods, which
    # costs rows especially at the start of each location's history.
    required = ["stress", "NDVI_lag1", "tmax_mean"]
    before = len(out)
    out = out.dropna(subset=required)
    out = out[out["gap_days"] <= MAX_LAG_GAP_DAYS]
    after = len(out)
    print(f"Rows before filtering: {before}, after dropping missing {required} "
          f"and gaps > {MAX_LAG_GAP_DAYS}d: {after}")

    out["stress"] = out["stress"].astype(int)
    print("Stress label distribution:")
    print(out["stress"].value_counts(normalize=True).rename("share"))
    print("\nRows per country:")
    print(out["country"].value_counts())
    print("\nMissingness in optional engineered features (imputed at train time):")
    optional = ["stress_lag1", "ndvi_roll_mean3", "ndvi_roll_std3", "evi_roll_mean3",
                "tmax_anom", "precip_anom", "precip_sum_2p", "heat_days_2p", "dry_days_2p", "et0_sum_2p"]
    print(out[optional].isna().mean().rename("share_missing"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows x {out.shape[1]} cols to {OUT_PATH}")


if __name__ == "__main__":
    main()
