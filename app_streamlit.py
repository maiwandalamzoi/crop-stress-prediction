"""
Crop Stress Prediction -- visualization app.

Shows the trained models' predictions on a map of the 36 field locations
(Afghanistan / Netherlands / New Zealand), lets you inspect a location's
NDVI history and stress flags over time, and compares stress incidence and
model performance across the three countries.

Run:
    streamlit run app_streamlit.py
"""
import json
from pathlib import Path

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# From extract_satellite.py: 2019-01-01 to 2024-12-31 in 16-day steps.
N_PERIODS_POSSIBLE = 137
COUNTRY_COLORS = {"Afghanistan": "#c53030", "Netherlands": "#2b6cb0", "New Zealand": "#2f855a"}

st.set_page_config(page_title="Crop Stress Prediction", layout="wide")


@st.cache_data
def load_features():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["period_start"])
    return df


@st.cache_resource
def load_model(name):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_feature_columns():
    with open(MODELS_DIR / "feature_columns.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    with open(REPORTS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_raw_satellite():
    """
    Every usable (cloud-free) Sentinel-2 composite actually pulled -- before
    build_features.py drops rows for modeling. This is the honest picture of
    how often a clean satellite reading was actually available, including the
    gaps that get filtered out of the training data.
    """
    df = pd.read_csv(RAW_DIR / "sentinel2_timeseries.csv", parse_dates=["period_start"])
    df = df.sort_values(["name", "period_start"]).reset_index(drop=True)
    df["gap_days"] = df.groupby("name")["period_start"].diff().dt.days
    return df


def build_matrix(df, feature_columns):
    numeric_cols = [c for c in feature_columns if not c.startswith(("country_", "climate_zone_"))]
    X_num = df[numeric_cols].copy()
    X_cat = pd.get_dummies(df[["country", "climate_zone"]], prefix=["country", "climate_zone"])
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


@st.cache_resource
def build_explainer(model_name):
    pipe = load_model(model_name)
    return shap.TreeExplainer(pipe.named_steps["clf"])


def explain_one_row(model_name, feature_columns, row_df):
    """
    Real per-prediction SHAP breakdown: how much each feature actually pushed
    this one row's prediction away from the model's average. Random Forest's
    TreeExplainer returns values already in probability space (one array per
    class); XGBoost's returns log-odds (its native training objective) --
    both are handled and reported honestly in whichever unit shap chooses,
    with a note converting back to a real probability.
    """
    pipe = load_model(model_name)
    row_X = build_matrix(row_df, feature_columns)  # same one-hot encoding as every other prediction
    row_imputed = pipe.named_steps["impute"].transform(row_X.values)
    explainer = build_explainer(model_name)
    sv = explainer(row_imputed)
    sv.feature_names = feature_columns
    values = np.array(sv.values)
    if values.ndim == 3:  # (n_rows, n_features, n_classes) -- Random Forest
        single = sv[0, :, 1]
        is_probability_space = True
    else:  # (n_rows, n_features) -- XGBoost, single log-odds output
        single = sv[0]
        is_probability_space = False
    return single, is_probability_space


def main():
    st.title("🌾 Crop Stress Prediction from Satellite Data")
    st.caption(
        "Sentinel-2 (Google Earth Engine) + Open-Meteo weather -> NDVI-anomaly stress "
        "proxy, predicted 2-4 weeks ahead. Afghanistan / Netherlands / New Zealand."
    )

    df = load_features()
    feature_columns = load_feature_columns()
    metrics = load_metrics()

    model_name = st.sidebar.selectbox("Model", ["random_forest", "xgboost"],
                                       index=0 if metrics["best_model"] == "random_forest" else 1)
    model = load_model(model_name)
    m = metrics[model_name]
    st.sidebar.metric("Test accuracy (2023-2024)", f"{m['test_accuracy']:.1%}")
    st.sidebar.metric("Test F1, stress class", f"{m['test_f1_stress']:.3f}")
    st.sidebar.metric("Best CV F1 (tuning)", f"{m['cv_f1_stress_mean']:.3f}")
    st.sidebar.caption(f"Decision threshold: {m['decision_threshold']:.2f} (tuned on train, not 0.5 default)")

    countries = ["All"] + sorted(df["country"].unique().tolist())
    country_filter = st.sidebar.selectbox("Country", countries)

    df_f = df if country_filter == "All" else df[df["country"] == country_filter]

    X_all = build_matrix(df, feature_columns)
    df = df.copy()
    df["pred_stress_prob"] = model.predict_proba(X_all)[:, 1]
    # Use the threshold tuned on training-set out-of-fold predictions (see
    # train.py), not the model's naive 0.5 default, so displayed predictions
    # match the reported test metrics.
    df["pred_stress"] = (df["pred_stress_prob"] >= m["decision_threshold"]).astype(int)
    df_f = df if country_filter == "All" else df[df["country"] == country_filter]

    latest = df_f.sort_values("period_start").groupby("name").tail(1)

    tab_map, tab_timeseries, tab_compare, tab_coverage, tab_explain = st.tabs(
        ["🗺️ Map (latest period)", "📈 Location history", "🌍 Country comparison",
         "📅 Data coverage", "🧮 How a prediction is made"]
    )

    with tab_map:
        st.subheader(f"Latest predicted stress risk per field ({country_filter})")
        center = [latest["lat"].mean(), latest["lon"].mean()] if len(latest) else [20, 20]
        fmap = folium.Map(location=center, zoom_start=2 if country_filter == "All" else 5,
                           tiles="CartoDB positron")
        for _, row in latest.iterrows():
            color = "#c53030" if row["pred_stress"] == 1 else "#2f855a"
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=8, color=color, fill=True, fill_opacity=0.8,
                popup=(f"<b>{row['name']}, {row['country']}</b><br>"
                       f"Period: {row['period_start'].date()}<br>"
                       f"Predicted stress prob: {row['pred_stress_prob']:.2f}<br>"
                       f"Actual label: {'stress' if row['stress'] == 1 else 'no stress'}"),
            ).add_to(fmap)
        st_folium(fmap, width=1100, height=520)
        st.caption("🔴 predicted stress this period · 🟢 predicted no stress")

    with tab_timeseries:
        loc = st.selectbox("Location", sorted(df_f["name"].unique()))
        loc_df = df[df["name"] == loc].sort_values("period_start")
        chart_df = loc_df.set_index("period_start")[["NDVI_lag1", "pred_stress_prob"]]
        chart_df.columns = ["NDVI (prior period)", "Predicted stress probability"]
        st.line_chart(chart_df)
        st.dataframe(
            loc_df[["period_start", "NDVI_lag1", "ndvi_zscore", "stress", "pred_stress", "pred_stress_prob"]]
            .sort_values("period_start", ascending=False).head(20),
            width="stretch",
        )

    with tab_compare:
        st.subheader("Vegetation-stress incidence by country (2019-2024, all periods)")
        rates = df.groupby("country")["stress"].mean().sort_values(ascending=False)
        st.bar_chart(rates)
        st.caption(
            "New Zealand is Southern Hemisphere, so its growing season is offset ~6 months "
            "from Afghanistan/Netherlands -- stress rates are computed against each "
            "location's own seasonal baseline, so the offset doesn't bias the comparison."
        )

        st.subheader("Per-country test-set performance (2023-2024)")
        pc = pd.DataFrame(metrics["per_country_test_metrics"]).T
        pc.index.name = "country"
        st.dataframe(pc, width="stretch")

        st.subheader("NDVI seasonal profile by country")
        seasonal = df.copy()
        seasonal["month"] = seasonal["period_start"].dt.month
        seasonal = seasonal.groupby(["month", "country"])["NDVI_lag1"].mean().unstack()
        st.line_chart(seasonal)

    with tab_coverage:
        st.subheader("How much usable satellite data did we actually get?")
        raw_sat = load_raw_satellite()
        n_locations = raw_sat["name"].nunique()
        max_possible = N_PERIODS_POSSIBLE * n_locations
        coverage = len(raw_sat) / max_possible

        col1, col2, col3 = st.columns(3)
        col1.metric("Usable satellite composites", f"{len(raw_sat):,} / {max_possible:,}")
        col2.metric("Overall coverage", f"{coverage:.1%}")
        col3.metric("Normal cadence", "16 days")
        st.caption(
            "Sentinel-2 (2 satellites) revisits every ~5 days, but clouds make most individual "
            "images unusable, so this pipeline groups images into 16-day windows and takes the "
            "cloud-free median of whatever falls in each window. Some windows still end up with "
            "zero clean pixels for a location -- those are skipped entirely, which is the real "
            "source of the gaps below (not a bug: satellite optical imagery just can't see "
            "through clouds)."
        )

        st.subheader(f"Valid readings captured per location (out of {N_PERIODS_POSSIBLE} possible, 2019-2024)")
        counts = raw_sat.groupby(["country", "name"]).size().reset_index(name="valid_periods")
        counts = counts.sort_values(["country", "valid_periods"])
        fig, ax = plt.subplots(figsize=(9, 8))
        labels = counts["name"] + " (" + counts["country"].str.slice(0, 2) + ")"
        ax.barh(labels, counts["valid_periods"], color=counts["country"].map(COUNTRY_COLORS))
        ax.axvline(N_PERIODS_POSSIBLE, color="#666", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Valid (cloud-free) composites out of {N_PERIODS_POSSIBLE} possible")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Afghanistan locations (AF) consistently have the most valid readings (dry climate, "
            "few clouds); Netherlands (NE) and New Zealand locations tend to have fewer "
            "(cloudier, maritime climate) -- this is the same effect discussed in the README's "
            "cross-country comparison."
        )

        st.subheader("Days between two consecutive usable satellite readings")
        gap_counts = raw_sat["gap_days"].dropna().value_counts().sort_index()
        gap_counts.index = gap_counts.index.astype(int).astype(str) + "d"
        st.bar_chart(gap_counts)
        st.caption(
            "Most consecutive readings are exactly 16 days apart -- the normal cadence. Bars at "
            "32, 48, 64+ days are stretches where one or more whole 16-day windows had zero "
            "cloud-free pixels in a row and got skipped -- longest gap in this dataset: "
            f"{int(raw_sat['gap_days'].max())} days."
        )

    with tab_explain:
        st.subheader("Step by step: how one prediction gets calculated")
        st.markdown(
            "1. **Raw satellite bands** (Sentinel-2, previous 16-day period) become vegetation "
            "indices: NDVI, EVI, SAVI, NDMI (`extract_satellite.py`).\n"
            "2. Those combine with **weather** accumulated over the same prior period(s), plus "
            "engineered signal: NDVI/EVI trend, a 3-period rolling average, whether the field "
            "was *already* stressed last period (`stress_lag1`), and weather anomalies vs. that "
            "location's own climate history (`build_features.py`).\n"
            "3. That's **43 numbers** total (after turning country/climate-zone into 0/1 "
            "columns) -- the model's actual input for one row.\n"
            "4. Missing values (a few percent, mostly at the start of a location's history) are "
            "filled with the training-set median.\n"
            "5. The trained model -- hundreds of decision trees, combined -- turns those 43 "
            "numbers into a **stress probability** between 0 and 1.\n"
            "6. That probability is compared against the model's tuned decision threshold "
            f"(**{m['decision_threshold']:.2f}**, not the default 0.5) -- above it: predicted "
            "stress; below: predicted no-stress."
        )

        st.markdown("**Pick one real (location, period) to see exactly how steps 5-6 happened for it:**")
        ec1, ec2 = st.columns(2)
        loc_e = ec1.selectbox("Location", sorted(df["name"].unique()), key="explain_loc")
        loc_e_df = df[df["name"] == loc_e].sort_values("period_start", ascending=False)
        period_labels = loc_e_df["period_start"].dt.date.astype(str).tolist()
        period_e = ec2.selectbox("Period", period_labels, key="explain_period")
        row = loc_e_df[loc_e_df["period_start"].dt.date.astype(str) == period_e].iloc[[0]]

        single, is_prob_space = explain_one_row(model_name, feature_columns, row)
        final_val = float(single.base_values) + float(np.sum(single.values))
        final_proba = final_val if is_prob_space else 1 / (1 + np.exp(-final_val))
        predicted_label = "STRESS" if final_proba >= m["decision_threshold"] else "no stress"
        actual_label = "stress" if int(row["stress"].iloc[0]) == 1 else "no stress"

        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(single, show=False, max_display=12)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

        units_note = (
            "Bars are in real probability units."
            if is_prob_space else
            f"Bars are in log-odds units (XGBoost's native training scale) -- converting the "
            f"final total to probability: sigmoid({final_val:.3f}) = {final_proba:.1%}."
        )
        st.caption(
            f"🔴/pink bars pushed this prediction toward 'stress'; 🔵 blue bars pushed toward "
            f"'no stress'. {units_note} Final predicted probability: **{final_proba:.1%}**, "
            f"vs. threshold **{m['decision_threshold']:.0%}** → predicted **{predicted_label}** "
            f"(what actually happened at {loc_e} that period: **{actual_label}**)."
        )

        st.divider()
        st.subheader("Random Forest vs. XGBoost — what's actually different")
        st.markdown(
            "Both are **decision-tree ensembles** (many trees, not one), but they build and "
            "combine those trees in fundamentally different ways:\n\n"
            "- **Random Forest — bagging.** Builds many trees *independently and in parallel*, "
            "each on a random subset of rows and features, then **averages** all their votes. "
            "No tree ever sees another tree's mistakes. This makes it hard to overfit and "
            "gives stable, probability-native outputs, but it can't specifically correct the "
            "errors its own trees are making.\n"
            "- **XGBoost — gradient boosting.** Builds trees *one at a time, sequentially*: "
            "every new tree is trained specifically to correct the residual errors of all the "
            "trees built so far, with explicit regularization (`reg_lambda` in this project's "
            "tuned params) to keep that error-chasing from overfitting. Outputs are in "
            "log-odds space, summed across trees, and only converted to a probability at the "
            "very end (the sigmoid step above).\n\n"
            f"**On this dataset**, that difference mattered only a little: XGBoost came out "
            f"slightly ahead (test F1 {metrics['xgboost']['test_f1_stress']:.3f} vs. Random "
            f"Forest's {metrics['random_forest']['test_f1_stress']:.3f}) after both were tuned "
            "the same way (5-fold CV hyperparameter search + threshold tuning) -- essentially "
            "a statistical tie, not a decisive win for boosting."
        )


if __name__ == "__main__":
    main()
