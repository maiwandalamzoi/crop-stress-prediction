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
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

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


def build_matrix(df, feature_columns):
    numeric_cols = [c for c in feature_columns if not c.startswith(("country_", "climate_zone_"))]
    X_num = df[numeric_cols].copy()
    X_cat = pd.get_dummies(df[["country", "climate_zone"]], prefix=["country", "climate_zone"])
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


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
    st.sidebar.metric("5-fold CV F1 (train)", f"{m['cv_f1_stress_mean']:.3f} ± {m['cv_f1_stress_std']:.3f}")

    countries = ["All"] + sorted(df["country"].unique().tolist())
    country_filter = st.sidebar.selectbox("Country", countries)

    df_f = df if country_filter == "All" else df[df["country"] == country_filter]

    X_all = build_matrix(df, feature_columns)
    df = df.copy()
    df["pred_stress_prob"] = model.predict_proba(X_all)[:, 1]
    df["pred_stress"] = model.predict(X_all)
    df_f = df if country_filter == "All" else df[df["country"] == country_filter]

    latest = df_f.sort_values("period_start").groupby("name").tail(1)

    tab_map, tab_timeseries, tab_compare = st.tabs(
        ["🗺️ Map (latest period)", "📈 Location history", "🌍 Country comparison"]
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
            use_container_width=True,
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
        st.dataframe(pc, use_container_width=True)

        st.subheader("NDVI seasonal profile by country")
        seasonal = df.copy()
        seasonal["month"] = seasonal["period_start"].dt.month
        seasonal = seasonal.groupby(["month", "country"])["NDVI_lag1"].mean().unstack()
        st.line_chart(seasonal)


if __name__ == "__main__":
    main()
