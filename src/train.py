"""
Train Random Forest and XGBoost classifiers to predict the vegetation-stress
proxy label 2-4 weeks ahead of time, from the previous period's satellite
indices and accumulated weather.

Evaluation:
  - Time-based split: train on periods from 2019-2022, test on 2023-2024
    (avoids temporal leakage -- no future data used to predict the past).
  - 5-fold stratified cross-validation on the training set only, as a
    robustness check independent of the specific train/test cut.
  - Test-set accuracy, F1 (macro and stress-class), confusion matrix.

Usage:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

NUMERIC_FEATURES = [
    "NDVI_lag1", "EVI_lag1", "SAVI_lag1", "NDMI_lag1",
    "ndvi_trend", "evi_trend",
    "tmax_mean", "tmin_mean", "precip_sum", "humidity_mean", "et0_sum",
    "heat_days", "dry_days", "frost_days",
    "lat", "lon", "month_sin", "month_cos",
]
CATEGORICAL_FEATURES = ["country", "climate_zone"]
TEST_YEARS = {2023, 2024}
RANDOM_STATE = 42


def build_matrix(df):
    X_num = df[NUMERIC_FEATURES].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    X = pd.concat([X_num, X_cat], axis=1)
    return X


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["period_start"])
    X = build_matrix(df)
    y = df["stress"].values
    feature_columns = list(X.columns)

    train_mask = ~df["year"].isin(TEST_YEARS)
    test_mask = df["year"].isin(TEST_YEARS)
    X_train, y_train = X[train_mask].values, y[train_mask]
    X_test, y_test = X[test_mask].values, y[test_mask]
    df_test = df[test_mask].reset_index(drop=True)

    print(f"Train: {len(X_train)} rows ({y_train.mean():.1%} stress) "
          f"| Test: {len(X_test)} rows ({y_test.mean():.1%} stress)")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")

        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"5-fold CV (train set) — accuracy: {cv_acc.mean():.3f} +/- {cv_acc.std():.3f} "
              f"| F1 (stress class): {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_stress = f1_score(y_test, y_pred, pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["no_stress", "stress"], output_dict=True)

        print(f"Test set (2023-2024) — accuracy: {acc:.3f} | F1 (stress class): {f1_stress:.3f} "
              f"| F1 (macro): {f1_macro:.3f}")
        print("Confusion matrix [rows=true, cols=pred] [[TN, FP], [FN, TP]]:")
        print(np.array(cm))

        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_columns)
            importances.sort_values(ascending=False).to_csv(
                REPORTS_DIR / f"feature_importance_{model_name}.csv", header=["importance"])

        joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")

        results[model_name] = {
            "cv_accuracy_mean": float(cv_acc.mean()), "cv_accuracy_std": float(cv_acc.std()),
            "cv_f1_stress_mean": float(cv_f1.mean()), "cv_f1_stress_std": float(cv_f1.std()),
            "test_accuracy": float(acc), "test_f1_stress": float(f1_stress),
            "test_f1_macro": float(f1_macro), "confusion_matrix": cm,
            "classification_report": report,
        }

    # Per-country breakdown for the better model (by test F1), for the cross-country comparison.
    best_name = max(results, key=lambda k: results[k]["test_f1_stress"])
    best_model = models[best_name]
    y_pred_best = best_model.predict(X_test)
    per_country = {}
    for country in df_test["country"].unique():
        mask = (df_test["country"] == country).values
        per_country[country] = {
            "n": int(mask.sum()),
            "stress_rate": float(y_test[mask].mean()),
            "accuracy": float(accuracy_score(y_test[mask], y_pred_best[mask])),
            "f1_stress": float(f1_score(y_test[mask], y_pred_best[mask], pos_label=1, zero_division=0)),
        }
    results["best_model"] = best_name
    results["per_country_test_metrics"] = per_country
    results["train_rows"] = int(len(X_train))
    results["test_rows"] = int(len(X_test))
    results["train_years"] = sorted(df[train_mask]["year"].unique().tolist())
    results["test_years"] = sorted(TEST_YEARS)

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"\nBest model on held-out test F1: {best_name}")
    print(f"Saved models to {MODELS_DIR}, metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
