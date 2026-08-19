"""
Train Random Forest and XGBoost classifiers to predict the vegetation-stress
proxy label 2-4 weeks ahead of time, from the previous period(s)' satellite
indices, stress persistence, and accumulated/anomalous weather.

Methodology, in order, all fit ONLY on the training set (2019-2022) -- the
test set (2023-2024) is touched exactly once, at the very end, for reporting:

  1. Median-impute the small amount of missingness in the engineered lag/
     rolling/anomaly features (see build_features.py for why they're
     allowed to be NaN rather than dropped).
  2. Randomized hyperparameter search (5-fold stratified CV, scoring=F1 on
     the stress class) for both models.
  3. Decision-threshold tuning: out-of-fold predicted probabilities on the
     training set (cross_val_predict) are used to pick the probability
     threshold that maximizes F1 -- the default 0.5 cutoff is not assumed
     to be optimal for a ~21% base rate. This threshold is chosen entirely
     from training data, then applied once to the test set.
  4. Final fit on the full training set, evaluated once on the held-out
     2023-2024 test set: accuracy, F1 (stress class, macro), confusion
     matrix, per-country breakdown.

Usage:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                      RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

NUMERIC_FEATURES = [
    "NDVI_lag1", "EVI_lag1", "SAVI_lag1", "NDMI_lag1",
    "ndvi_trend", "evi_trend",
    "ndvi_roll_mean3", "ndvi_roll_std3", "evi_roll_mean3",
    "stress_lag1",
    "tmax_mean", "tmin_mean", "precip_sum", "humidity_mean", "et0_sum",
    "heat_days", "dry_days", "frost_days",
    "tmax_anom", "precip_anom",
    "precip_sum_2p", "heat_days_2p", "dry_days_2p", "et0_sum_2p",
    "lat", "lon", "month_sin", "month_cos",
]
CATEGORICAL_FEATURES = ["country", "climate_zone"]
TEST_YEARS = {2023, 2024}
RANDOM_STATE = 42
N_SEARCH_ITER = 40


def build_matrix(df):
    X_num = df[NUMERIC_FEATURES].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    X = pd.concat([X_num, X_cat], axis=1)
    return X


def tune_threshold(y_true, y_proba, metric=f1_score):
    thresholds = np.arange(0.10, 0.91, 0.02)
    scores = [metric(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_i = int(np.argmax(scores))
    return float(thresholds[best_i]), float(scores[best_i])


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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_distributions = {
        "random_forest": {
            "clf__n_estimators": [200, 300, 400, 600, 800],
            "clf__max_depth": [4, 6, 8, 10, 14, None],
            "clf__min_samples_leaf": [1, 2, 3, 5, 8, 12],
            "clf__max_features": ["sqrt", "log2", 0.5, 0.7],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        },
        "xgboost": {
            "clf__n_estimators": [150, 250, 400, 600],
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
            "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__min_child_weight": [1, 3, 5, 8],
            "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
    }

    base_estimators = {
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    fitted_models = {}

    for model_name, base_clf in base_estimators.items():
        print(f"\n=== {model_name}: hyperparameter search (5-fold CV, scoring=F1 stress) ===")
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", base_clf),
        ])
        search = RandomizedSearchCV(
            pipe, param_distributions[model_name], n_iter=N_SEARCH_ITER, cv=cv,
            scoring="f1", n_jobs=-1, random_state=RANDOM_STATE, refit=True,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        print(f"Best CV F1 (stress): {search.best_score_:.3f}")
        print(f"Best params: {search.best_params_}")

        # Out-of-fold probabilities from the *tuned* pipeline, for threshold
        # selection -- still train-only, no test-set information used.
        oof_proba = cross_val_predict(best_pipe, X_train, y_train, cv=cv,
                                       method="predict_proba", n_jobs=-1)[:, 1]
        best_threshold, oof_f1_at_threshold = tune_threshold(y_train, oof_proba)
        print(f"Tuned decision threshold (from train OOF predictions): {best_threshold:.2f} "
              f"(OOF F1 at this threshold: {oof_f1_at_threshold:.3f}, vs {f1_score(y_train, (oof_proba>=0.5).astype(int)):.3f} at 0.50)")

        best_pipe.fit(X_train, y_train)
        y_proba_test = best_pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba_test >= best_threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1_stress = f1_score(y_test, y_pred, pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["no_stress", "stress"], output_dict=True)

        print(f"Test set (2023-2024) @ threshold={best_threshold:.2f} — accuracy: {acc:.3f} | "
              f"F1 (stress class): {f1_stress:.3f} | F1 (macro): {f1_macro:.3f}")
        print("Confusion matrix [rows=true, cols=pred] [[TN, FP], [FN, TP]]:")
        print(np.array(cm))

        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = pd.Series(clf.feature_importances_, index=feature_columns)
            importances.sort_values(ascending=False).to_csv(
                REPORTS_DIR / f"feature_importance_{model_name}.csv", header=["importance"])

        joblib.dump(best_pipe, MODELS_DIR / f"{model_name}.joblib")
        fitted_models[model_name] = best_pipe

        results[model_name] = {
            "cv_f1_stress_mean": float(search.best_score_),
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "decision_threshold": best_threshold,
            "test_accuracy": float(acc), "test_f1_stress": float(f1_stress),
            "test_f1_macro": float(f1_macro), "confusion_matrix": cm,
            "classification_report": report,
        }

    # Per-country breakdown for the better model (by test F1), for the cross-country comparison.
    best_name = max(results, key=lambda k: results[k]["test_f1_stress"])
    best_pipe = fitted_models[best_name]
    best_threshold = results[best_name]["decision_threshold"]
    y_proba_best = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_best = (y_proba_best >= best_threshold).astype(int)

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
    results["n_features"] = len(feature_columns)

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"\nBest model on held-out test F1: {best_name}")
    print(f"Saved models to {MODELS_DIR}, metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
