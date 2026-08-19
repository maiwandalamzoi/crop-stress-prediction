"""
Generate evaluation figures from the trained models and processed features:
  - confusion matrices for both models (test set, 2023-2024)
  - feature importance bar chart for the best model
  - cross-country stress-rate comparison
  - NDVI seasonal profile by country (illustrates the NH/SH growing-season
    offset between Afghanistan/Netherlands and New Zealand)

Usage:
    python src/evaluate.py   (run after train.py)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def plot_confusion_matrices(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, model_name in zip(axes, ["random_forest", "xgboost"]):
        cm = np.array(metrics[model_name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no_stress", "stress"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["no_stress", "stress"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        f1 = metrics[model_name]["test_f1_stress"]
        acc = metrics[model_name]["test_accuracy"]
        ax.set_title(f"{model_name}\nacc={acc:.3f}  F1(stress)={f1:.3f}")
    fig.suptitle("Test set (2023-2024) confusion matrices")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(best_model_name):
    path = REPORTS_DIR / f"feature_importance_{best_model_name}.csv"
    if not path.exists():
        return
    imp = pd.read_csv(path, index_col=0).iloc[:12]
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.iloc[::-1].plot.barh(ax=ax, legend=False, color="#2b6cb0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top features — {best_model_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_stress_rate_by_country(df):
    rates = df.groupby("country")["stress"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    rates.plot.bar(ax=ax, color=["#c53030", "#2b6cb0", "#2f855a"])
    ax.set_ylabel("Share of periods labeled 'stress'")
    ax.set_title("Vegetation-stress incidence by country (2019-2024, all periods)")
    ax.set_xticklabels(rates.index, rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stress_rate_by_country.png", dpi=150)
    plt.close(fig)
    return rates


def plot_ndvi_seasonality(df):
    df = df.copy()
    df["month"] = pd.to_datetime(df["period_start"]).dt.month
    seasonal = df.groupby(["country", "month"])["NDVI_lag1"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seasonal.plot(ax=ax, marker="o")
    ax.set_xlabel("Calendar month"); ax.set_ylabel("Mean NDVI (prior period)")
    ax.set_title("NDVI seasonal profile by country\n(New Zealand is Southern Hemisphere -> offset ~6 months)")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ndvi_seasonality_by_country.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["period_start"])

    plot_confusion_matrices(metrics)
    plot_feature_importance(metrics["best_model"])
    rates = plot_stress_rate_by_country(df)
    plot_ndvi_seasonality(df)

    lines = ["# Evaluation summary\n"]
    lines.append(f"Best model (by test F1, stress class): **{metrics['best_model']}**\n")
    for name in ["random_forest", "xgboost"]:
        m = metrics[name]
        lines.append(
            f"- **{name}** — test accuracy: {m['test_accuracy']:.3f}, "
            f"test F1 (stress class): {m['test_f1_stress']:.3f}, "
            f"test F1 (macro): {m['test_f1_macro']:.3f}, "
            f"5-fold CV F1: {m['cv_f1_stress_mean']:.3f} +/- {m['cv_f1_stress_std']:.3f}"
        )
    lines.append("\n## Vegetation-stress incidence by country (all 2019-2024 periods)\n")
    for country, rate in rates.items():
        lines.append(f"- {country}: {rate:.1%}")
    lines.append("\n## Per-country test-set performance (best model)\n")
    for country, m in metrics["per_country_test_metrics"].items():
        lines.append(f"- {country}: n={m['n']}, stress_rate={m['stress_rate']:.1%}, "
                      f"accuracy={m['accuracy']:.3f}, F1(stress)={m['f1_stress']:.3f}")

    (REPORTS_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
