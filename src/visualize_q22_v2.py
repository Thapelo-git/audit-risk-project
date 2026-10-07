"""
Q2.2 visualisations for modeling_dataset.csv.
Run from the project root:  python src/visualize_q22_v2.py
Saves 5 PNG charts into figures/, each with a title, labels, and units.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA = Path("data/processed/modeling_dataset.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

TARGET_ORDER = ["not_at_risk", "at_risk"]


def main():
    df = pd.read_csv(DATA, low_memory=False)

    # ---------------- Chart 1: Target distribution ----------------
    plt.figure(figsize=(6, 5))
    counts = df["target_next_year"].value_counts().reindex(TARGET_ORDER)
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                      palette=["#2a9d8f", "#e76f51"], legend=False)
    ax.set_title("Distribution of Target: Audit Risk at Year t+1", fontsize=13)
    ax.set_xlabel("Risk category (year t+1)")
    ax.set_ylabel("Number of municipality-year records")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=150)
    plt.close()

    # ---------------- Chart 2: Missingness ----------------
    id_cols = ["demarcation.code", "Name", "financial_year_end.year", "target_next_year",
               "prior_audit_opinion_raw", "target_opinion_raw"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    missing_pct = (df[feature_cols].isna().mean() * 100).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=missing_pct.values, y=missing_pct.index, hue=missing_pct.index,
                      palette="flare", legend=False)
    ax.set_title("Missingness by Feature", fontsize=13)
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_missingness_by_feature.png", dpi=150)
    plt.close()

    # ---------------- Chart 3: Feature-target relationship 1 ----------------
    # Operating surplus vs next-year risk
    plot_df = df.dropna(subset=["operating_surplus"]).copy()
    cap_lo, cap_hi = plot_df["operating_surplus"].quantile([0.05, 0.95])
    plot_df["surplus_capped"] = plot_df["operating_surplus"].clip(cap_lo, cap_hi)

    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(data=plot_df, x="target_next_year", y="surplus_capped",
                      order=TARGET_ORDER, hue="target_next_year",
                      palette=["#2a9d8f", "#e76f51"], legend=False)
    ax.set_title("Year t Operating Surplus vs Year t+1 Audit Risk\n(capped at 5th-95th percentile)",
                 fontsize=12)
    ax.set_xlabel("Risk category (year t+1)")
    ax.set_ylabel("Operating surplus (R, capped)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_surplus_vs_risk.png", dpi=150)
    plt.close()

    # ---------------- Chart 4: Feature-target relationship 2 ----------------
    # Debtor days vs next-year risk
    plot_df = df.dropna(subset=["debtor_days"]).copy()
    cap_hi = plot_df["debtor_days"].quantile(0.95)
    plot_df["debtor_days_capped"] = plot_df["debtor_days"].clip(upper=cap_hi)

    plt.figure(figsize=(7, 5))
    ax = sns.boxplot(data=plot_df, x="target_next_year", y="debtor_days_capped",
                      order=TARGET_ORDER, hue="target_next_year",
                      palette=["#2a9d8f", "#e76f51"], legend=False)
    ax.set_title("Year t Debtor Days vs Year t+1 Audit Risk\n(capped at 95th percentile)",
                 fontsize=12)
    ax.set_xlabel("Risk category (year t+1)")
    ax.set_ylabel("Debtor days")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_debtordays_vs_risk.png", dpi=150)
    plt.close()

    # ---------------- Chart 5: Time view ----------------
    # Share of at-risk outcomes by year (since municipality count is now small,
    # a time trend is more informative than a province split for 240 rows)
    df["at_risk_flag"] = (df["target_next_year"] == "at_risk")
    by_year = df.groupby("financial_year_end.year")["at_risk_flag"].mean() * 100

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x=by_year.index, y=by_year.values, marker="o", color="#e76f51")
    ax.set_title("Share of Municipalities Predicted At-Risk, by Year (t+1 outcome)", fontsize=13)
    ax.set_xlabel("Year t (feature year; outcome is for t+1)")
    ax.set_ylabel("% of records that are at-risk in t+1")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_risk_trend_by_year.png", dpi=150)
    plt.close()

    print("Saved 5 charts to figures/:")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(" -", f.name)

    print("\nRow count per year (sample size behind chart 5):")
    print(df.groupby("financial_year_end.year").size().to_string())


if __name__ == "__main__":
    main()
