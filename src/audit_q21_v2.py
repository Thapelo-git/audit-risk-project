"""
Q2.1 data audit for modeling_dataset.csv (the lagged t -> t+1 dataset).
Run from the project root:  python src/audit_q21_v2.py
Produces:
  - data_dictionary.csv
  - audit_report.txt
"""

import pandas as pd
from pathlib import Path

DATA = Path("data/processed/modeling_dataset.csv")

# Identifiers and the target are not model input features
ID_AND_TARGET_COLS = [
    "demarcation.code", "Name", "financial_year_end.year", "target_next_year",
    "prior_audit_opinion_raw", "target_opinion_raw",
]
# prior_audit_outcome IS a feature (audit history up to t), not excluded


def main():
    df = pd.read_csv(DATA, low_memory=False)
    feature_cols = [c for c in df.columns if c not in ID_AND_TARGET_COLS]

    # ---------------- 1. Data dictionary ----------------
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "role": "identifier/target" if col in ID_AND_TARGET_COLS else "feature",
            "dtype": str(s.dtype),
            "non_null_count": s.notna().sum(),
            "missing_pct": round(100 * s.isna().mean(), 1),
            "n_unique": s.nunique(),
            "example_value": s.dropna().iloc[0] if s.notna().any() else None,
        })
    data_dict = pd.DataFrame(rows)
    data_dict.to_csv("data_dictionary.csv", index=False)

    report = []
    report.append(f"MODELLING DATASET AUDIT (lagged t -> t+1)\nShape: {df.shape}\n")

    # ---------------- 2. Class balance ----------------
    report.append("=== CLASS BALANCE (target_next_year) ===")
    counts = df["target_next_year"].value_counts()
    pct = df["target_next_year"].value_counts(normalize=True).round(3) * 100
    for k in counts.index:
        report.append(f"  {k}: {counts[k]} rows ({pct[k]:.1f}%)")

    report.append("\n  prior_audit_outcome (year t, used as a feature):")
    report.append(df["prior_audit_outcome"].value_counts().to_string())
    report.append("")

    # ---------------- 3. Duplicates ----------------
    report.append("=== DUPLICATES ===")
    key_dupes = df.duplicated(subset=["demarcation.code", "financial_year_end.year"]).sum()
    full_dupes = df.duplicated().sum()
    report.append(f"  Duplicate municipality-year keys: {key_dupes}")
    report.append(f"  Fully duplicate rows: {full_dupes}\n")

    # ---------------- 4. Missingness ----------------
    report.append("=== MISSINGNESS ===")
    overall = df[feature_cols].isna().mean().mean() * 100
    report.append(f"  Average missingness across feature columns: {overall:.1f}%")
    report.append("  Missingness by feature:")
    for c in feature_cols:
        pct_missing = round(100 * df[c].isna().mean(), 1)
        if pct_missing > 0:
            report.append(f"    {c}: {pct_missing}% missing")
    report.append("  Row counts by year (t, not t+1):")
    report.append(df.groupby("financial_year_end.year").size().to_string())
    report.append("")

    # ---------------- 5. Invalid values ----------------
    report.append("=== INVALID VALUES ===")
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns
    all_zero_cols = [c for c in numeric_cols if df[c].fillna(0).eq(0).all()]
    report.append(f"  Feature columns that are all zero (check these): {all_zero_cols}")
    negative_counts = {c: (df[c] < 0).sum() for c in numeric_cols if (df[c] < 0).any()}
    report.append(f"  Columns with negative values (may be valid, e.g. operating deficit): {negative_counts}")
    report.append("")

    # ---------------- 6. Outliers ----------------
    report.append("=== OUTLIERS (IQR method) ===")
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n_out = ((s < lo) | (s > hi)).sum()
        if n_out > 0:
            report.append(f"  {c}: {n_out} extreme values (outside [{lo:.1f}, {hi:.1f}])")
    report.append("")

    # ---------------- 7. Leakage check ----------------
    report.append("=== LEAKAGE CHECK ===")
    report.append("  Features are year t (Audited Actual); target is audit opinion at year t+1.")
    report.append("  Year t's own audit was already finalised before t+1 is predicted, so using")
    report.append("  Audited Actual figures from year t is NOT leakage — it reflects what would")
    report.append("  genuinely be known at prediction time.")
    report.append("  'demarcation.code' and 'financial_year_end.year' are identifiers, excluded from modelling.")

    Path("audit_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\nSaved data_dictionary.csv and audit_report.txt")


if __name__ == "__main__":
    main()
