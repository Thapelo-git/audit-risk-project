"""
Q2.4 preprocessing pipeline for the municipal audit-risk project.
Run from the project root:  python src/pipeline_q24.py

Builds a single scikit-learn Pipeline (imputation -> encoding -> scaling) that is
FIT ONLY ON THE TRAINING SPLIT, then applied to train/val/test. This is the
leakage-safety rule from Q2.3: nothing learned from val/test (medians, scaling
factors, category lists) is allowed to influence how training data was prepared.

Saves:
  - preprocessor.joblib        the fitted pipeline, reusable by every modelling lead
  - X_train.csv / y_train.csv  (and same for val, test) — ready-to-model arrays
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA = PROJECT_ROOT / "data" / "processed"

NUMERIC_FEATURES = [
    "total_revenue", "total_expenditure", "operating_surplus",
    "cash_coverage_ratio", "collection_rate", "debtor_days",
    "repairs_maintenance_ratio", "irregular_expenditure_ratio", "grant_dependency",
]
CATEGORICAL_FEATURES = ["prior_audit_outcome"]

TARGET_COL = "target_next_year"
TARGET_MAP = {"not_at_risk": 0, "at_risk": 1}  # fixed, documented mapping


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def load_split(name):
    df = pd.read_csv(DATA / f"{name}.csv", low_memory=False)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].map(TARGET_MAP)
    return X, y


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print("Raw shapes -> train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

    preprocessor = build_pipeline()

    # FIT only on training data — this is the one line that keeps the pipeline leakage-safe
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    print("\nTransformed feature count:", len(feature_names))
    print("Feature names:", list(feature_names))

    joblib.dump(preprocessor, DATA / "preprocessor.joblib")
    print(f"\nSaved fitted preprocessor to {DATA / 'preprocessor.joblib'}")

    for name, Xt, y in [("train", X_train_t, y_train), ("val", X_val_t, y_val), ("test", X_test_t, y_test)]:
        pd.DataFrame(Xt, columns=feature_names).to_csv(DATA / f"X_{name}.csv", index=False)
        y.to_csv(DATA / f"y_{name}.csv", index=False)
        print(f"Saved X_{name}.csv, y_{name}.csv  (shape {Xt.shape})")

    print("\nClass balance check after target encoding (0=not_at_risk, 1=at_risk):")
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"  {name}: {y.value_counts(normalize=True).round(2).to_dict()}")

    print("\nNote on class imbalance: the target is only mildly imbalanced "
          "(roughly 55/45), so class_weight='balanced' at the model-training "
          "step is a reasonable, simpler alternative to oversampling (e.g. SMOTE) "
          "— document this choice in Q2.4, and whichever modelling lead trains "
          "each model should pass class_weight='balanced' (or the neural-network "
          "equivalent) rather than leaving it unaddressed.")


if __name__ == "__main__":
    main()
