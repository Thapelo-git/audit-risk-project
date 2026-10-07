"""
Q2.3 train/validation/test split for modeling_dataset.csv.
Run from the project root:  python src/split_q23_v2.py

Design choice: TIME-BASED split, same reasoning as before — the model must only
ever learn from the past to predict the future, matching real deployment.

With only 240 rows total, holding out a single year for test and another for
validation leaves fairly small val/test sets (see the printed row counts below).
This is a real, defensible limitation of a small public dataset — write it into
your Q2.3 justification rather than hiding it. If val/test end up too thin once
you see the real numbers, an alternative worth considering (and mentioning as a
rejected option, which also earns marks) is year-based cross-validation: train
on years up to Y, validate on Y+1, repeat for several Y, and only touch the
final held-out year once. This script gives you the simpler single-split version
to start modelling with; say the word if you want the walk-forward version too.
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
OUT = PROJECT_ROOT / "data" / "processed"


def main():
    df = pd.read_csv(DATA, low_memory=False)

    years = sorted(df["financial_year_end.year"].unique())
    print("Years present (feature year t):", years)

    if len(years) < 3:
        raise ValueError(
            f"Only {len(years)} distinct years found — need at least 3 "
            f"(train/val/test) for a time-based split. Check modeling_dataset.csv."
        )

    test_year = years[-1]
    val_year = years[-2]
    train_years = years[:-2]

    train = df[df["financial_year_end.year"].isin(train_years)]
    val = df[df["financial_year_end.year"] == val_year]
    test = df[df["financial_year_end.year"] == test_year]

    print(f"\nTrain years: {train_years} -> {len(train)} rows")
    print(f"Validation year: {val_year} -> {len(val)} rows")
    print(f"Test year: {test_year} -> {len(test)} rows")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{name} class balance:")
        print(split["target_next_year"].value_counts())
        if len(split) > 0 and split["target_next_year"].nunique() < 2:
            print(f"  WARNING: {name} has only one class present — this split may not be usable as-is")

    train.to_csv(OUT / "train.csv", index=False)
    val.to_csv(OUT / "val.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"\nSaved train.csv, val.csv, test.csv to {OUT}/")


if __name__ == "__main__":
    main()
