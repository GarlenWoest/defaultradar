"""Load the loans, drop leakage, engineer a few features, and split.

The leakage audit matters: columns like `recoveries` or `last_pymnt_amnt` are only
known AFTER a loan has run its course, so training on them gives a fantastic score
that collapses in production. We drop them explicitly and loudly.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = "data/loans.csv"
TARGET = "default"

# Known only after origination -> would leak the outcome. Never train on these.
LEAKAGE = ["recoveries", "total_rec_prncp", "last_pymnt_amnt"]

CATEGORICAL = ["grade", "home_ownership", "purpose", "term"]


def load(path=DATA):
    if not __import__("os").path.exists(path):
        import generate_data
        generate_data.main()
    return pd.read_csv(path)


def engineer(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["installment_to_income"] = (df["installment"] * 12) / df["annual_income"]
    df["high_utilization"] = (df["revol_util"] > 80).astype(int)
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)
    return df


def prepare(path=DATA):
    """Return (df, numeric_cols, dropped_leakage)."""
    df = load(path)
    dropped = [c for c in LEAKAGE if c in df.columns]
    df = df.drop(columns=dropped)
    df = engineer(df)
    numeric = [c for c in df.columns
               if c not in CATEGORICAL + [TARGET] and df[c].dtype != "object"]
    return df, numeric, dropped


def split(df, test_size=0.25, seed=42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=seed)


if __name__ == "__main__":
    df, num, dropped = prepare()
    print(f"[preprocess] rows={len(df):,}  default rate={df[TARGET].mean():.1%}")
    print(f"[preprocess] dropped leakage columns: {dropped}")
    print(f"[preprocess] {len(num)} numeric + {len(CATEGORICAL)} categorical features")
