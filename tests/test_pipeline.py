"""Tests for the credit-risk pipeline. Fast: unit checks plus one small model
smoke test on a subsample."""
import os
import sys
import warnings

sys.path.insert(0, "src")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import decision as dec
from evaluate import evaluate, ks_statistic
from preprocess import LEAKAGE, prepare, split


def test_leakage_dropped():
    df, numeric, dropped = prepare()
    assert set(dropped) == set(LEAKAGE)
    for col in LEAKAGE:
        assert col not in df.columns          # never train on post-origination data


def test_engineered_features_exist():
    df, numeric, _ = prepare()
    for col in ["loan_to_income", "installment_to_income", "high_utilization"]:
        assert col in df.columns


def test_break_even_probability():
    # MARGIN 0.12 / (0.12 + 0.70) ~ 0.146
    assert abs(dec.break_even_probability() - 0.146) < 0.01


def test_ks_in_range():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.3])
    assert 0 <= ks_statistic(y, s) <= 1


def test_profit_curve_beats_or_matches_approve_all():
    df, numeric, _ = prepare()
    X = df.drop(columns=["default"]).head(2000)
    y = df["default"].head(2000)
    proba = np.clip((30 - (X["credit_score"] - 600) / 10) / 40, 0, 1)  # crude score
    out = dec.profit_curve(X, y, proba.to_numpy())
    assert 0 < out["best_cutoff"] < 1
    assert out["best_profit"] >= out["approve_all_profit"]  # optimizing can't hurt


def test_model_smoke():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from features import build_preprocessor
    df, numeric, _ = prepare()
    df = df.sample(3000, random_state=0)
    X_tr, X_te, y_tr, y_te = split(df)
    pipe = Pipeline([("pre", build_preprocessor(numeric)),
                     ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
    pipe.fit(X_tr, y_tr)
    assert evaluate(pipe, X_te, y_te)["roc_auc"] > 0.65


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
