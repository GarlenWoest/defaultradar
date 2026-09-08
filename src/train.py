"""Compare models with cross-validation, tune the winner, and fit the final
pipeline. Class imbalance is handled with balanced class weights (SMOTE is an
easy swap but adds a dependency).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from features import build_preprocessor

# benign "matmul" RuntimeWarnings from macOS Accelerate BLAS
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

CV = StratifiedKFold(5, shuffle=True, random_state=42)


def model_zoo():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=6,
            class_weight="balanced", random_state=42),
    }


def compare(X, y, numeric):
    pre = build_preprocessor(numeric)
    rows = []
    for name, clf in model_zoo().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        scores = cross_validate(pipe, X, y, cv=CV,
                                scoring=["roc_auc", "average_precision"],
                                n_jobs=1)
        rows.append({
            "model": name,
            "roc_auc": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "pr_auc": scores["test_average_precision"].mean(),
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def tune(X, y, numeric):
    """Small randomized search over the gradient-boosting model."""
    pre = build_preprocessor(numeric)
    pipe = Pipeline([("pre", pre),
                     ("clf", HistGradientBoostingClassifier(
                         class_weight="balanced", random_state=42))])
    grid = {
        "clf__max_iter": [200, 300, 400],
        "clf__learning_rate": [0.03, 0.06, 0.1],
        "clf__max_depth": [4, 6, 8, None],
        "clf__l2_regularization": [0.0, 0.5, 1.0],
        "clf__max_leaf_nodes": [15, 31, 63],
    }
    search = RandomizedSearchCV(pipe, grid, n_iter=15, cv=CV, scoring="roc_auc",
                                n_jobs=-1, random_state=42)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


if __name__ == "__main__":
    from preprocess import prepare, split
    df, numeric, _ = prepare()
    X_tr, X_te, y_tr, y_te = split(df)
    print(compare(X_tr, y_tr, numeric).round(4).to_string(index=False))
