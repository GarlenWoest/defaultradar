"""Test-set metrics for a fitted model: ROC-AUC, PR-AUC, the KS statistic, and
the Brier score (calibration). Plus reliability-curve data."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)


def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(model, X_test, y_test):
    p = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, p)),
        "pr_auc": float(average_precision_score(y_test, p)),
        "ks": ks_statistic(y_test, p),
        "brier": float(brier_score_loss(y_test, p)),
        "proba": p,
    }


def reliability(y_test, p, bins=10):
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=bins, strategy="quantile")
    return {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}
