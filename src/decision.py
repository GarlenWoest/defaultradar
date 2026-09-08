"""Turn default probabilities into a lending decision. A model score is only
useful once it's a rule: approve a loan when its predicted default probability is
below a cutoff. We pick the cutoff that maximizes expected profit on the test set
using each loan's real economics, not a generic 0.5 threshold.

Economics per funded loan (share of principal):
  repaid  -> gain = loan_amount * MARGIN     [net lifetime margin]
  default -> loss = loan_amount * LGD         [loss given default]

Break-even default probability = MARGIN / (MARGIN + LGD). With the values below
that's ~15%, so the profit-max rule declines loans whose predicted default
probability is meaningfully above that.
"""
import numpy as np
import pandas as pd

MARGIN = 0.12         # net profit on a repaid loan, as a share of principal
LGD = 0.70            # loss given default (net of recoveries), share of principal


def loan_economics(X):
    gain = X["loan_amount"] * MARGIN
    loss = X["loan_amount"] * LGD
    return gain.to_numpy(), loss.to_numpy()


def break_even_probability():
    return MARGIN / (MARGIN + LGD)


def profit_curve(X_test, y_test, proba, steps=51):
    gain, loss = loan_economics(X_test)
    y = y_test.to_numpy()
    approve_all = np.where(y == 1, -loss, gain).sum()

    rows = []
    for cut in np.linspace(0.02, 0.98, steps):
        approved = proba <= cut
        pnl = np.where(y[approved] == 1, -loss[approved], gain[approved]).sum()
        rows.append({
            "cutoff": float(cut),
            "profit": float(pnl),
            "approval_rate": float(approved.mean()),
            "bad_rate_of_approved": float(y[approved].mean()) if approved.any() else 0.0,
        })
    df = pd.DataFrame(rows)
    best = df.loc[df["profit"].idxmax()]
    return {
        "curve": df,
        "best_cutoff": float(best["cutoff"]),
        "best_profit": float(best["profit"]),
        "best_approval_rate": float(best["approval_rate"]),
        "best_bad_rate": float(best["bad_rate_of_approved"]),
        "approve_all_profit": float(approve_all),
        "uplift_vs_approve_all": float(best["profit"] - approve_all),
    }
