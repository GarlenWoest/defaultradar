"""Make a realistic consumer-loan dataset shaped like LendingClub, so the project
runs without a Kaggle download. Default risk is driven by credit score, rate,
DTI, utilization and delinquencies (plus noise), so a model has real signal to
find but can't hit a suspicious AUC of 1.

The file also includes a few POST-ORIGINATION columns (recoveries,
last_pymnt_amnt, total_rec_prncp) on purpose -- they leak the outcome and the
preprocessing step drops them, which is the leakage audit a real model needs.

To use the real data instead: download the LendingClub 'accepted' loans CSV,
keep the origination-time columns, and point preprocess.py at it.
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2024)
OUT = "data/loans.csv"

PURPOSES = ["debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "medical", "car", "small_business", "other"]
PURPOSE_RISK = {"debt_consolidation": 0.1, "credit_card": 0.15,
                "home_improvement": -0.1, "major_purchase": 0.0,
                "medical": 0.2, "car": -0.05, "small_business": 0.5, "other": 0.1}
HOME = ["RENT", "MORTGAGE", "OWN"]


def _installment(principal, annual_rate, term_months):
    r = annual_rate / 12
    return principal * r / (1 - (1 + r) ** -term_months)


def build(n=30000):
    credit_score = np.clip(RNG.normal(690, 55, n), 520, 830)
    annual_income = np.clip(RNG.lognormal(11.0, 0.55, n), 15000, 400000)
    dti = np.clip(RNG.normal(18, 9, n), 0, 45)
    revol_util = np.clip(RNG.normal(52, 24, n), 0, 130)
    emp_length = RNG.integers(0, 11, n)
    delinq_2yrs = RNG.poisson(0.3, n)
    inq_last_6mths = RNG.poisson(0.9, n)
    open_acc = np.clip(RNG.normal(11, 5, n), 1, 40).astype(int)
    term = RNG.choice([36, 60], n, p=[0.7, 0.3])
    home = RNG.choice(HOME, n, p=[0.4, 0.45, 0.15])
    purpose = RNG.choice(PURPOSES, n,
                         p=_norm([35, 20, 10, 8, 7, 6, 4, 10]))
    loan_amount = np.clip(RNG.normal(15000, 8500, n), 1000, 40000)

    # latent risk z (higher = riskier), standardized-ish
    z = (
        -1.10 * (credit_score - 690) / 55
        + 0.55 * (dti - 18) / 9
        + 0.45 * (revol_util - 52) / 24
        + 0.45 * delinq_2yrs
        + 0.30 * inq_last_6mths
        + 0.35 * (term == 60)
        - 0.25 * (annual_income > 90000)
        + 0.30 * (loan_amount / annual_income)
        + np.array([PURPOSE_RISK[p] for p in purpose])
    )
    # grade + rate reflect the lender's own risk read (predictive, origination-time)
    grade_idx = np.clip(((z + 2) / 0.8).astype(int), 0, 6)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    int_rate = np.round(0.06 + 0.028 * grade_idx + RNG.normal(0, 0.006, n), 4)
    int_rate = np.clip(int_rate, 0.05, 0.31)

    # generous noise so the achievable AUC lands near real credit models (~0.70),
    # not an unrealistically clean separation
    logit = -2.75 + 0.95 * z + RNG.normal(0, 1.6, n)
    p_default = 1 / (1 + np.exp(-logit))
    default = (RNG.random(n) < p_default).astype(int)

    installment = _installment(loan_amount, int_rate, term).round(2)

    # post-origination LEAKAGE columns (dropped by preprocess.py)
    recoveries = np.where(default == 1, RNG.uniform(0, 3000, n), 0).round(2)
    frac_paid = np.where(default == 1, RNG.uniform(0.0, 0.5, n),
                         RNG.uniform(0.6, 1.0, n))
    total_rec_prncp = (loan_amount * frac_paid).round(2)
    last_pymnt_amnt = np.where(default == 1, RNG.uniform(0, 300, n),
                               installment).round(2)

    df = pd.DataFrame({
        "loan_amount": loan_amount.round(2), "term": term, "int_rate": int_rate,
        "grade": grade, "installment": installment,
        "annual_income": annual_income.round(0), "dti": dti.round(2),
        "credit_score": credit_score.round(0), "revol_util": revol_util.round(1),
        "emp_length": emp_length, "home_ownership": home, "purpose": purpose,
        "open_acc": open_acc, "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        # leakage (post-origination) -- present on purpose:
        "recoveries": recoveries, "total_rec_prncp": total_rec_prncp,
        "last_pymnt_amnt": last_pymnt_amnt,
        "default": default,
    })
    return df


def _norm(x):
    x = np.asarray(x, float)
    return x / x.sum()


def main():
    os.makedirs("data", exist_ok=True)
    df = build()
    df.to_csv(OUT, index=False)
    print(f"[generate_data] {len(df):,} loans -> {OUT}")
    print(f"[generate_data] default rate = {df['default'].mean():.1%}")


if __name__ == "__main__":
    main()
