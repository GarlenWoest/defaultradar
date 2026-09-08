# Model card & underwriting recommendation — DefaultRadar

## What it is

A credit-default model (calibrated logistic regression) plus a decision rule that
approves a loan when its predicted default probability is below a profit-maximizing
cutoff. Trained on 22,500 loans, evaluated on a held-out 7,500.

## Performance (test set)

- ROC-AUC **0.747**, PR-AUC 0.443, KS 0.373.
- Calibrated: Brier **0.144** (from 0.208 raw). Predicted probabilities can be
  read as real default rates.

## The decision

Under a **12% net margin** on repaid principal and a **70% loss-given-default**,
the break-even default probability is ~15%.

- **Approve** applicants with predicted default probability ≤ **0.15**.
- That approves **~42%** of applicants, at an approved-book default rate of **8%**
  (vs. 21% overall), for a test-set profit of ~$2.5M against a $6.7M loss if
  everyone were approved.

## What drives risk (permutation importance)

Credit score, DTI, installment-to-income, revolving utilization, loan-to-income,
prior delinquencies, recent inquiries, and 60-month term. All move risk in the
expected direction.

## Underwriting recommendations

1. **Hard-decline** applicants above the 0.15 cutoff unless offset by
   compensating factors — this is where the profit swing comes from.
2. **Price on the top drivers.** DTI and installment-to-income carry independent
   signal beyond credit score; use them in risk-based pricing.
3. **Recalibrate on a schedule.** The decision is only as good as the probability;
   monitor the reliability curve and Brier score over time.
4. **Watch for leakage on any new feature.** Only origination-time data is
   allowed; the audit in `preprocess.py` is the guardrail.

## Limitations

- Trained on synthetic (but realistically structured) data; retrain on the real
  LendingClub file before drawing lending conclusions.
- Economic assumptions are explicit and adjustable in `decision.py`.
- No fairness/adverse-impact analysis here — required before any real deployment.
