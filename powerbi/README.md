# Power BI — credit-risk dashboard

`run_all.py` writes `data/exports/powerbi/scored_loans.csv`: the held-out test
loans with their features, `default_probability`, `default_actual`, and the
model's `decision` (Approve / Decline). Import that one file.

## Measures (DAX)

```DAX
Loans = COUNTROWS ( scored_loans )

Approval Rate =
DIVIDE ( CALCULATE ( [Loans], scored_loans[decision] = "Approve" ), [Loans] )

Approved Bad Rate =
CALCULATE ( AVERAGE ( scored_loans[default_actual] ),
            scored_loans[decision] = "Approve" )

Book Bad Rate = AVERAGE ( scored_loans[default_actual] )

-- expected profit using the model's economics (12% margin, 70% LGD)
Approved Profit =
SUMX (
    FILTER ( scored_loans, scored_loans[decision] = "Approve" ),
    IF ( scored_loans[default_actual] = 1,
         -0.70 * scored_loans[loan_amount],
          0.12 * scored_loans[loan_amount] )
)
```

## Suggested layout

- **KPI cards:** Approval Rate, Approved Bad Rate, Book Bad Rate, Approved Profit.
- **Risk bands:** bucket `default_probability` into deciles; bar of count with the
  actual default rate as a line — shows the score ranks risk.
- **Approve vs. decline:** clustered bar of actual default rate by `decision`
  (the approved group should be far lower).
- **Slicers:** grade, term, purpose, home_ownership.
- **Scatter:** `credit_score` vs. `default_probability`, colored by decision.

Everything works in free Power BI Desktop. Export Page 1 to `../assets/` as the
dashboard hero for the repo README.
