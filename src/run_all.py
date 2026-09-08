"""Reproduce the whole project:

    python src/run_all.py

Compare models (CV) -> pick the best on a held-out test set -> calibrate its
probabilities -> find the profit-maximizing approval cutoff -> explain the drivers
-> save the model, charts, metrics, and a Power BI export.
"""
import json
import os
import warnings

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

import decision as dec_mod
import viz
from evaluate import evaluate, reliability
from explain import permutation_drivers
from features import build_preprocessor
from preprocess import prepare, split
from train import compare, model_zoo


def main():
    for d in ["models", "reports", "data/exports/powerbi"]:
        os.makedirs(d, exist_ok=True)

    df, numeric, dropped = prepare()
    X_tr, X_te, y_tr, y_te = split(df)

    # 1. cross-validated comparison
    cv_table = compare(X_tr, y_tr, numeric)

    # 2. fit each on train, score the held-out test set
    fitted, roc_data, test_rows = {}, [], []
    for name, clf in model_zoo().items():
        pipe = Pipeline([("pre", build_preprocessor(numeric)), ("clf", clf)]).fit(X_tr, y_tr)
        m = evaluate(pipe, X_te, y_te)
        fpr, tpr, _ = roc_curve(y_te, m["proba"])
        roc_data.append((name, fpr, tpr, m["roc_auc"]))
        test_rows.append({"model": name, **{k: round(m[k], 4)
                          for k in ["roc_auc", "pr_auc", "ks", "brier"]}})
        fitted[name] = pipe

    test_table = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False)
    final_name = test_table.iloc[0]["model"]
    final = fitted[final_name]
    raw = evaluate(final, X_te, y_te)

    # 3. calibrate the winner's probabilities (balanced class weights distort them)
    calibrated = CalibratedClassifierCV(
        Pipeline([("pre", build_preprocessor(numeric)), ("clf", model_zoo()[final_name])]),
        method="isotonic", cv=5).fit(X_tr, y_tr)
    cal = evaluate(calibrated, X_te, y_te)

    rel_raw = reliability(y_te, raw["proba"])
    rel_cal = reliability(y_te, cal["proba"])

    # 4. decision layer on calibrated probabilities
    dec = dec_mod.profit_curve(X_te, y_te, cal["proba"])

    # 5. drivers
    drivers = permutation_drivers(final, X_te, y_te)

    # ---- persist ----
    joblib.dump(calibrated, "models/model.joblib")

    scored = X_te.copy()
    scored["default_actual"] = y_te.to_numpy()
    scored["default_probability"] = cal["proba"].round(4)
    scored["decision"] = ["Approve" if p <= dec["best_cutoff"] else "Decline"
                          for p in cal["proba"]]
    scored.to_csv("data/exports/powerbi/scored_loans.csv", index=False)

    metrics = {
        "final_model": final_name,
        "cv_comparison": cv_table.round(4).to_dict("records"),
        "test_comparison": test_rows,
        "calibration": {"brier_raw": round(raw["brier"], 4),
                        "brier_calibrated": round(cal["brier"], 4)},
        "ks": round(raw["ks"], 4),
        "decision": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in dec.items() if k != "curve"},
        "dropped_leakage_columns": dropped,
        "top_drivers": [d["feature"] for d in drivers[:8]],
        "default_rate": round(float(df["default"].mean()), 4),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- charts ----
    viz.roc_comparison(roc_data)
    viz.calibration(rel_raw, rel_cal, raw["brier"], cal["brier"])
    viz.profit_curve(dec)
    viz.importance(drivers)

    print(f"\n===== DEFAULTRADAR =====")
    print(f"dropped leakage: {dropped}")
    print(test_table.to_string(index=False))
    print(f"\nfinal model: {final_name}")
    print(f"Brier raw {raw['brier']:.3f} -> calibrated {cal['brier']:.3f}   KS {raw['ks']:.3f}")
    print(f"profit-max cutoff {dec['best_cutoff']:.2f}: approve "
          f"{dec['best_approval_rate']:.0%}, bad rate {dec['best_bad_rate']:.0%}")
    print(f"profit uplift vs approve-all: ${dec['uplift_vs_approve_all']:,.0f}")
    print(f"top drivers: {', '.join(metrics['top_drivers'][:5])}")
    print("\ncharts -> assets/   model -> models/   metrics -> reports/metrics.json")


if __name__ == "__main__":
    main()
