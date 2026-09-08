"""Which features drive the predictions. Uses permutation importance, which is
model-agnostic (works for the logistic model or the trees) and measures the real
drop in ROC-AUC when a feature is shuffled -- honest about what the model relies
on. (SHAP is a drop-in for tree models via the `shap` package.)"""
from sklearn.inspection import permutation_importance


def permutation_drivers(model, X_test, y_test, n_repeats=8, seed=42):
    r = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=seed, n_jobs=1)
    drivers = sorted(
        [{"feature": f, "importance": float(m), "std": float(s)}
         for f, m, s in zip(X_test.columns, r.importances_mean, r.importances_std)],
        key=lambda d: d["importance"], reverse=True)
    return drivers
