"""Charts for the README: model ROC comparison, calibration, the profit-vs-cutoff
decision curve, and permutation feature importance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ASSETS = "assets"
INK = "#23252B"
SLATE = "#5C6773"
STEEL = "#2F6F8F"
BURG = "#8C2F39"
AMBER = "#C08A2D"
GREEN = "#2E7D5B"
GRID = "#E4E7EC"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "text.color": INK, "xtick.color": SLATE,
    "ytick.color": SLATE, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, name):
    fig.tight_layout()
    p = f"{ASSETS}/{name}"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def roc_comparison(roc_data):
    """roc_data: list of (name, fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    colors = [STEEL, BURG, AMBER]
    for (name, fpr, tpr, auc), c in zip(roc_data, colors):
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Model comparison — ROC curves (test set)", fontsize=13,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(color=GRID, lw=0.8)
    return _save(fig, "01_roc_comparison.png")


def calibration(rel_raw, rel_cal, brier_raw, brier_cal):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1, label="Perfect")
    ax.plot(rel_raw["mean_predicted"], rel_raw["fraction_positive"], color=BURG,
            lw=2, marker="o", ms=4, label=f"Raw (Brier {brier_raw:.3f})")
    ax.plot(rel_cal["mean_predicted"], rel_cal["fraction_positive"], color=STEEL,
            lw=2, marker="o", ms=4, label=f"Calibrated (Brier {brier_cal:.3f})")
    ax.set_xlabel("Predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (reliability curve)", fontsize=13,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    return _save(fig, "02_calibration.png")


def profit_curve(dec):
    df = dec["curve"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(df["cutoff"], df["profit"] / 1e6, color=STEEL, lw=2.3, label="Policy profit")
    ax.axhline(dec["approve_all_profit"] / 1e6, color=SLATE, ls="--", lw=1.3,
               label="Approve everyone")
    ax.axvline(dec["best_cutoff"], color=BURG, lw=1.8)
    ax.scatter([dec["best_cutoff"]], [dec["best_profit"] / 1e6], color=BURG,
               zorder=5, s=45)
    ax.annotate(f"  best cutoff {dec['best_cutoff']:.2f}\n"
                f"  approve {dec['best_approval_rate']:.0%}, "
                f"bad rate {dec['best_bad_rate']:.0%}",
                (dec["best_cutoff"], dec["best_profit"] / 1e6),
                fontsize=9, color=BURG, va="top")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.1f}M"))
    ax.set_xlabel("Approve if predicted default probability ≤ cutoff")
    ax.set_ylabel("Test-set profit")
    ax.set_title("Profit vs. approval cutoff — the decision", fontsize=13,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.grid(color=GRID, lw=0.8)
    return _save(fig, "03_profit_curve.png")


def importance(drivers, top=12):
    d = drivers[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([x["feature"] for x in d], [x["importance"] for x in d],
            xerr=[x["std"] for x in d], color=STEEL, edgecolor="white",
            error_kw={"ecolor": SLATE, "lw": 0.8})
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("What drives default risk (permutation importance)",
                 fontsize=13, fontweight="bold", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    return _save(fig, "04_feature_importance.png")
