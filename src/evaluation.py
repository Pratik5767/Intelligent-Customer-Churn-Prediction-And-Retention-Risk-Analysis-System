"""
Evaluation, Comparison and Feature importance + the PDF report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from . import config as C



# fixed categorical order from the validated palette: blue, orange, aqua, yellow
SERIES = [C.BLUE, C.ORANGE, C.AQUA, C.YELLOW]

plt.rcParams.update({
    "axes.spines.top": False, 
    "axes.spines.right": False, 
    "axes.edgecolor": C.GRID,
    "axes.labelcolor": C.MUTED, 
    "xtick.color": C.MUTED, 
    "ytick.color": C.MUTED, 
    "font.size": 10
})


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }


def comparison_table(fitted: dict, X_test, y_test) -> pd.DataFrame:
    rows = {name: evaluate(m, X_test, y_test) for name, m in fitted.items()}
    return pd.DataFrame(rows).T


def select_best(table: pd.DataFrame, exclude=("Baseline (everyone stays)",)):
    """Business-driven selection: among models that catch at least MIN_RECALL of churners,
    take the best ROC-AUC (best at ranking customers by risk). Not simply the highest accuracy."""
    
    t = table.drop(index=[e for e in exclude if e in table.index])
    eligible = t[t["Recall"] >= C.MIN_RECALL_FOR_SELECTION]
    pool = eligible if len(eligible) else t
    best = pool["ROC-AUC"].astype(float).idxmax()
    return best, len(eligible) > 0


# ------------------------------- figures -------------------------------
def fig_roc(fitted: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    
    for color, (name, m) in zip(SERIES, [(n, m) for n, m in fitted.items() if "Baseline" not in n]):
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc:.3f})")
    
    ax.plot([0, 1], [0, 1], color=C.MUTED, ls="--", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set", loc="left", fontweight="bold", color=C.INK)
    ax.yaxis.grid(True, color=C.GRID)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    
    return fig


def fig_confusions(fitted: dict, X_test, y_test):
    items = list(fitted.items())
    fig, axes = plt.subplots(1, len(items), figsize=(3.4 * len(items), 3.6))
    
    for ax, (name, m) in zip(np.atleast_1d(axes), items):
        cm = confusion_matrix(y_test, (m.predict_proba(X_test)[:, 1] >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else C.INK)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stay", "Churn"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Stay", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=10, color=C.INK)
    
    fig.tight_layout()
    
    return fig


def importance_tables(fitted: dict, best_name: str, X_test, y_test) -> dict:
    """
    (a) built-in importances of tree models, 
    (b) permutation importance for the selected model (works for ANY model and reports in business-friendly column names).
    """

    out = {}
    
    for name in ("Random Forest", "Decision Tree"):
        pipe = fitted[name]
        names = pipe.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        out[f"{name} (built-in)"] = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    
    pi = permutation_importance(fitted[best_name], X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=C.RANDOM_STATE, n_jobs=-1)
    out[f"{best_name} (permutation)"] = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
    
    if best_name == "Logistic Regression":
        pipe = fitted[best_name]
        names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
        out["Logistic Regression (coefficients)"] = pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values(key=abs, ascending=False)
    
    return out


def fig_importance(tables: dict, best_name: str, top: int = 10):
    perm = tables[f"{best_name} (permutation)"].head(top)[::-1]
    rf = tables["Random Forest (built-in)"].head(top)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    for ax, s, title, color in [
        (axes[0], perm, f"{best_name}: permutation importance\n(drop in ROC-AUC when shuffled)", SERIES[0]),
        (axes[1], rf, "Random Forest: built-in importance", SERIES[2])
    ]:
        ax.barh(s.index, s.values, color=color, height=0.62)
        ax.set_title(title, loc="left", fontweight="bold", color=C.INK, fontsize=10)
        ax.xaxis.grid(True, color=C.GRID)
        ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig


def fig_risk_overview(report: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(report["Churn_Probability"], bins=30, color=SERIES[0])
    
    for x in (C.RISK_LOW_MAX, C.RISK_HIGH_MIN):
        ax.axvline(x, color=SERIES[1], ls="--", lw=1.5)
    
    ax.set_title("Churn probability across customers", loc="left", fontweight="bold", color=C.INK)
    ax.set_xlabel("Churn probability (dashed = risk thresholds)"); ax.set_ylabel("Customers")
    ax = axes[1]
    order = ["Low", "Medium", "High"]
    g = report.groupby("Risk_Level")["Actual_Churn"].mean().reindex(order) * 100
    bars = ax.bar(order, g.values, color=SERIES[0], width=0.55)
    
    for b, v in zip(bars, g.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%", ha="center", color=C.INK)
    
    ax.set_title("Actual churn rate inside each risk level", loc="left", fontweight="bold", color=C.INK)
    ax.set_ylabel("Actual churn rate (%)")
    ax.yaxis.grid(True, color=C.GRID)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig


def build_pdf(path, table: pd.DataFrame, cv: pd.DataFrame, best_name: str, rationale: str, figs: dict, report_summary: dict):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        
        ax = fig.add_axes([0.07, 0.05, 0.86, 0.9])
        ax.axis("off")
        ax.text(0, 1.0, "Customer Churn Model Report", fontsize=20, fontweight="bold", va="top", color=C.INK)
        ax.text(0, 0.955, f"Selected model: {best_name}", fontsize=12, va="top", color=C.MUTED)
        
        t = table[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].astype(float).round(3)
        tb = ax.table(
            cellText=t.reset_index().values, 
            colLabels=["Model"] + list(t.columns),
            colWidths=[0.30] + [0.14] * 5, 
            loc="upper center", 
            bbox=[0, 0.62, 1, 0.28]
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.5)
        
        ax.text(0, 0.57, "Why this model", fontsize=12, fontweight="bold", va="top", color=C.INK)
        ax.text(0, 0.54, rationale, fontsize=9.5, va="top", wrap=True, color=C.INK)
        y = 0.38
        ax.text(0, y, "Accuracy is not enough", fontsize=12, fontweight="bold", va="top", color=C.INK)
        ax.text(
            0, y - 0.03, 
            f"The 'everyone stays' baseline scores {table.loc['Baseline (everyone stays)', 'Accuracy']:.0%} "
            "accuracy yet finds 0% of churners.\n"
            "Judge models by recall, precision, F1 and ROC-AUC.", 
            fontsize=9.5, 
            va="top", 
            color=C.INK
        )
        ax.text(0, 0.26, "Risk scoring on all customers (out-of-fold probabilities)", fontsize=12, fontweight="bold", va="top", color=C.INK)
        ax.text(0, 0.23, "\n".join(f"{k}: {v}" for k, v in report_summary.items()), fontsize=9.5, va="top", color=C.INK)
        pdf.savefig(fig)
        plt.close(fig)
        
        for f in figs.values():
            pdf.savefig(f)
