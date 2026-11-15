"""
End-to-end pipeline: raw CSV -> cleaning -> features -> models -> evaluation -> risk report -> saved model.
"""

import datetime as dt

import matplotlib
matplotlib.use("Agg")   # headless: pipeline only saves figures

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config as C
from src.data_loader import load_raw
from src.evaluation import build_pdf, comparison_table, fig_confusions, fig_importance, fig_risk_overview, fig_roc, importance_tables, select_best
from src.feature_engineering import add_features, add_warning_signals, customer_based_group_analysis
from src.model_training import cross_validate_models, get_baseline, get_models, make_final_pipeline, save_artifacts, split_data
from src.recommendation import contact_action, key_risk_factor, retention_priority, risk_level, value_cutoffs, value_tier
from src.prediction import load_artifacts, predict_customer



pd.set_option("display.width", 200)
C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
C.DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)


def banner(t): 
    print(f"\n{'=' * 8} {t} {'=' * 8}")


# ---------------- Data cleaning ----------------
from src.preprocessing import clean_data
banner("cleaning")
clean, log = clean_data(load_raw())
print(log)
clean.to_csv(C.DATA_CLEAN, index=False)


# ---------------- Customer behaviour analysis ----------------
banner("rule-based behaviour groups (no ML)")
df = add_warning_signals(add_features(clean))
df["Rule_Risk_Group"] = customer_based_group_analysis(df["Warning_Signals"])
print(df.groupby("Rule_Risk_Group")["Churn"].agg(actual_churn_rate="mean", customers="count").reindex(["Low", "Medium", "High"]).round(3))


# ---------------- features engineering ----------------
banner("engineered features vs churn (correlation)")
print(df[["Customer_Lifetime_Value", "Engagement_Score", "Support_Risk", "Inactivity_Flag", C.TARGET]]
      .corr()[C.TARGET].drop(C.TARGET).round(3).to_string())
df.to_csv(C.DATA_FEATURES, index=False)


# ---------------- splitting ----------------
banner("stratified 80/20 split")
X_train, X_test, y_train, y_test = split_data(df)
print(
    f"train={len(X_train)} test={len(X_test)} | churn rate train={y_train.mean():.3f} test={y_test.mean():.3f}"
)


# ----------------Model training ----------------
banner("training")
models = get_models()
cv = cross_validate_models(models, X_train, y_train)
print("5-fold CV ROC-AUC on training data:\n", cv.round(3))

fitted = {}
for name, m in models.items():
    fitted[name] = m.fit(X_train, y_train)

baseline = get_baseline().fit(X_train, y_train)
fitted_all = {"Baseline (everyone stays)": baseline, **fitted}


# ---------------- Model evaluation and comparing ----------------
banner("comparison on the held-out test set (threshold 0.5)")
table = comparison_table(fitted_all, X_test, y_test)
print(table.astype({"TN": int, "FP": int, "FN": int, "TP": int}).round(3).to_string())
table.to_csv(C.REPORTS_DIR / "model_comparison.csv")
best_name, had_eligible = select_best(table)
best_row = table.loc[best_name]
rationale = (
    f"{best_name} has the best ROC-AUC ({best_row['ROC-AUC']:.3f}) among the models that catch at least "
    f"{C.MIN_RECALL_FOR_SELECTION:.0%} of churners (recall {best_row['Recall']:.2f}). "
    "ROC-AUC measures how well a model RANKS customers by risk, which is what the retention team "
    "needs to decide whom to call first. Accuracy was not used to choose."
)
print("\nSELECTED:", best_name, "\n", rationale)


# ---------------- feature importance ----------------
banner("feature importance")
imp = importance_tables(fitted, best_name, X_test, y_test)

for k, s in imp.items():
    print(f"\n{k} - top 8:\n", s.head(8).round(4).to_string())

pd.concat({
    k: v.reset_index().rename(columns={"index": "feature", 0: "value"}) 
    for k, v in imp.items()
}).to_csv(C.REPORTS_DIR / "feature_importance.csv")


# ---------------- scoring, priority, recommendations, report ----------------
banner("risk scoring on all customers")
X_all, y_all = df[C.ALL_FEATURES], df[C.TARGET]
final = make_final_pipeline(best_name)

# out-of-fold probabilities: every customer is scored by a model that never saw them
oof = cross_val_predict(
    final, X_all, y_all, method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE), n_jobs=-1
)[:, 1]

cut = value_cutoffs(df.loc[X_train.index])      # tier cut-offs come from training customers only
report = pd.DataFrame({
    "Customer_ID": df["Customer_ID"], 
    "Churn_Probability": oof.round(4)
})

report["Risk_Level"] = risk_level(oof)
report["Customer_Value"] = value_tier(df[C.VALUE_COLUMN].to_numpy(), cut)

issues = df.apply(key_risk_factor, axis=1)
report["Key_Risk_Factor"] = [i[0] for i in issues]
report["Retention_Priority"] = [retention_priority(r, v) for r, v in zip(report["Risk_Level"], report["Customer_Value"])]
report["Recommended_Action"] = [contact_action(r, i[1]) for r, i in zip(report["Risk_Level"], issues)]
report["Rule_Risk_Group"] = df["Rule_Risk_Group"]
report["Actual_Churn"] = y_all

order = {"Priority 1": 1, "Priority 2": 2, "Priority 3": 3, "Priority 4": 4, "Priority 5": 5, "No action": 9}
report = (
    report.assign(_o=report["Retention_Priority"].map(order)).sort_values(["_o", "Churn_Probability"], ascending=[True, False]).drop(columns="_o")
)
report.to_csv(C.REPORTS_DIR / "customer_risk_report.csv", index=False)

print(
    "Risk level distribution:\n", 
    report["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).to_string()
)

bands = report.groupby("Risk_Level").agg(
    customers=("Actual_Churn", "size"), 
    avg_probability=("Churn_Probability", "mean"), 
    actual_churn_rate=("Actual_Churn", "mean")
).reindex(["Low", "Medium", "High"])

print("\nCalibration check (predicted vs actual inside each band):\n", bands.round(3))
print("\nThreshold sensitivity for the HIGH cut-off (change RISK_HIGH_MIN in config.py):")

for t in (0.5, 0.6, 0.7):
    m = report["Churn_Probability"] >= t
    print(f"  >= {t:.0%}: {int(m.sum()):4d} customers, actual churn rate {report.loc[m, 'Actual_Churn'].mean():.0%}")

print(
    "\nRetention priorities:\n", 
    report["Retention_Priority"].value_counts().reindex(["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5", "No action"]).dropna().to_string()
)

print(
    "\nRule-based group vs ML risk level:\n", 
    pd.crosstab(report["Rule_Risk_Group"], report["Risk_Level"]).reindex(index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"]).fillna(0).astype(int)
)

contact = report["Churn_Probability"] >= C.RISK_LOW_MAX
caught = report.loc[contact, "Actual_Churn"].sum() / report["Actual_Churn"].sum()
top = report.sort_values("Churn_Probability", ascending=False).head(len(report) // 10)
summary = {
    "Customers scored": f"{len(report):,}",
    "High risk": int((report['Risk_Level'] == 'High').sum()),
    "Medium risk": int((report['Risk_Level'] == 'Medium').sum()),
    "Contacting Medium+High": f"{contact.mean():.0%} of customers -> catches {caught:.0%} of churners",
    "Top 10% by probability": f"catches {top['Actual_Churn'].sum() / report['Actual_Churn'].sum():.0%} of all churners"
}

print("\n", summary)

# figures + pdf
figs = {
    "roc": fig_roc(fitted, X_test, y_test), 
    "cm": fig_confusions(fitted_all, X_test, y_test),
    "imp": fig_importance(imp, best_name), 
    "risk": fig_risk_overview(report)
}

for k, f in figs.items():
    f.savefig(C.FIGURES_DIR / f"{k}.png", dpi=150)

build_pdf(C.REPORTS_DIR / "model_report.pdf", table, cv, best_name, rationale, figs, summary)


# ---------------- persistence (refit on ALL data after validation) ----------------
banner("saving model")

final.fit(X_all, y_all)
raw_cols = [
    c for c in C.ALL_FEATURES 
    if c not in ("Customer_Lifetime_Value", "Engagement_Score", "Support_Risk", "Inactivity_Flag")
]
defaults = {
    c: (float(df[c].median()) if c in df.select_dtypes("number") else str(df[c].mode()[0])) for c in raw_cols
}

meta = {
    "model": best_name, 
    "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    "features": C.ALL_FEATURES, 
    "defaults": defaults, 
    "value_cutoffs": cut,
    "test_metrics": {
        k: float(best_row[k]) 
        for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    }
}

save_artifacts(final, meta)
print("saved ->", sorted(p.name for p in C.MODELS_DIR.iterdir()))


# ---------------- Demo ----------------
banner("prediction demo")

art = load_artifacts()

for demo in [
    {
        "Customer_ID": "C1025", 
        "Tenure_Months": 4, 
        "Monthly_Charges": 999, 
        "Contract_Type": "Month-to-Month", 
        "Support_Tickets": 5, 
        "Complaints": 3, 
        "Usage_Hours": 6, 
        "Customer_Satisfaction": 2, 
        "Last_Login_Days": 40
    },
    {
        "Customer_ID": "C1102", 
        "Tenure_Months": 48, 
        "Monthly_Charges": 599, 
        "Contract_Type": "Two Year", 
        "Support_Tickets": 0, 
        "Usage_Hours": 35, 
        "Customer_Satisfaction": 5, 
        "Last_Login_Days": 2
    }
]:
    print(predict_customer(demo, art))
