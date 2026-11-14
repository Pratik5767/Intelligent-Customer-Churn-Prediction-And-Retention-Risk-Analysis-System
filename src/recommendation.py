"""
Retention priority and Recommendation engine. Pure business rules.
"""

import numpy as np
import pandas as pd

from . import config as C



# Ordered: the FIRST rule that fires is the "key risk factor".
# (column-test, label, recommended action)
ISSUE_RULES = [
    (lambda r: r["Complaints"] >= C.HIGH_COMPLAINTS or r["Support_Tickets"] >= C.HIGH_TICKETS,
     "Complaints / support issues", "Customer support intervention"),
    (lambda r: r["Customer_Satisfaction"] <= C.LOW_SATISFACTION,
     "Low satisfaction", "Satisfaction recovery call"),
    (lambda r: r["Last_Login_Days"] > C.INACTIVITY_DAYS,
     "Recent inactivity", "Re-engagement campaign"),
    (lambda r: r["Usage_Hours"] < C.LOW_USAGE_HOURS,
     "Low usage", "Engagement campaign"),
    (lambda r: r["Monthly_Charges"] >= C.HIGH_CHARGES,
     "High charges", "Pricing / discount review"),
    (lambda r: r["Contract_Type"] == "Month-to-Month",
     "No long-term contract", "Contract upgrade offer"),
]


def key_risk_factor(row) -> tuple:
    for test, label, action in ISSUE_RULES:
        if test(row):
            return label, action
    return "Stable engagement", "No action"


def value_cutoffs(train_df: pd.DataFrame) -> dict:
    v = train_df[C.VALUE_COLUMN]
    return {"low": float(v.quantile(C.VALUE_LOW_QUANTILE)), "high": float(v.quantile(C.VALUE_HIGH_QUANTILE))}


def value_tier(values, cutoffs: dict):
    return np.select([values >= cutoffs["high"], values >= cutoffs["low"]], ["High", "Medium"], default="Low")


def risk_level(prob, low_max: float = None, high_min: float = None):
    low_max = C.RISK_LOW_MAX if low_max is None else low_max
    high_min = C.RISK_HIGH_MIN if high_min is None else high_min
    return np.select([np.asarray(prob) >= high_min, np.asarray(prob) >= low_max], ["High", "Medium"], default="Low")


def retention_priority(risk: str, value: str) -> str:
    table = {
        ("High", "High"): "Priority 1",
        ("High", "Medium"): "Priority 2",
        ("Medium", "High"): "Priority 3",
        ("High", "Low"): "Priority 4",
        ("Medium", "Medium"): "Priority 4",
    }
    if risk == "Low":
        return "No action"
    return table.get((risk, value), "Priority 5")


def contact_action(risk: str, issue_action: str) -> str:
    """What the retention team should actually do, combining risk level and the issue-specific action."""
    
    if risk == "High":
        return f"Retention team contact - {issue_action}" if issue_action != "No action" else "Retention team contact"
    if risk == "Medium":
        return issue_action if issue_action != "No action" else "Monitor"
    
    return "No action"

