"""
Customer behaviour analysis and feature engineering.
"""

import numpy as np
import pandas as pd

from . import config as C



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Works on one customer or many. Uses fixed caps from config (no data min/max)."""
    
    out = df.copy()
    # Customer lifetime value proxy: what this customer has paid so far
    out["Customer_Lifetime_Value"] = (out["Monthly_Charges"] * out["Tenure_Months"]).round(2)

    # Forward-looking value used for retention priority (not a model input)
    out["Annual_Revenue_At_Risk"] = (out["Monthly_Charges"] * 12).round(2)

    # Engagement score 0-100: logins + usage + recency of last login
    w = C.ENGAGEMENT_WEIGHTS
    logins = np.clip(out["Login_Frequency"] / C.LOGIN_CAP, 0, 1)
    usage = np.clip(out["Usage_Hours"] / C.USAGE_CAP, 0, 1)
    recency = 1 - np.clip(out["Last_Login_Days"] / C.RECENCY_CAP_DAYS, 0, 1)
    out["Engagement_Score"] = (100 * (w["logins"] * logins + w["usage"] * usage + w["recency"] * recency)).round(1)

    # Support risk: complaints are worse than tickets, so they are weighted higher
    out["Support_Risk"] = out["Support_Tickets"] + C.COMPLAINT_WEIGHT * out["Complaints"]

    # Inactivity flag
    out["Inactivity_Flag"] = (out["Last_Login_Days"] > C.INACTIVITY_DAYS).astype(int)
    
    return out


def add_warning_signals(df: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 column per warning signal, plus their count."""
    
    out = df.copy()
    out["Sig_Low_Usage"] = (out["Usage_Hours"] < C.LOW_USAGE_HOURS).astype(int)
    out["Sig_High_Complaints"] = (out["Complaints"] >= C.HIGH_COMPLAINTS).astype(int)
    out["Sig_High_Tickets"] = (out["Support_Tickets"] >= C.HIGH_TICKETS).astype(int)
    out["Sig_Low_Satisfaction"] = (out["Customer_Satisfaction"] <= C.LOW_SATISFACTION).astype(int)
    out["Sig_Recent_Inactivity"] = (out["Last_Login_Days"] > C.INACTIVITY_DAYS).astype(int)
    
    sig_cols = [c for c in out.columns if c.startswith("Sig_")]
    out["Warning_Signals"] = out[sig_cols].sum(axis=1)
    
    return out


def customer_based_group_analysis(signals: pd.Series) -> pd.Series:
    """High / Medium / Low purely from business rules (no ML)."""
    
    return pd.Series(
        np.select([signals >= C.RULE_HIGH_MIN_SIGNALS, 
                   signals >= C.RULE_MEDIUM_MIN_SIGNALS],
                  ["High", "Medium"], default="Low"), 
                  index=signals.index, name="Rule_Risk_Group"
                )

