"""
Cleaning, Encoding and Scaling.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C



CAT_COLS = ["Gender", "Location", "Subscription_Type", "Payment_Method", "Contract_Type"]

PLACEHOLDERS = ["", "?", "n/a", "na", "unknown", "none", "nan"]

ALIASES = {
    "Gender": {
        "M": "Male", 
        "F": "Female"
    },
    "Location": {
        "Bangalore": "Bengaluru", 
        "Bombay": "Mumbai", 
        "Calcutta": "Kolkata"
    },
    "Payment_Method": {
        "Creditcard": "Credit Card", 
        "Debit": "Debit Card",
        "Netbanking": "Net Banking", 
        "Upi": "UPI"
    },
    "Contract_Type": {
        "Month To Month": "Month-to-Month", 
        "Month-To-Month": "Month-to-Month",
        "1 Year": "One Year", 
        "2 Year": "Two Year"
    }
}

# (min, max) valid range per column - anything outside is treated as missing and re-filled
VALID_RANGES = {
    "Age": (18, 100), 
    "Tenure_Months": (1, 120), 
    "Monthly_Charges": (100, 2000),
    "Usage_Hours": (0, 300), 
    "Customer_Satisfaction": (1, 5), 
    "Support_Tickets": (0, 30),
    "Discount_Received": (0, 50), 
    "Login_Frequency": (0, 100), 
    "Last_Login_Days": (0, 365),
    "Complaints": (0, 30),
}

INT_COLS = ["Age", "Tenure_Months", "Login_Frequency", "Last_Login_Days", "Support_Tickets", "Complaints", "Discount_Received", "Customer_Satisfaction", "Churn"]


def _to_number(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def clean_data(raw: pd.DataFrame):
    """Return (clean_df, log). Order matters: text -> types -> invalid -> duplicates -> impute."""
    
    df = raw.copy()
    log = {"rows_raw": len(df)}

    # 1. text standardisation + fake missing values
    for c in CAT_COLS:
        s = df[c].astype("string").str.strip()
        s = s.where(~s.str.lower().isin(PLACEHOLDERS), pd.NA)
        df[c] = s.str.title().replace(ALIASES.get(c, {}))
    
    # 2. numbers stored as text
    for c in ["Age", "Total_Charges"]:
        df[c] = _to_number(df[c])
    
    # 3. invalid values -> NaN
    for c, (lo, hi) in VALID_RANGES.items():
        bad = (df[c] < lo) | (df[c] > hi)
        log[f"invalid_{c}"] = int(bad.sum())
        df.loc[bad, c] = np.nan
    
    # 4. duplicates (before imputing so medians are not skewed)
    log["exact_duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df["_missing"] = df.isna().sum(axis=1)
    log["repeated_ids"] = int(df["Customer_ID"].duplicated().sum())
    df = (df.sort_values(["Customer_ID", "_missing"]).drop_duplicates("Customer_ID", keep="first")
            .drop(columns="_missing").sort_values("Customer_ID").reset_index(drop=True))
    
    # 5. missing values
    for c in CAT_COLS:
        df[c] = df[c].fillna(df[c].mode()[0]).astype(str)
    
    df["Monthly_Charges"] = df["Monthly_Charges"].fillna(
        df.groupby("Subscription_Type")["Monthly_Charges"].transform("median")
    )
    
    for c in ["Age", "Tenure_Months", "Usage_Hours", "Login_Frequency", "Last_Login_Days", "Support_Tickets", "Discount_Received", "Complaints"]:
        df[c] = df[c].fillna(df[c].median())
    
    df["Customer_Satisfaction"] = df["Customer_Satisfaction"].fillna(df["Customer_Satisfaction"].mode()[0])
    
    expected = df["Monthly_Charges"] * df["Tenure_Months"]
    ratio = df["Total_Charges"] / expected
    illogical = (ratio < 0.5) | (ratio > 1.1)
    log["illogical_total_charges"] = int(illogical.sum())
    df.loc[illogical, "Total_Charges"] = np.nan
    df["Total_Charges"] = df["Total_Charges"].fillna((expected * 0.96).round(2))
    
    # 6. final types
    for c in INT_COLS:
        df[c] = df[c].round().astype(int)
    log["rows_clean"] = len(df)
    log["remaining_nan"] = int(df.isna().sum().sum())
    return df, log


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """
    scale=True  -> StandardScaler on numeric columns (needed by Logistic Regression and KNN)
    scale=False -> numeric columns pass through unchanged (trees do not need scaling)
    """

    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", C.NUMERIC_FEATURES),
        ("ord", OrdinalEncoder(categories=list(C.ORDINAL_FEATURES.values())), list(C.ORDINAL_FEATURES)),
        ("nom", OneHotEncoder(handle_unknown="ignore"), C.NOMINAL_FEATURES),
        ("bin", "passthrough", C.BINARY_FEATURES),
    ])

