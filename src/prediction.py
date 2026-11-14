"""
Risk scoring and Reusable prediction system.
"""

import json

import joblib
import pandas as pd

from . import config as C
from . feature_engineering import add_features
from . recommendation import contact_action, key_risk_factor, retention_priority, risk_level, value_tier



def load_artifacts() -> dict:
    return {  
        "preprocessor": joblib.load(C.MODELS_DIR / "preprocessing.pkl"),
        "model": joblib.load(C.MODELS_DIR / "churn_model.pkl"),
        "meta": json.loads((C.MODELS_DIR / "metadata.json").read_text())
    }


def score_probabilities(df_with_features: pd.DataFrame, art: dict):
    X = art["preprocessor"].transform(df_with_features[C.ALL_FEATURES])
    return art["model"].predict_proba(X)[:, 1]


def predict_customer(info: dict, art: dict = None) -> dict:
    """
    `info` = any subset of customer fields. Missing fields fall back to the training-data median/mode, so a form with only 7 inputs still works.
    """
    
    art = art or load_artifacts()
    row = {
        **art["meta"]["defaults"], 
        **{k: v for k, v in info.items() if v is not None}
    }
    
    df = add_features(pd.DataFrame([row]))
    prob = float(score_probabilities(df, art)[0])
    risk = str(risk_level(prob))
    value = str(value_tier(df[C.VALUE_COLUMN].to_numpy(), art["meta"]["value_cutoffs"])[0])
    issue, issue_action = key_risk_factor(df.iloc[0])
    
    return {
        "Customer_ID": info.get("Customer_ID", "NEW"),
        "Churn_Probability": round(prob, 4),
        "Risk_Level": risk.upper(),
        "Customer_Value": value,
        "Retention_Priority": retention_priority(risk, value),
        "Key_Risk_Factor": issue,
        "Recommended_Action": contact_action(risk, issue_action),
    }

