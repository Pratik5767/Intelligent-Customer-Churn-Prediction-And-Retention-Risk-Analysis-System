"""
Train/Test Split, Models training and Persistence.
"""

import json

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from . preprocessing import make_preprocessor



def split_data(df: pd.DataFrame):
    """80/20 split. stratify=y keeps the 16% churn rate identical in both parts."""
    
    X, y = df[C.ALL_FEATURES], df[C.TARGET]
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)


def get_models() -> dict:
    """
    Every model is a Pipeline (preprocess -> classifier), so the encoder and scaler
    are fitted on training data only. Scaling is on for distance/gradient models, off for trees.
    class_weight='balanced' makes mistakes on the rare churn class cost more.
    """
    
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs))
        ]),
        "Decision Tree": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, class_weight="balanced", random_state=rs))
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=8, class_weight="balanced", n_jobs=-1, random_state=rs))
        ]),
        "KNN": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", KNeighborsClassifier(n_neighbors=25))
        ])
    }


def get_baseline() -> Pipeline:
    """'Predict everyone stays' - the useless-but-accurate model from the project brief."""
    return Pipeline([
        ("prep", make_preprocessor(scale=False)),
        ("clf", DummyClassifier(strategy="most_frequent"))
    ])


def cross_validate_models(models: dict, X, y, cv: int = 5) -> pd.DataFrame:
    """5-fold ROC-AUC on the TRAINING data only: how stable is each model?"""
    
    skf = StratifiedKFold(cv, shuffle=True, random_state=C.RANDOM_STATE)
    rows = []
    
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        rows.append({"Model": name, "CV_AUC_mean": s.mean(), "CV_AUC_std": s.std()})
    
    return pd.DataFrame(rows).set_index("Model")


def make_final_pipeline(best_name: str) -> Pipeline:
    """
    Selected model + probability calibration.
    'balanced' class weights inflate probabilities, so we recalibrate (sigmoid) to make
    "0.30" really mean "about 30% of such customers churn" - required for risk bands.
    """
    
    base = get_models()[best_name]
    return Pipeline([
        ("prep", base.named_steps["prep"]),
        ("clf", CalibratedClassifierCV(base.named_steps["clf"], method="sigmoid", cv=5))
    ])


def save_artifacts(final_pipe: Pipeline, meta: dict):
    """Preprocessor and model are saved separately, plus metadata (defaults, cut-offs)."""
    
    C.MODELS_DIR.mkdir(exist_ok=True)
    
    joblib.dump(final_pipe.named_steps["prep"], C.MODELS_DIR / "preprocessing.pkl")
    joblib.dump(final_pipe.named_steps["clf"], C.MODELS_DIR / "churn_model.pkl")
    
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

