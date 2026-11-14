import pandas as pd
from . import config as C



def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.DATA_RAW)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(C.DATA_CLEAN)


def load_features() -> pd.DataFrame:
    return pd.read_csv(C.DATA_FEATURES)
