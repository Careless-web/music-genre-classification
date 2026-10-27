"""Load the dataset from the database (builds it from the CSV if needed)."""
import numpy as np

from src.config import NON_FEATURE_COLS, DB_PATH
from src import database


def load_dataframe():
    """Return the full songs table. Builds the database on first use."""
    if not DB_PATH.exists():
        database.load_csv_to_db()
    try:
        return database.read_table("songs")
    except Exception:  # table missing -> rebuild
        return database.load_csv_to_db()


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def get_xy(df=None):
    """Return X (DataFrame of features), y (Series of genres), feature names."""
    df = load_dataframe() if df is None else df
    cols = feature_columns(df)
    return df[cols], df["label"], cols


def to_numpy(df=None):
    X, y, cols = get_xy(df)
    return X.to_numpy(dtype=np.float64), y.to_numpy(), cols
