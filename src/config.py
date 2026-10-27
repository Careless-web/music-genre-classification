"""Central configuration: all paths and constants live here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "database"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Real GTZAN feature file (download from Kaggle, see README)
CSV_PATH = DATA_DIR / "features_30_sec.csv"
# Only used to test the code when the real file is missing
SYNTHETIC_CSV_PATH = DATA_DIR / "synthetic_features_30_sec.csv"

DB_PATH = DB_DIR / "music.db"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Columns that are not audio features
NON_FEATURE_COLS = ["id", "filename", "length", "label"]

for _d in (DATA_DIR, DB_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
