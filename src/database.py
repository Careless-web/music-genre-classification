"""SQLite database layer: create, read, update and delete (CRUD).

Run:  python -m src.database
This loads the CSV into the database and demonstrates read / write / modify.
"""
import sqlite3
import pandas as pd

from src.config import CSV_PATH, SYNTHETIC_CSV_PATH, DB_PATH


def get_connection():
    return sqlite3.connect(DB_PATH)


def resolve_csv():
    """Use the real GTZAN CSV if present, else the synthetic test file."""
    if CSV_PATH.exists():
        return CSV_PATH
    if SYNTHETIC_CSV_PATH.exists():
        print("WARNING: real dataset not found, using SYNTHETIC test data.")
        return SYNTHETIC_CSV_PATH
    raise FileNotFoundError(
        f"Put the GTZAN file at {CSV_PATH} (see README, step 'Get the data')."
    )


# ------------------------------------------------------------------ CREATE
def init_db():
    """Create the results tables (the songs table is created by load_csv_to_db)."""
    with get_connection() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS model_results (
                   model TEXT, feature_set TEXT,
                   cv_accuracy REAL, test_accuracy REAL, test_macro_f1 REAL,
                   run_time TEXT DEFAULT CURRENT_TIMESTAMP)"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                   filename TEXT, true_label TEXT, predicted_label TEXT,
                   model TEXT, correct INTEGER)"""
        )


def load_csv_to_db(csv_path=None):
    """Read the CSV with pandas and WRITE it into the 'songs' table."""
    df = pd.read_csv(csv_path or resolve_csv())
    df.insert(0, "id", range(1, len(df) + 1))
    with get_connection() as con:
        df.to_sql("songs", con, if_exists="replace", index=False)
    init_db()
    print(f"Loaded {len(df)} songs into {DB_PATH.name}")
    return df


# -------------------------------------------------------------------- READ
def read_table(name):
    with get_connection() as con:
        return pd.read_sql(f"SELECT * FROM {name}", con)


def get_song(song_id):
    with get_connection() as con:
        return pd.read_sql("SELECT * FROM songs WHERE id = ?", con, params=(song_id,))


def count_by_genre():
    with get_connection() as con:
        return pd.read_sql(
            "SELECT label, COUNT(*) AS n FROM songs GROUP BY label ORDER BY n DESC", con
        )


# ------------------------------------------------------------------ UPDATE
def update_song_label(song_id, new_label):
    with get_connection() as con:
        cur = con.execute("UPDATE songs SET label = ? WHERE id = ?", (new_label, song_id))
        return cur.rowcount


# ------------------------------------------------------------------ DELETE
def delete_song(song_id):
    with get_connection() as con:
        cur = con.execute("DELETE FROM songs WHERE id = ?", (song_id,))
        return cur.rowcount


# ----------------------------------------------- results written by models
def save_model_results(results_df):
    """Replace model_results with a fresh DataFrame from train.py."""
    init_db()
    with get_connection() as con:
        con.execute("DELETE FROM model_results")
        results_df.to_sql("model_results", con, if_exists="append", index=False)


def save_predictions(pred_df):
    init_db()
    with get_connection() as con:
        con.execute("DELETE FROM predictions")
        pred_df.to_sql("predictions", con, if_exists="append", index=False)


# -------------------------------------------------------------------- DEMO
if __name__ == "__main__":
    load_csv_to_db()
    print("\n[READ] songs per genre:\n", count_by_genre())

    first_id = 1
    old = get_song(first_id)["label"].iloc[0]
    print(f"\n[READ] song {first_id} label = {old}")

    update_song_label(first_id, "TEMP_LABEL")
    print("[MODIFY] new label =", get_song(first_id)["label"].iloc[0])
    update_song_label(first_id, old)  # restore
    print("[MODIFY] restored  =", get_song(first_id)["label"].iloc[0])

    # delete demo on the last row, then restore the full table
    n_before = len(read_table("songs"))
    delete_song(n_before)
    print(f"[DELETE] rows {n_before} -> {len(read_table('songs'))}")
    load_csv_to_db()  # restore full dataset
