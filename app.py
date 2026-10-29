"""Streamlit demo.  Run:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from src.config import BEST_MODEL_PATH, REPORTS_DIR, FIGURES_DIR
from src.data_loader import load_dataframe
from src.recommender import SongRecommender

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="wide")
st.title("🎵 Music Genre Classification")
st.caption("Predict a song's genre from audio features, and find similar songs.")


@st.cache_data
def get_data():
    return load_dataframe()


@st.cache_resource
def get_model():
    return joblib.load(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else None


df = get_data()
bundle = get_model()
tab1, tab2, tab3 = st.tabs(["🎧 Predict & recommend", "📊 Model results", "🖼️ EDA gallery"])

with tab1:
    if bundle is None:
        st.error("No trained model found. Run `python -m src.train` first.")
    else:
        genre = st.selectbox("Filter songs by genre", ["all"] + sorted(df["label"].unique()))
        pool = df if genre == "all" else df[df["label"] == genre]
        song = st.selectbox("Choose a song", pool["filename"].tolist())
        row = df[df["filename"] == song]

        proba = bundle["pipeline"].predict_proba(row[bundle["features"]])[0]
        probs = pd.Series(proba, index=bundle["classes"]).sort_values(ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted genre", probs.index[0], f"{probs.iloc[0]:.0%} confidence")
            st.write(f"Actual genre: **{row['label'].iloc[0]}**  |  Model: {bundle['name']}")
            st.bar_chart(probs)
        with c2:
            st.subheader("Similar songs (cosine similarity)")
            st.dataframe(SongRecommender(df).recommend(song, 5), use_container_width=True)

with tab2:
    res_path = REPORTS_DIR / "model_results.csv"
    if res_path.exists():
        st.dataframe(pd.read_csv(res_path), use_container_width=True)
    for name in ("10_model_comparison.png", "11_confusion_matrix.png"):
        if (FIGURES_DIR / name).exists():
            st.image(str(FIGURES_DIR / name))

with tab3:
    for img in sorted(FIGURES_DIR.glob("0*.png")):
        st.image(str(img), caption=img.stem)
