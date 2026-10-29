"""Cosine-similarity song recommender (project extension).

Run:  python -m src.recommender
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataframe, feature_columns


class SongRecommender:
    def __init__(self, df=None):
        self.df = (load_dataframe() if df is None else df).reset_index(drop=True)
        self.cols = feature_columns(self.df)
        # scale first, otherwise features with large numbers dominate the similarity
        self.matrix = StandardScaler().fit_transform(self.df[self.cols])

    def recommend(self, filename, k=5):
        idx = self.df.index[self.df["filename"] == filename]
        if len(idx) == 0:
            raise ValueError(f"{filename} not found")
        i = idx[0]
        sims = cosine_similarity(self.matrix[i:i + 1], self.matrix).ravel()
        sims[i] = -np.inf                       # exclude the song itself
        top = np.argsort(sims)[::-1][:k]
        out = self.df.loc[top, ["filename", "label"]].copy()
        out["similarity"] = sims[top].round(3)
        return out.reset_index(drop=True)

    def genre_hit_rate(self, k=5):
        """Share of recommended songs that have the same genre as the query song."""
        sims = cosine_similarity(self.matrix)
        np.fill_diagonal(sims, -np.inf)
        top = np.argsort(sims, axis=1)[:, ::-1][:, :k]
        labels = self.df["label"].to_numpy()
        return float((labels[top] == labels[:, None]).mean())


if __name__ == "__main__":
    rec = SongRecommender()
    song = rec.df["filename"].iloc[0]
    print(f"Songs similar to {song} ({rec.df['label'].iloc[0]}):")
    print(rec.recommend(song))
    print(f"\nGenre hit-rate@5: {rec.genre_hit_rate(5):.3f}")
