"""Feature understanding and reduction. Run:  python -m src.feature_reduction

Four feature sets are compared with the SAME classifier (SVM) using 5-fold CV:
  1. all features
  2. correlation pruning (drop one of every pair with |r| > 0.90)
  3. PCA keeping 95% of the variance
  4. top-20 features picked by Random Forest importance
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import FIGURES_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS
from src.data_loader import get_xy


def drop_correlated(X, threshold=0.90):
    """Return (kept_columns, dropped_columns) using the absolute correlation matrix."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    kept = [c for c in X.columns if c not in dropped]
    return kept, dropped


def plot_pca_variance(X):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.argmax(cum >= 0.95) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    ax.axhline(0.95, color="red", ls="--", label="95% variance")
    ax.axvline(n95, color="green", ls="--", label=f"{n95} components")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA: how many dimensions do we really need?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_pca_variance.png", dpi=130)
    plt.close(fig)
    return n95


def plot_pca_2d(X, y):
    Xs = StandardScaler().fit_transform(X)
    pts = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=y.values, s=25, ax=ax)
    ax.set_title("Songs projected on the first 2 principal components")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_pca_2d.png", dpi=130)
    plt.close(fig)


def plot_importance(X, y, top=20):
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot.barh(ax=ax)
    ax.set_title(f"Top {top} features (Random Forest importance)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_feature_importance.png", dpi=130)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def compare_feature_sets(X, y, kept_cols, n_pca):
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    svm = lambda: SVC(kernel="rbf", C=10, gamma="scale")
    setups = {
        "All features": (X, [("scale", StandardScaler()), ("clf", svm())]),
        "Correlation pruning (|r|>0.90)": (
            X[kept_cols], [("scale", StandardScaler()), ("clf", svm())]),
        f"PCA ({n_pca} comps, 95% var)": (
            X, [("scale", StandardScaler()), ("pca", PCA(n_components=0.95)), ("clf", svm())]),
        "Top-20 by RF importance": (
            X, [("scale", StandardScaler()),
                ("select", SelectFromModel(
                    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
                    max_features=20, threshold=-np.inf)),
                ("clf", svm())]),
    }
    rows = []
    for name, (data, steps) in setups.items():
        scores = cross_val_score(Pipeline(steps), data, y, cv=cv, scoring="accuracy")
        n_feat = {"All features": X.shape[1], "Top-20 by RF importance": 20}.get(
            name, len(kept_cols) if name.startswith("Corr") else n_pca)
        rows.append({"feature_set": name, "n_features": n_feat,
                     "cv_accuracy_mean": round(float(scores.mean()), 4),
                     "cv_accuracy_std": round(float(scores.std()), 4)})
        print(rows[-1])
    return pd.DataFrame(rows)


def run_all():
    X, y, cols = get_xy()
    kept, dropped = drop_correlated(X, 0.90)
    print(f"Correlation pruning: kept {len(kept)}, dropped {len(dropped)}")
    pd.Series(dropped, name="dropped_features").to_csv(
        REPORTS_DIR / "dropped_correlated_features.csv", index=False)

    n95 = plot_pca_variance(X)
    plot_pca_2d(X, y)
    imp = plot_importance(X, y)
    imp.to_csv(REPORTS_DIR / "feature_importance.csv", header=["importance"])

    result = compare_feature_sets(X, y, kept, n95)
    result.to_csv(REPORTS_DIR / "feature_reduction_results.csv", index=False)
    print(result)


if __name__ == "__main__":
    run_all()
