"""Exploratory Data Analysis. Run:  python -m src.eda
Saves every plot to reports/figures/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR
from src.data_loader import load_dataframe, feature_columns

sns.set_theme(style="whitegrid")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130)
    plt.close(fig)
    print("saved", name)


def overview(df):
    """Shape, missing values, duplicates, summary statistics."""
    cols = feature_columns(df)
    info = {
        "rows": len(df),
        "features": len(cols),
        "genres": df["label"].nunique(),
        "missing_values": int(df[cols].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(subset=cols).sum()),
    }
    return info, df[cols].describe().T


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df["label"].value_counts().index
    sns.countplot(data=df, x="label", order=order, hue="label", legend=False, ax=ax)
    ax.set_title("Class balance: songs per genre")
    ax.set_xlabel("")
    _save(fig, "01_class_balance.png")


def plot_feature_by_genre(df, feature, fname):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = df.groupby("label")[feature].median().sort_values().index
    sns.boxplot(data=df, x="label", y=feature, order=order, hue="label",
                legend=False, ax=ax)
    ax.set_title(f"{feature} by genre")
    ax.set_xlabel("")
    _save(fig, fname)


def plot_mfcc_heatmap(df):
    mfcc = [c for c in df.columns if c.startswith("mfcc") and c.endswith("_mean")]
    means = df.groupby("label")[mfcc].mean()
    z = (means - means.mean()) / means.std()      # compare genres on one scale
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(z, cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "z-score across genres"})
    ax.set_title("Average MFCC pattern per genre")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, "04_mfcc_heatmap.png")


def plot_correlation(df):
    cols = feature_columns(df)
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, xticklabels=False,
                yticklabels=False, cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Feature correlation matrix")
    _save(fig, "05_correlation_heatmap.png")


def plot_distributions(df):
    feats = ["tempo", "spectral_centroid_mean", "rms_mean", "zero_crossing_rate_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, f in zip(axes.ravel(), feats):
        sns.histplot(df[f], kde=True, ax=ax)
        ax.set_title(f)
    _save(fig, "06_distributions.png")


def run_all():
    df = load_dataframe()
    info, desc = overview(df)
    print(info)
    print(desc.head(10))
    plot_class_balance(df)
    plot_feature_by_genre(df, "tempo", "02_tempo_by_genre.png")
    plot_feature_by_genre(df, "spectral_centroid_mean", "03_spectral_centroid_by_genre.png")
    plot_mfcc_heatmap(df)
    plot_correlation(df)
    plot_distributions(df)
    desc.to_csv(FIGURES_DIR.parent / "feature_summary_stats.csv")


if __name__ == "__main__":
    run_all()
