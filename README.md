# 🎵 Music Genre Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://music-genre-classification-t2jd5prkssrzckiscqbav7.streamlit.app/)

Classify songs into 10 genres from audio features (MFCC, chroma, spectral centroid, tempo, energy) using classic machine-learning models. The project also includes a cosine-similarity song recommender and an interactive Streamlit demo.

**Live demo:** [Open Music Genre Classification App](https://music-genre-classification-t2jd5prkssrzckiscqbav7.streamlit.app/)

## Team

| Member | GitHub | Responsibility |
|---|---|---|
| Saurabh Kumar | [Careless-web](https://github.com/Careless-web) | Repo setup, SQLite database, data loader, README, deployment |
| Ayush Raj | [ms23255-ops](https://github.com/ms23255-ops) | Exploratory data analysis, feature understanding and reduction |
| Dhanraj Kumar | [Dhanraj-76](https://github.com/Dhanraj-76) | Models, evaluation, recommender, Streamlit app |

## Dataset

[GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification): 1000 audio clips of 30 seconds, 10 genres (100 clips each), already summarised into 57 numeric audio features in `features_30_sec.csv`. The data is loaded with pandas and NumPy and stored in a SQLite database.

Genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.

## Project structure

```
├── app.py                          Streamlit demo
├── data/                           features_30_sec.csv
├── database/                       SQLite file (created at run time)
├── docs/                           presentation
├── models/                         saved best model
├── notebooks/                      00 database, 01 EDA, 02 features, 03 models
├── reports/                        result tables and figures/
├── scripts/                        helper scripts
└── src/
    ├── config.py                   paths and constants
    ├── database.py                 SQLite create / read / update / delete
    ├── data_loader.py              database -> pandas / NumPy
    ├── eda.py                      exploratory analysis
    ├── feature_reduction.py        correlation pruning, PCA, importance
    ├── models.py                   model definitions
    ├── train.py                    training, cross-validation, metrics
    └── recommender.py              cosine-similarity recommender
```

## Setup

```bash
git clone https://github.com/Careless-web/music-genre-classification.git
cd music-genre-classification
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install ipykernel           # only needed to run the notebooks in VS Code
```

The dataset is already included in `data/`. If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

## How to run

```bash
python -m src.database            # build the database, demo read / write / modify / delete
python -m src.eda                 # EDA figures
python -m src.feature_reduction   # feature reduction comparison
python -m src.train               # train and evaluate all models
python -m src.recommender         # recommender demo and hit-rate
streamlit run app.py              # interactive demo
```

## Methods

**Storage.** pandas and NumPy for processing; SQLite for persistence with three tables: `songs` (the data), `model_results` and `predictions`. `src/database.py` supports create, read, update and delete.

**EDA.** Class balance, tempo and spectral centroid by genre, average MFCC pattern per genre, correlation matrix and feature distributions (see `notebooks/01_eda.ipynb` and `reports/figures/`).

**Feature reduction.** Four feature sets are compared with the same SVM under 5-fold cross-validation: all features, correlation pruning (|r| > 0.90), PCA (95% of the variance) and the top 20 features by Random Forest importance.

| Feature set | Features | CV accuracy (mean ± std) |
|---|---|---|
| All features | 57 | 0.730 ± 0.029 |
| Correlation pruning (\|r\|>0.90) | 54 | 0.737 ± 0.026 |
| PCA (95% variance) | 33 | 0.731 ± 0.031 |
| Top-20 by RF importance | 20 | 0.702 ± 0.026 |

_Source: `reports/feature_reduction_results.csv`. These numbers use all 1000 songs with 5-fold cross-validation, so they differ slightly from the model comparison below, which cross-validates on the 800 training songs only._

**Findings.** Removing the 3 highly correlated features did not hurt accuracy (0.737 vs 0.730, well within one standard deviation). PCA cut the feature count from 57 to 33 (42% fewer) with the same accuracy (0.731), so the data contains a lot of redundant information. Keeping only the top 20 features by importance lowered accuracy by about 3 points (0.702), so the remaining features still carry useful signal. All differences are of the same size as the fold-to-fold variation (about 0.03), so they should be read as trends, not proof.

**Models.** Logistic Regression, KNN, SVM (RBF), SVM + PCA, Random Forest and a neural network (MLP). Every model is a scikit-learn Pipeline, so the scaler is fitted on training data only (no data leakage). The data is split 80/20 (stratified), models are compared with 5-fold cross-validation on the training set, and the best model is chosen on cross-validation, not on the test set.

## Results

Test set: 200 songs (20 per genre), never used for training or model selection.

| Model | CV accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|
| **SVM (RBF)** | **0.749** | **0.765** | **0.764** |
| SVM + PCA (95%) | 0.733 | 0.740 | 0.740 |
| Logistic Regression | 0.716 | 0.710 | 0.711 |
| Neural Network (MLP) | 0.678 | 0.710 | 0.703 |
| KNN | 0.688 | 0.705 | 0.704 |
| Random Forest | 0.699 | 0.705 | 0.698 |

The best model is the **SVM with an RBF kernel (76.5% test accuracy)**. The gaps between the top models are small, and with only 200 test songs each song is worth 0.5%, so differences of a few points should not be over-interpreted. PCA to 95% variance cost about 1.5 points of accuracy.

**Per-genre results (SVM).** Metal (F1 0.865), country (0.850) and hip-hop (0.821) are the easiest genres. Disco (0.611), reggae (0.682) and rock (0.684) are the hardest.

**Most confused genre pairs** (true genre → predicted as, number of test songs): classical → jazz (3), disco → pop (3), metal → blues (3), pop → reggae (3), hip-hop → disco (2), disco → rock (2), jazz → classical (2). Classical/jazz and disco/pop are explained by shared instruments, harmony and rhythm. With only 20 test songs per genre these counts are small, so a different split could change them. The confusion matrix is in `reports/figures/11_confusion_matrix.png`.

## Song recommender (extension)

For a chosen song, features are standardised and the 5 most similar songs are returned by cosine similarity. Averaged over all 1000 songs, **61.1% of the top-5 recommendations share the query song's genre** (random chance would be about 10%). The recommender finds songs that sound alike overall, which is not always the same as the same genre, so it is weaker than the classifier.

## Demo

The Streamlit app (`app.py`) lets you pick any song from the dataset and shows the predicted genre with class probabilities, the actual genre, and the 5 most similar songs. It also shows the model comparison, the confusion matrix and the EDA plots. Note that songs from the training split look better than unseen data would; use the test-set results above for performance.

## Limitations

- GTZAN has only 1000 songs, and the dataset has known flaws (some mislabelled and duplicated clips), so results cannot be generalised to all music.
- Features are summaries of 30 seconds, so changes over time within a song are lost.
- Genres overlap by nature, so some confusion is expected.
- The test set is small (200 songs): small differences between models are not statistically meaningful.
- The demo predicts for songs already in the dataset. A new audio file would need feature extraction with a library such as librosa, and its features would differ slightly from GTZAN's.
- The recommender uses all 57 features equally, so it matches genre only 61% of the time.

## Future work

CNN on mel-spectrograms, feature extraction from uploaded audio files with librosa, a recommender based on a smaller feature set (e.g. MFCCs only), and larger datasets such as FMA.

## Reproducibility

The random seed is fixed (`RANDOM_STATE = 42` in `src/config.py`), so running `python -m src.train` reproduces the results above. Library versions are pinned in `requirements.txt`.
