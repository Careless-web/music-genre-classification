# 🎵 Music Genre Classification

Classify songs into 10 genres from audio features (MFCC, chroma, spectral centroid, tempo, energy) using classic machine-learning models, plus a cosine-similarity song recommender and a Streamlit demo.

**Live demo:** _paste your Streamlit link here_

## Team

| Member | Responsibility |
|---|---|
| Person A (name) | Data loading, SQLite database, repo setup, README |
| Person B (name) | EDA, feature understanding and reduction |
| Person C (name) | Models, evaluation, recommender, Streamlit app |

## Dataset

[GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification): 1000 clips of 30 seconds, 10 genres (100 each), pre-extracted into 57 numeric audio features (`features_30_sec.csv`).

## Project structure

```
├── app.py                      Streamlit demo
├── data/                       features_30_sec.csv
├── database/                   SQLite file created at run time
├── docs/                       presentation (.pptx)
├── models/                     saved best model
├── notebooks/                  00 database, 01 EDA, 02 features, 03 models
├── reports/                    result tables and figures/
├── scripts/make_synthetic_data.py   fake data for testing only
├── scripts/fill_deck.py             fills the presentation with real results
└── src/
    ├── config.py               paths and constants
    ├── database.py             SQLite create / read / update / delete
    ├── data_loader.py          DB -> pandas / NumPy
    ├── eda.py                  exploratory analysis
    ├── feature_reduction.py    correlation pruning, PCA, importance
    ├── models.py               model definitions
    ├── train.py                training, CV, metrics, confusion matrix
    └── recommender.py          cosine-similarity recommender
```

## Setup

```bash
git clone <repo-url>
cd music-genre-classification
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
pip install notebook           # only needed to open the notebooks
```

Download `features_30_sec.csv` from the Kaggle link above and place it in `data/`.

## Run order

```bash
python -m src.database            # 1. build DB, demo read/write/modify
python -m src.eda                 # 2. EDA figures
python -m src.feature_reduction   # 3. feature reduction comparison
python -m src.train               # 4. train + evaluate models
python -m src.recommender         # 5. recommender demo
streamlit run app.py              # 6. web demo
```

## Methods

- **Storage:** pandas and NumPy for processing, SQLite for persistence (songs, model_results, predictions tables).
- **EDA:** class balance, tempo and spectral centroid by genre, MFCC heatmap, correlation matrix.
- **Feature reduction:** correlation pruning, PCA (95% variance), Random Forest importance; each compared with the same SVM under 5-fold CV.
- **Models:** Logistic Regression, KNN, SVM (RBF), SVM + PCA, Random Forest, MLP neural network. Scaling is done inside a Pipeline so it is fitted on training data only.
- **Metrics:** accuracy, macro F1, 5-fold cross-validation, confusion matrix, per-genre report. The best model is chosen on cross-validation, not on the test set.

## Results

_Fill in after running `python -m src.train` on the real data (copy from `reports/model_results.csv`)._

| Model | CV accuracy | Test accuracy | Test macro F1 |
|---|---|---|---|
| | | | |

Most confused genres: _see `reports/top_confusions.csv`_

## Limitations

- GTZAN has only 1000 songs, and its labels and recordings have known flaws (duplicates, mislabelled clips).
- Features are summary statistics of 30 seconds, so temporal structure is lost.
- Genres overlap by nature (rock/metal, pop/disco), so some confusion is expected.
- The demo predicts for songs already in the dataset; uploading a new audio file would need librosa feature extraction.
- Results come from one train/test split plus cross-validation, so small differences between models are not statistically meaningful.

## Future work

CNN on mel-spectrograms, librosa feature extraction for uploaded songs, larger datasets such as FMA.
