"""Train, compare and evaluate all models.  Run:  python -m src.train

Outputs
  reports/model_results.csv            comparison table
  reports/classification_report.txt    per-genre precision / recall / F1
  reports/top_confusions.csv           most confused genre pairs
  reports/figures/10_model_comparison.png, 11_confusion_matrix.png
  models/best_model.joblib             best model (chosen by CV macro F1)
  database tables: model_results, predictions
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from src import database
from src.config import (BEST_MODEL_PATH, CV_FOLDS, FIGURES_DIR, RANDOM_STATE,
                        REPORTS_DIR, TEST_SIZE)
from src.data_loader import load_dataframe, feature_columns
from src.models import get_models


def top_confusions(cm, labels, n=8):
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true_genre": labels[i], "predicted_as": labels[j],
                              "count": int(cm[i, j])})
    out = pd.DataFrame(pairs, columns=["true_genre", "predicted_as", "count"])
    return out.sort_values("count", ascending=False).head(n)


def main():
    df = load_dataframe()
    cols = feature_columns(df)
    X, y = df[cols], df["label"]
    le = LabelEncoder()                      # genres -> 0..9 (required by the MLP)
    y_num = le.fit_transform(y)
    X_tr, X_te, ytr_num, yte_num, df_tr, df_te = train_test_split(
        X, y_num, df, test_size=TEST_SIZE, stratify=y_num, random_state=RANDOM_STATE)
    y_tr, y_te = le.inverse_transform(ytr_num), le.inverse_transform(yte_num)
    print(f"train={len(X_tr)}  test={len(X_te)}  features={len(cols)}")

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, (fset, pipe) in get_models().items():
        cvres = cross_validate(pipe, X_tr, ytr_num, cv=cv,
                               scoring=["accuracy", "f1_macro"], n_jobs=1)
        pipe.fit(X_tr, ytr_num)
        pred = le.inverse_transform(pipe.predict(X_te))
        rows.append({
            "model": name, "feature_set": fset,
            "cv_accuracy": round(float(cvres["test_accuracy"].mean()), 4),
            "cv_macro_f1": round(float(cvres["test_f1_macro"].mean()), 4),
            "test_accuracy": round(accuracy_score(y_te, pred), 4),
            "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        })
        fitted[name] = (pipe, pred)
        print(rows[-1])

    results = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    results.to_csv(REPORTS_DIR / "model_results.csv", index=False)
    database.save_model_results(
        results[["model", "feature_set", "cv_accuracy", "test_accuracy", "test_macro_f1"]])

    # ---- best model (selected on CV, NOT on the test set)
    best_name = results.iloc[0]["model"]
    best_pipe, best_pred = fitted[best_name]
    print(f"\nBest model by CV macro F1: {best_name}")

    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, best_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix: {best_name} (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_confusion_matrix.png", dpi=130)
    plt.close(fig)

    report = classification_report(y_te, best_pred, digits=3)
    (REPORTS_DIR / "classification_report.txt").write_text(report)
    print(report)
    conf = top_confusions(cm, labels)
    conf.to_csv(REPORTS_DIR / "top_confusions.csv", index=False)
    print("Most confused genre pairs:\n", conf)

    # ---- model comparison chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    r = results.set_index("model")[["cv_accuracy", "test_accuracy", "test_macro_f1"]]
    r.plot.bar(ax=ax, rot=20)
    ax.set_ylim(max(0, r.values.min() - 0.1), 1.0)
    ax.set_title("Model comparison")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_model_comparison.png", dpi=130)
    plt.close(fig)

    # ---- save model + predictions to database
    joblib.dump({"pipeline": best_pipe, "name": best_name, "features": cols,
                 "classes": list(le.classes_)}, BEST_MODEL_PATH)
    pred_df = pd.DataFrame({
        "filename": df_te["filename"].values, "true_label": y_te,
        "predicted_label": best_pred, "model": best_name,
        "correct": (y_te == best_pred).astype(int)})
    database.save_predictions(pred_df)
    print("Saved model and predictions.")


if __name__ == "__main__":
    main()
