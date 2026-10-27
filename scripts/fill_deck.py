"""Fill docs/Music_Genre_Classification.pptx with YOUR real results.

Run AFTER:  python -m src.eda, python -m src.feature_reduction, python -m src.train
Then:       python -m scripts.fill_deck
Output:     docs/Music_Genre_Classification_filled.pptx

It replaces every dashed "Picture goes here" box with the real figure and fills
the result tables and text boxes from the CSV files in reports/.
"""
import copy
import pandas as pd
from PIL import Image
from pptx import Presentation

from src.config import ROOT, FIGURES_DIR, REPORTS_DIR

SRC = ROOT / "docs" / "Music_Genre_Classification.pptx"
OUT = ROOT / "docs" / "Music_Genre_Classification_filled.pptx"


def set_text(shape, lines):
    """Replace text but keep the formatting of the first run."""
    tf = shape.text_frame
    first = tf.paragraphs[0]
    template = copy.deepcopy(first._p)
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for i, line in enumerate(lines):
        if i == 0:
            para = first
        else:
            new_p = copy.deepcopy(template)
            first._p.getparent().append(new_p)
            para = tf.paragraphs[-1]
        runs = para.runs
        runs[0].text = line
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


def set_cell(cell, text):
    para = cell.text_frame.paragraphs[0]
    para.runs[0].text = str(text)


def fill_picture(slide, shape, path):
    with Image.open(path) as im:
        iw, ih = im.size
    box_w, box_h = shape.width, shape.height
    scale = min(box_w / iw, box_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    left = shape.left + (box_w - w) // 2
    top = shape.top + (box_h - h) // 2
    shape._element.getparent().remove(shape._element)
    slide.shapes.add_picture(str(path), left, top, w, h)


def main():
    prs = Presentation(SRC)
    done, skipped = [], []
    for slide in prs.slides:
        for shape in list(slide.shapes):
            name = shape.name

            if name.startswith("FIG:"):
                f = FIGURES_DIR / name[4:]
                if f.exists():
                    fill_picture(slide, shape, f); done.append(name)
                else:
                    skipped.append(name)

            elif name == "TABLE:model_results":
                p = REPORTS_DIR / "model_results.csv"
                if not p.exists(): skipped.append(name); continue
                df = pd.read_csv(p)
                order = ["Logistic Regression", "KNN", "SVM (RBF)", "SVM + PCA(95%)",
                         "Random Forest", "Neural Network (MLP)"]
                df = df.set_index("model").reindex(order)
                tbl = shape.table
                for r, (_, row) in enumerate(df.iterrows(), start=1):
                    set_cell(tbl.cell(r, 1), f"{row['cv_accuracy']:.3f}")
                    set_cell(tbl.cell(r, 2), f"{row['test_accuracy']:.3f}")
                    set_cell(tbl.cell(r, 3), f"{row['test_macro_f1']:.3f}")
                done.append(name)

            elif name == "TABLE:feature_reduction":
                p = REPORTS_DIR / "feature_reduction_results.csv"
                if not p.exists(): skipped.append(name); continue
                df = pd.read_csv(p)
                tbl = shape.table
                for r, (_, row) in enumerate(df.iterrows(), start=1):
                    set_cell(tbl.cell(r, 1), int(row["n_features"]))
                    set_cell(tbl.cell(r, 2), f"{row['cv_accuracy_mean']:.3f}")
                done.append(name)

            elif name == "TEXT:top_confusions":
                p = REPORTS_DIR / "top_confusions.csv"
                if not p.exists(): skipped.append(name); continue
                df = pd.read_csv(p).head(4)
                lines = ([f"{r.true_genre} → {r.predicted_as}  ({r['count']}x)"
                          for _, r in df.iterrows()]
                         or ["No mistakes on this test split"])
                set_text(shape, lines); done.append(name)

            elif name == "TEXT:best_model":
                p = REPORTS_DIR / "model_results.csv"
                if not p.exists(): skipped.append(name); continue
                best = pd.read_csv(p).iloc[0]
                set_text(shape, [f"Best: {best['model']} - CV macro F1 {best['cv_macro_f1']:.2f}, "
                                 f"test accuracy {best['test_accuracy']:.2f}"])
                done.append(name)

    prs.save(OUT)
    print(f"Filled {len(done)} items -> {OUT}")
    if skipped:
        print("Not filled (file missing, run the scripts first):", ", ".join(skipped))


if __name__ == "__main__":
    main()
