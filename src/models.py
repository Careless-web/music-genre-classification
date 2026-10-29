"""Model definitions. Every model is a scikit-learn Pipeline so that scaling is
fitted ONLY on training data (no data leakage)."""
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RANDOM_STATE


def get_models():
    """Return {name: (feature_set_label, pipeline)}."""
    return {
        "Logistic Regression": ("all features", Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])),
        "KNN": ("all features", Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))])),
        "SVM (RBF)": ("all features", Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                        random_state=RANDOM_STATE))])),
        "SVM + PCA(95%)": ("PCA 95% variance", Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                        random_state=RANDOM_STATE))])),
        "Random Forest": ("all features", Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                           n_jobs=-1))])),
        "Neural Network (MLP)": ("all features", Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                                  early_stopping=True, random_state=RANDOM_STATE))])),
    }
