"""Simple classification helpers for maneuvers demo."""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score

# optional backends
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras import models, layers, optimizers
    HAS_TF = True
except Exception:  # pragma: no cover - optional
    tf = None
    models = None
    layers = None
    optimizers = None
    HAS_TF = False


def segment_aggregated_features(features_df: pd.DataFrame, segment: Tuple[int, int]) -> np.ndarray:
    s, e = segment
    seg = features_df.iloc[s:e]
    # richer aggregations
    mean_accel = seg["accel_mag"].mean()
    std_accel = seg["accel_mag"].std()
    max_accel = seg["accel_mag"].max()
    min_accel = seg["accel_mag"].min()
    mean_gyro = seg["gyro_mag"].mean()
    energy = (seg["accel_mag"] ** 2).sum()
    length = e - s
    return np.array([mean_accel, std_accel, max_accel, min_accel, mean_gyro, energy, length], dtype=float)


def build_training_data_from_sequence(seq, features_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build X,y from a synthetic sequence using ground-truth segments.

    Each ground-truth segment becomes one training example (aggregated features).
    We also add a few negative/background examples sampled from non-maneuver intervals labeled 'none'.
    """
    X = []
    y = []

    # positive examples
    for s, e, label in seq.segments:
        feat = segment_aggregated_features(features_df, (s, e))
        X.append(feat)
        y.append(label)

    N = len(features_df)
    # negative examples: sample random windows that do not overlap GT
    rng = np.random.default_rng(0)
    attempts = 0
    while len([yy for yy in y if yy == 'none']) < max(3, len(seq.segments)) and attempts < 100:
        attempts += 1
        w = int(rng.integers(10, max(20, N // 10)))
        start = int(rng.integers(0, max(1, N - w)))
        end = start + w
        # skip if overlaps GT
        overlap = False
        for s, e, _ in seq.segments:
            if not (end <= s or start >= e):
                overlap = True
                break
        if overlap:
            continue
        feat = segment_aggregated_features(features_df, (start, end))
        X.append(feat)
        y.append("none")

    return np.vstack(X), y


def build_pipeline(model_type: str = "rf", **kwargs) -> Pipeline:
    """Return a standard scaler + classifier pipeline.

    Supported model_type values:
      - 'rf' : RandomForestClassifier (default)
      - 'logistic' : LogisticRegression
      - 'mlp' : sklearn MLPClassifier (lightweight neural net)
      - 'xgb' : XGBoost XGBClassifier (optional, requires `xgboost` package)
      - 'cnn' : simple 1D-CNN using TensorFlow (optional, requires `tensorflow`)

    The 'cnn' pipeline expects input arrays shaped for time-series (n_samples, seq_len, n_channels)
    and is provided as an optional experiment. If the required backend isn't installed
    an informative error is raised.
    """
    model_type = model_type.lower()
    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=0, **kwargs)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if model_type == "logistic":
        clf = LogisticRegression(max_iter=1000)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if model_type == "xgb":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install with `pip install xgboost` to use model_type='xgb'.")
        clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=0, **kwargs)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if model_type == "cnn":
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed. Install `tensorflow` to use model_type='cnn'.")
        # note: we wrap a Keras model via a sklearn-compatible class defined below
        return Pipeline([("scaler", StandardScaler()), ("clf", _KerasDenseClassifier(**kwargs))])

    raise ValueError(f"unknown model_type: {model_type}")


# lightweight Keras wrapper for dense models (works with vector inputs)
class _KerasDenseClassifier:
    def __init__(self, input_dim: int | None = None, n_classes: int | None = None, epochs: int = 20, batch_size: int = 16):
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build(self, input_dim: int, n_classes: int):
        m = models.Sequential()
        m.add(layers.Input(shape=(input_dim,)))
        m.add(layers.Dense(64, activation="relu"))
        m.add(layers.Dense(32, activation="relu"))
        m.add(layers.Dense(n_classes, activation="softmax"))
        m.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("_KerasDenseClassifier expects 2D input (n_samples, n_features)")
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.n_classes is None:
            self.n_classes = int(np.unique(y).shape[0])
        self.model = self._build(self.input_dim, self.n_classes)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X):
        return self.model.predict(np.asarray(X))

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


def train_classifier(X: np.ndarray, y: List[str], model_type: str = "rf", cv: int = 5, **kwargs) -> Dict:
    """Train a classifier and return a dict with model, encoder, and CV scores.

    This function is careful about very small sample regimes. If there are fewer
    than 2 classes or one of the classes has only a single sample, we skip
    cross-validation and train on the full set to avoid StratifiedKFold errors.
    """
    import warnings

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # special handling for keras wrapper: it expects dense vectors
    pipe = build_pipeline(model_type, **kwargs)

    unique, counts = np.unique(y_enc, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    cv_scores = {}

    # Need at least two classes and each class must have >= 2 samples for StratifiedKFold
    if len(unique) < 2 or min_count < 2:
        warnings.warn(
            "Not enough class variety or too few samples per class for cross-validation; skipping CV."
        )
    else:
        n_splits = min(cv, int(min_count))
        n_splits = max(2, n_splits)
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            cv_scores = cross_validate(
                pipe,
                X,
                y_enc,
                cv=skf,
                scoring={"f1": make_scorer(f1_score, average="macro")},
                return_train_score=False,
                error_score="raise",
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Cross-validation failed: {exc}; training without CV")
            cv_scores = {}

    pipe.fit(X, y_enc)
    return {"model": pipe, "encoder": le, "cv_scores": cv_scores}


def train_with_grid_search(X: np.ndarray, y: List[str], model_type: str = "rf", param_grid: dict | list | None = None, cv: int = 3, **kwargs) -> Dict:
    """Train with GridSearchCV and return the best model and results.

    If grid search fails (e.g., tiny dataset or backend error) we fall back to `train_classifier`.
    """
    import warnings

    if param_grid is None:
        # simple default grid for common types
        if model_type == "rf":
            param_grid = {"clf__n_estimators": [50, 100]}
        elif model_type == "xgb":
            param_grid = {"clf__n_estimators": [50, 100], "clf__max_depth": [3, 6]}
        else:
            param_grid = {}

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pipe = build_pipeline(model_type, **kwargs)

    unique, counts = np.unique(y_enc, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0

    # If dataset too small to stratify, skip grid search
    if len(unique) < 2 or min_count < 2:
        warnings.warn("Skipping grid search due to too few classes or tiny class sizes; training without grid search")
        return train_classifier(X, y, model_type=model_type, cv=cv, **kwargs)

    try:
        skf = StratifiedKFold(n_splits=max(2, min(cv, int(min_count))), shuffle=True, random_state=0)
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=skf, scoring=make_scorer(f1_score, average="macro"))
        gs.fit(X, y_enc)
        return {"model": gs.best_estimator_, "best_params": gs.best_params_, "cv_results": gs.cv_results_}
    except Exception as exc:
        warnings.warn(f"GridSearchCV failed: {exc}; falling back to plain training")
        return train_classifier(X, y, model_type=model_type, cv=cv, **kwargs)


def save_model(obj: Dict, path: str) -> None:
    joblib.dump(obj, path)


def load_model(path: str) -> Dict:
    return joblib.load(path)


def predict_segment_labels(model_obj: Dict, features_df: pd.DataFrame, segments: List[Tuple[int, int]]) -> List[str]:
    model = model_obj["model"]
    le: LabelEncoder = model_obj["encoder"]
    X = [segment_aggregated_features(features_df, seg) for seg in segments]
    proba = model.predict_proba(np.vstack(X))
    idx = proba.argmax(axis=1)
    return list(le.inverse_transform(idx))
