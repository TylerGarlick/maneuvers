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
from maneuvers.models.cnn import _CNN1DClassifier, _CNN1DMultiBranchClassifier

import numpy as np
from collections import Counter
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
      - 'cnn_multi' : multi-branch 1D-CNN processing accel and gyro channels separately (optional)

    The 'cnn'/'cnn_multi' pipelines produce Keras-backed wrappers that expect
    3D time-series inputs and therefore don't play with sklearn cross-validation
    in a standard way. Use the specialized training helpers in this module to
    train and evaluate them (for example `train_cnn_from_sequence`).
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
        return _CNN1DClassifier
    if model_type == "cnn_multi":
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed. Install `tensorflow` to use model_type='cnn_multi'.")
        return _CNN1DMultiBranchClassifier

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
    Note: Grid search is not supported for 'cnn'/'cnn_multi' which are trained via separate APIs.
    """
    if model_type in ("cnn", "cnn_multi"):
        raise ValueError("Grid search is not supported for CNN models; train them using train_cnn_from_sequence or train_cnn")
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


# --- CNN training helpers -----------------------------------------------------

def balance_classes(X: np.ndarray, y: np.ndarray, strategy: str = "oversample") -> Tuple[np.ndarray, np.ndarray]:
    """Balance classes by oversampling or undersampling.

    - 'oversample': duplicate samples from minority classes
    - 'undersample': subsample majority classes
    """
    unique, counts = np.unique(y, return_counts=True)
    if strategy == "oversample":
        target = counts.max()
        X_new, y_new = [], []
        for cls in unique:
            idx = np.where(y == cls)[0]
            n_samples = len(idx)
            # oversample to target
            sample_idx = np.random.choice(idx, size=target, replace=True)
            X_new.append(X[sample_idx])
            y_new.append(y[sample_idx])
        return np.vstack(X_new), np.concatenate(y_new)
    elif strategy == "undersample":
        target = counts.min()
        X_new, y_new = [], []
        for cls in unique:
            idx = np.where(y == cls)[0]
            sample_idx = np.random.choice(idx, size=target, replace=False)
            X_new.append(X[sample_idx])
            y_new.append(y[sample_idx])
        return np.vstack(X_new), np.concatenate(y_new)
    return X, y


def augment_windows(X: np.ndarray, y: np.ndarray, n_aug: int = 1, noise_std: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Simple data augmentation: add Gaussian noise to create additional samples."""
    X_aug, y_aug = [], []
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        for _ in range(n_aug):
            noise = np.random.normal(0, noise_std, X[i].shape)
            X_aug.append(X[i] + noise)
            y_aug.append(y[i])
    return np.asarray(X_aug), np.asarray(y_aug)


def train_cnn_from_sequence(seq, model_type: str = "cnn_multi", window_s: float = 1.0, hop_s: float = 0.5, fs: int = 100, balance: str | None = "oversample", augment: int = 0, epochs: int = 20, batch_size: int = 16) -> Dict:
    """Train a CNN model from a Sequence using windowing and optional balancing/augmentation.

    Returns a dict with 'model' (trained CNN), 'encoder' (LabelEncoder), and 'cv_scores' (empty).
    """
    from maneuvers.preprocessing import windowed_examples_from_sequence

    X, y, windows = windowed_examples_from_sequence(seq, window_s=window_s, hop_s=hop_s, fs=fs)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # balance
    if balance:
        X, y_enc = balance_classes(X, y_enc, strategy=balance)
    # augment
    if augment > 0:
        X, y_enc = augment_windows(X, y_enc, n_aug=augment)

    # build model
    seq_len = X.shape[1]
    n_channels = X.shape[2]
    n_classes = len(np.unique(y_enc))

    if model_type == "cnn":
        clf = _CNN1DClassifier(seq_len=seq_len, n_channels=n_channels, n_classes=n_classes, epochs=epochs, batch_size=batch_size)
    elif model_type == "cnn_multi":
        n_accel = 3
        n_gyro = 3
        clf = _CNN1DMultiBranchClassifier(seq_len=seq_len, n_accel=n_accel, n_gyro=n_gyro, n_classes=n_classes, epochs=epochs, batch_size=batch_size)
    else:
        raise ValueError(f"model_type {model_type} not supported for CNN training; use 'cnn' or 'cnn_multi'")

    clf.fit(X, y_enc)
    return {"model": clf, "encoder": le, "cv_scores": {}}


def save_model(obj: Dict, path: str) -> None:
    """Save a model object. If model contains a Keras model, save it separately.

    Behavior:
    - For Keras-based classifiers (CNN), save the Keras model to path + '.keras',
      and joblib-dump the rest of the object (with a placeholder indicating the keras path).
    - For sklearn objects, joblib-dump the whole dict.
    """
    # detect keras-backed wrappers
    model_obj = obj.get("model") if isinstance(obj, dict) else None
    if model_obj is not None and hasattr(model_obj, "model") and getattr(model_obj, "model") is not None:
        keras_path = path + ".keras"
        try:
            # save keras model
            model_obj.model.save(keras_path, overwrite=True, include_optimizer=True)
            # remove large keras object from dict and replace with a reference
            stub = obj.copy()
            stub = dict(stub)
            stub["_keras_path"] = keras_path
            # don't include the model object itself (not picklable safely)
            if "model" in stub:
                del stub["model"]
            joblib.dump(stub, path)
            return
        except Exception:
            # fallback to saving whole object (might fail)
            pass
    joblib.dump(obj, path)


def load_model(path: str) -> Dict:
    """Load a model saved by `save_model`. Restores keras models when present."""
    obj = joblib.load(path)
    if isinstance(obj, dict) and "_keras_path" in obj:
        keras_path = obj["_keras_path"]
        # load keras model
        try:
            from tensorflow.keras.models import load_model as _tf_load

            keras_m = _tf_load(keras_path)
            # set as 'model' on a small wrapper
            class _Wrapper:
                def __init__(self, m):
                    self.model = m

                def predict(self, X):
                    return self.model.predict(X)

                def predict_proba(self, X):
                    return self.model.predict(X)

            w = _Wrapper(keras_m)
            obj["model"] = w
        except Exception:
            pass
    return obj

def predict_segment_labels(model_obj: Dict, features_df: pd.DataFrame, segments: List[Tuple[int, int]], seq=None) -> List[str]:
    """Predict labels for detected segments.

    For CNN-based models, `seq` (Sequence) should be provided to extract the raw
    accel/gyro windows; for feature-based models the existing aggregated-feature
    logic is used.
    """
    model = model_obj["model"]
    le: LabelEncoder = model_obj["encoder"]

    # detect whether model is keras-backed (has predict_proba but expects 3D input)
    is_keras = hasattr(model, "predict_proba") and not hasattr(model, "__class__") or getattr(model, "model", None) is not None

    if is_keras:
        if seq is None:
            raise ValueError("For CNN models, pass the original Sequence `seq` to extract windows for each segment")
        # for each segment, crop/pad to model input size
        preds = []
        for s, e in segments:
            seg_len = e - s
            # determine seq_len from model input
            try:
                seq_len = int(model.model.input_shape[1])
            except Exception:
                # fallback to 100 samples
                seq_len = 100
            start = max(0, s)
            end = min(seq.accel.shape[0], e)
            win_acc = seq.accel[start:end]
            win_gyro = seq.gyro[start:end]
            # pad or crop to seq_len
            if win_acc.shape[0] < seq_len:
                pad_n = seq_len - win_acc.shape[0]
                win_acc = np.vstack([win_acc, np.zeros((pad_n, win_acc.shape[1]))])
                win_gyro = np.vstack([win_gyro, np.zeros((pad_n, win_gyro.shape[1]))])
            else:
                win_acc = win_acc[:seq_len]
                win_gyro = win_gyro[:seq_len]
            Xwin = np.concatenate([win_acc, win_gyro], axis=1)[None, ...]
            proba = model.predict_proba(Xwin)
            idx = int(proba.argmax(axis=1)[0])
            preds.append(le.inverse_transform([idx])[0])
        return preds

    # fallback to aggregated features
    X = [segment_aggregated_features(features_df, seg) for seg in segments]
    proba = model.predict_proba(np.vstack(X))
    idx = proba.argmax(axis=1)
    return list(le.inverse_transform(idx))
