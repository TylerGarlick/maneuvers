"""Simple classification helpers for maneuvers demo."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


import numpy as np
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score


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
    """Return a standard scaler + classifier pipeline."""
    clf = RandomForestClassifier(n_estimators=100, random_state=0, **kwargs) if model_type == "rf" else LogisticRegression(max_iter=1000)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_classifier(X: np.ndarray, y: List[str], model_type: str = "rf", cv: int = 5, **kwargs) -> Dict:
    """Train a classifier and return a dict with model, encoder, and CV scores."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pipe = build_pipeline(model_type, **kwargs)
    scores = cross_validate(pipe, X, y_enc, cv=max(2, cv), scoring={"f1": make_scorer(f1_score, average="macro")}, return_train_score=False)
    pipe.fit(X, y_enc)
    return {"model": pipe, "encoder": le, "cv_scores": scores}


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
