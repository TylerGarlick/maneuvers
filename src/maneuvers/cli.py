"""Command-line interface for the maneuvers baseline (Typer)."""

from __future__ import annotations
import json
from typing import Optional

import typer
from .data.loader import generate_synthetic_sequence
from .preprocessing import compute_features_from_sequence
from .detection import detect_segments
from .eval import evaluate_detection

app = typer.Typer(help="maneuvers CLI - minimal baseline")


@app.command()
def detect_synthetic(
    duration: float = 10.0,
    fs: int = 100,
    threshold: float = 0.5,
    min_len: int = 5,
    out: Optional[str] = None,
    model: Optional[str] = None,
):
    """Generate a synthetic sequence and run baseline detection. Optionally label segments using a saved model."""
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)
    feats = compute_features_from_sequence(seq)
    preds = detect_segments(
        feats, method="threshold", threshold=threshold, min_len=min_len
    )

    labels = None
    if model:
        try:
            from .classify import load_model, predict_segment_labels

            mobj = load_model(model)
            labels = predict_segment_labels(mobj, feats, preds)
        except Exception as e:
            typer.echo(f"Failed to load/label with model {model}: {e}")

    if labels is not None:
        paired = list(zip(preds, labels))
        typer.echo(f"Detected {len(preds)} segments: {paired}")
    else:
        typer.echo(f"Detected {len(preds)} segments: {preds}")

    if out:
        with open(out, "w") as fh:
            json.dump({"predictions": preds, "labels": labels}, fh)


@app.command()
def eval_synthetic(
    duration: float = 10.0, fs: int = 100, threshold: float = 0.5, min_len: int = 5
):
    """Run detection on synthetic data and print evaluation against ground truth."""
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)
    feats = compute_features_from_sequence(seq)
    preds = detect_segments(
        feats, method="threshold", threshold=threshold, min_len=min_len
    )

    gt = [(s, e) for s, e, _ in seq.segments]
    res = evaluate_detection(gt, preds)
    typer.echo(res)


@app.command()
def train_synthetic(
    duration: float = 10.0,
    fs: int = 100,
    out: str = "model.joblib",
    model_type: str = "rf",
):
    """Train a simple classifier on one synthetic sequence and save the model.

    Supported model_type: rf, logistic, mlp, xgb (requires xgboost), cnn_multi (requires tensorflow).
    For CNN models, uses windowing and augmentation.
    """
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)

    if model_type in ("cnn", "cnn_multi"):
        from .classify import train_cnn_from_sequence, save_model

        model_obj = train_cnn_from_sequence(
            seq,
            model_type=model_type,
            window_s=1.0,
            hop_s=0.5,
            fs=fs,
            balance="oversample",
            augment=1,
            epochs=10,
            batch_size=8,
        )
        save_model(model_obj, out)
        typer.echo(f"Saved CNN model to {out}")
    else:
        feats = compute_features_from_sequence(seq)
        from .classify import (
            build_training_data_from_sequence,
            train_classifier,
            save_model,
        )

        X, y = build_training_data_from_sequence(seq, feats)
        cv = min(3, max(2, len(X)))
        try:
            model_obj = train_classifier(X, y, model_type=model_type, cv=cv)
        except ValueError:
            model_obj = train_classifier(X, y, model_type=model_type, cv=2)
        save_model(model_obj, out)
        typer.echo(f"Saved model to {out}")


if __name__ == "__main__":
    app()
