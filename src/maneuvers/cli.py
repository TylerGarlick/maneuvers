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
    preds = detect_segments(feats, method="threshold", threshold=threshold, min_len=min_len)

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
def eval_synthetic(duration: float = 10.0, fs: int = 100, threshold: float = 0.5, min_len: int = 5):
    """Run detection on synthetic data and print evaluation against ground truth."""
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)
    feats = compute_features_from_sequence(seq)
    preds = detect_segments(feats, method="threshold", threshold=threshold, min_len=min_len)

    gt = [(s, e) for s, e, _ in seq.segments]
    res = evaluate_detection(gt, preds)
    typer.echo(res)


@app.command()
def train_synthetic(duration: float = 10.0, fs: int = 100, out: str = "model.joblib"):
    """Train a simple classifier on one synthetic sequence and save the model."""
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)
    feats = compute_features_from_sequence(seq)

    from .classify import build_training_data_from_sequence, train_classifier, save_model

    X, y = build_training_data_from_sequence(seq, feats)
    # choose a conservative cv based on available samples; handle small sample sizes
    cv = min(3, max(2, len(X)))
    try:
        model_obj = train_classifier(X, y, cv=cv)
    except ValueError:
        # fallback to small CV if dataset too small
        model_obj = train_classifier(X, y, cv=2)
    save_model(model_obj, out)
    typer.echo(f"Saved model to {out}")


if __name__ == "__main__":
    app()
