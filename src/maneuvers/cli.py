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
    duration: float = 10.0, fs: int = 100, threshold: float = 0.5, min_len: int = 5, out: Optional[str] = None
):
    """Generate a synthetic sequence and run baseline detection."""
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)
    feats = compute_features_from_sequence(seq)
    preds = detect_segments(feats, method="threshold", threshold=threshold, min_len=min_len)
    typer.echo(f"Detected {len(preds)} segments: {preds}")
    if out:
        with open(out, "w") as fh:
            json.dump({"predictions": preds}, fh)


@app.command()
def eval_synthetic(duration: float = 10.0, fs: int = 100, threshold: float = 0.5, min_len: int = 5):
    """Run detection on synthetic data and print evaluation against ground truth."""
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)
    feats = compute_features_from_sequence(seq)
    preds = detect_segments(feats, method="threshold", threshold=threshold, min_len=min_len)

    gt = [(s, e) for s, e, _ in seq.segments]
    res = evaluate_detection(gt, preds)
    typer.echo(res)


if __name__ == "__main__":
    app()
