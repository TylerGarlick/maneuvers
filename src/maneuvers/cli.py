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
def plot_3d(
    duration: float = 10.0,
    fs: int = 100,
    threshold: float = 0.5,
    min_len: int = 5,
    out: str = "flight_path_3d.png",
    model: Optional[str] = None,
    interactive: bool = False,
):
    """Generate synthetic sequence, detect maneuvers, and plot 3D flight path."""
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

    from .visualize import plot_3d_segments
    plot_3d_segments(seq, preds, labels, out, interactive)
    typer.echo(f"Saved 3D plot to {out}")


@app.command()
def export_detect(
    duration: float = 10.0,
    fs: int = 100,
    threshold: float = 0.5,
    min_len: int = 5,
    out: str = "maneuvers_export.json",
    model: Optional[str] = None,
):
    """Generate synthetic sequence, detect maneuvers, label with model, score, and export."""
    seq = generate_synthetic_sequence(duration_s=duration, fs=fs)
    feats = compute_features_from_sequence(seq)
    preds = detect_segments(
        feats, method="threshold", threshold=threshold, min_len=min_len
    )

    labels = None
    scores = None
    if model:
        try:
            from .classify import load_model, predict_segment_labels
            from .export import score_segment

            mobj = load_model(model)
            labels = predict_segment_labels(mobj, feats, preds)
            scores = [score_segment(seq, seg, mobj) for seg in preds]
        except Exception as e:
            typer.echo(f"Failed to load/label/score with model {model}: {e}")

    from .export import export_segments
    export_segments(seq, preds, labels, scores, out)
    typer.echo(f"Exported {len(preds)} segments to {out}")


@app.command()
def train_synthetic(
    duration: float = 10.0,
    fs: int = 100,
    out: str = "model.joblib",
    model_type: str = "rf",
    num_sequences: int = 10,
    use_maneuvers_small: bool = True,
):
    """Train a classifier on multiple synthetic sequences and optionally maneuvers_small dataset.

    Supported model_type: rf, logistic, mlp, xgb (requires xgboost), cnn_multi (requires tensorflow).
    For CNN models, uses windowing and augmentation.
    """
    from pathlib import Path
    import json
    
    sequences = []
    
    # Generate multiple synthetic sequences
    for i in range(num_sequences):
        seq = generate_synthetic_sequence(duration_s=duration, fs=fs, seed=i)
        sequences.append(seq)
    
    # Load maneuvers_small if available
    if use_maneuvers_small:
        outdir = Path("examples/datasets/maneuvers_small")
        manifest_path = outdir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as fh:
                manifest = json.load(fh)
            from .data.loader import from_csv
            for item in manifest:
                seq = from_csv(item["file"])
                seq.segments = item["segments"]
                sequences.append(seq)

    if model_type in ("cnn", "cnn_multi"):
        from .classify import train_cnn_from_sequence, save_model

        # Use the first sequence for CNN training (as before)
        model_obj = train_cnn_from_sequence(
            sequences[0],
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
        from .classify import (
            build_training_data_from_sequence,
            train_classifier,
            save_model,
        )

        # Build training data from all sequences
        X_list = []
        y_list = []
        for seq in sequences:
            feats = compute_features_from_sequence(seq)
            X_seq, y_seq = build_training_data_from_sequence(seq, feats)
            X_list.append(X_seq)
            y_list.extend(y_seq)
        
        X = np.vstack(X_list) if X_list else np.array([])
        y = y_list
        
        if len(X) == 0:
            typer.echo("No training data generated")
            return
        
        # Use grid search for tuning
        param_grid = None
        if model_type == "rf":
            param_grid = {"clf__n_estimators": [50, 100], "clf__max_depth": [None, 10]}
        
        cv = min(5, max(2, len(np.unique(y))))
        model_obj = train_classifier(X, y, model_type=model_type, cv=cv, param_grid=param_grid)
        
        # Add metadata
        model_obj["training_params"] = {
            "duration": duration,
            "fs": fs,
            "num_sequences": num_sequences,
            "use_maneuvers_small": use_maneuvers_small,
            "model_type": model_type,
        }
        
        save_model(model_obj, out)
        typer.echo(f"Saved model to {out} with CV scores: {model_obj.get('cv_scores', {})}")


if __name__ == "__main__":
    app()
