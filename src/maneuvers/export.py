"""Utilities for exporting detected maneuvers with scores and metadata."""

from __future__ import annotations
import json
import csv
from typing import List, Tuple
from pathlib import Path

import numpy as np
from .data.loader import Sequence


def export_segments(
    seq: Sequence,
    segments: List[Tuple[int, int]],
    labels: List[str] = None,
    scores: List[float] = None,
    out_path: str = "maneuvers_export.json",
) -> None:
    """Export detected segments with labels, scores, and timing info to JSON or CSV.
    
    Args:
        seq: The Sequence object with timestamps.
        segments: List of (start_idx, end_idx) tuples.
        labels: Optional list of maneuver labels.
        scores: Optional list of quality scores (0-1).
        out_path: Output file path (.json or .csv).
    """
    if labels is None:
        labels = ["unknown"] * len(segments)
    if scores is None:
        scores = [1.0] * len(segments)
    
    data = []
    for i, (s, e) in enumerate(segments):
        start_time = seq.timestamps[s] if s < len(seq.timestamps) else 0.0
        end_time = seq.timestamps[e-1] if e-1 < len(seq.timestamps) else seq.timestamps[-1]
        duration = end_time - start_time
        data.append({
            "label": labels[i],
            "start_idx": s,
            "end_idx": e,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "score": scores[i],
        })
    
    path = Path(out_path)
    if path.suffix.lower() == ".json":
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
    elif path.suffix.lower() == ".csv":
        with open(path, "w", newline="") as fh:
            if data:
                writer = csv.DictWriter(fh, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    else:
        raise ValueError("Unsupported file extension; use .json or .csv")


def score_segment(seq: Sequence, segment: Tuple[int, int], model=None) -> float:
    """Compute a quality score for a segment using model probability or fallback.
    
    Args:
        seq: The Sequence object.
        segment: (start_idx, end_idx) tuple.
        model: Optional trained model dict for probability scoring.
    
    Returns:
        Score between 0 and 1.
    """
    if model is None:
        # Fallback: simple heuristic based on accel magnitude variance
        s, e = segment
        accel_mag = np.linalg.norm(seq.accel[s:e], axis=1)
        if len(accel_mag) > 1:
            score = min(1.0, np.std(accel_mag) / 0.5)  # normalize by expected std
        else:
            score = 0.5
        return score
    
    # Use model probability
    from .preprocessing import compute_features_from_sequence
    from .classify import segment_aggregated_features
    
    feats = compute_features_from_sequence(seq)
    feat_vec = segment_aggregated_features(feats, segment).reshape(1, -1)
    
    try:
        probs = model["model"].predict_proba(feat_vec)
        # Take max probability as score
        score = np.max(probs)
    except:
        score = 0.5  # fallback
    
    return float(score)