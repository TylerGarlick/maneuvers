"""Detection and segmentation algorithms (baseline)."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


def threshold_segment(
    signal: np.ndarray, thr: float, min_len: int = 5
) -> List[Tuple[int, int]]:
    """Simple threshold-based segmentation on a 1D signal.

    Returns list of (start_idx, end_idx) where end_idx is exclusive.
    """
    mask = signal > thr
    segments: List[Tuple[int, int]] = []
    i = 0
    N = len(mask)
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            if (j - i) >= min_len:
                segments.append((i, j))
            i = j
        else:
            i += 1
    return segments


def detect_segments(features, method: str = "threshold", **kwargs):
    """Dispatch to a detection method.

    method: 'threshold' uses accel_smooth > thr
    """
    if method == "threshold":
        thr = float(kwargs.get("threshold", 0.5))
        min_len = int(kwargs.get("min_len", 5))
        signal = features["accel_smooth"].values
        return threshold_segment(signal, thr=thr, min_len=min_len)
    else:
        raise ValueError(f"Unknown method: {method}")
