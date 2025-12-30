"""Data loader and synthetic data generator for maneuvers demo."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Sequence:
    timestamps: np.ndarray  # shape (N,)
    accel: np.ndarray  # shape (N, 3)
    gyro: np.ndarray  # shape (N, 3)
    segments: List[Tuple[int, int, str]] = None  # list of (start_idx, end_idx, label)


def generate_synthetic_sequence(
    duration_s: float = 10.0, fs: int = 100, seed: int | None = 0
) -> Sequence:
    """Generate a synthetic flight sequence with a few maneuvers.

    The sequence contains background noise and a few maneuvers represented as
    increases in acceleration magnitude.
    """
    rng = np.random.default_rng(seed)
    N = int(duration_s * fs)
    t = np.linspace(0.0, duration_s, N, endpoint=False)

    # Background sensor noise
    accel = 0.1 * rng.standard_normal((N, 3))
    gyro = 0.01 * rng.standard_normal((N, 3))

    # Insert a few synthetic maneuvers (bursts of acceleration)
    segments = []
    # Two maneuvers at predefined intervals
    maneuvers = [
        (int(1.0 * fs), int(2.2 * fs), "left_roll"),
        (int(4.0 * fs), int(5.0 * fs), "right_roll"),
        (int(7.0 * fs), int(7.7 * fs), "climb"),
    ]
    for s, e, label in maneuvers:
        # skip maneuvers that start after the sequence
        if s >= N:
            continue
        # clip maneuvers that extend past the end
        e = min(e, N)
        length = e - s
        if length <= 0:
            continue
        # make an accel bump in x and z axes
        bump = np.hanning(length)
        accel[s:e, 0] += 1.0 * bump
        accel[s:e, 2] += 0.8 * bump
        gyro[s:e, 1] += 0.3 * bump
        segments.append((s, e, label))

    return Sequence(t, accel, gyro, segments)


def from_csv(path: str) -> Sequence:
    """Minimal CSV loader expected to have columns: t, ax,ay,az, gx,gy,gz"""
    import csv

    import numpy as np

    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("CSV contains no rows")

    t = np.array([float(r["t"]) for r in rows], dtype=float)
    accel = np.vstack([[float(r["ax"]), float(r["ay"]), float(r["az"])] for r in rows])
    gyro = np.vstack([[float(r["gx"]), float(r["gy"]), float(r["gz"])] for r in rows])

    return Sequence(timestamps=t, accel=accel, gyro=gyro, segments=None)
