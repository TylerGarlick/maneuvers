"""Simple preprocessing and feature extraction helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def accel_magnitude(accel: np.ndarray) -> np.ndarray:
    """Return L2 norm of accelerometer axes per sample."""
    return np.linalg.norm(accel, axis=1)


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average (causal)"""
    if window <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def compute_features_from_sequence(seq) -> pd.DataFrame:
    """Compute a small set of features used by the baseline detector.

    Features:
    - accel_mag: magnitude of acceleration
    - accel_mag_smooth: moving-average smoothed magnitude
    - gyro_mag: magnitude of angular rate
    """
    accel_mag = accel_magnitude(seq.accel)
    gyro_mag = np.linalg.norm(seq.gyro, axis=1)

    # Smooth - pad to keep same length
    smooth = moving_average(accel_mag, window=7)
    pad = np.full(len(accel_mag) - len(smooth), smooth[0])
    accel_smooth = np.concatenate([pad, smooth])

    df = pd.DataFrame(
        {
            "t": seq.timestamps,
            "accel_mag": accel_mag,
            "accel_smooth": accel_smooth,
            "gyro_mag": gyro_mag,
        }
    )
    return df
