"""Simple preprocessing and feature extraction helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def accel_magnitude(accel: np.ndarray) -> np.ndarray:
    """Return L2 norm of accelerometer axes per sample."""
    return np.linalg.norm(accel, axis=1)


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average (centered `same` convolution). Handles short input gracefully.

    Uses numpy.convolve with 'same' so the output length matches the input and
    very short inputs (len(x) < window) return a reasonable smoothed signal.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) == 0:
        return x
    # If the sequence is shorter than the window, return the mean as a flat signal
    if len(x) < window:
        return np.full_like(x, np.mean(x))
    kernel = np.ones(window, dtype=float) / float(window)
    # mode='same' returns an array of the same length as x
    return np.convolve(x, kernel, mode="same")


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


# --- Windowing helpers --------------------------------------------------------

def windowed_examples_from_sequence(seq, window_s: float = 1.0, hop_s: float = 0.5, fs: int | None = None):
    """Extract sliding windows from a Sequence and return (X, y, windows)

    - X: np.ndarray shaped (n_windows, seq_len, n_channels) where channels are accel (3) then gyro (3) concatenated
    - y: list of labels (str) for each window; label is the GT segment label if window overlaps GT by >=50%, otherwise 'none'
    - windows: list of (start_idx, end_idx) per window

    The sampling rate `fs` is inferred from timestamps if not provided.
    """
    import math

    if fs is None:
        dt = seq.timestamps[1] - seq.timestamps[0]
        fs = int(round(1.0 / dt))

    seq_len = int(round(window_s * fs))
    hop = int(round(hop_s * fs))
    N = seq.accel.shape[0]

    X = []
    y = []
    windows = []

    gt_segments = seq.segments or []

    for start in range(0, max(1, N - seq_len + 1), max(1, hop)):
        end = start + seq_len
        if end > N:
            break
        accel_win = seq.accel[start:end]
        gyro_win = seq.gyro[start:end]
        # stack channels: (seq_len, 6)
        win = np.concatenate([accel_win, gyro_win], axis=1)
        # determine label: if overlap with any GT segment >=50% of window, use that label
        label = "none"
        for s, e, lbl in gt_segments:
            overlap = max(0, min(e, end) - max(s, start))
            if overlap >= 0.5 * seq_len:
                label = lbl
                break
        X.append(win)
        y.append(label)
        windows.append((start, end))

    return np.asarray(X), y, windows