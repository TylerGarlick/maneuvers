"""Simple preprocessing and feature extraction helpers."""

from __future__ import annotations
import numpy as np
import pandas as pd


# Numerical stability constants
MIN_DT = 1e-6  # Minimum time delta to avoid division by zero


def accel_magnitude(accel: np.ndarray) -> np.ndarray:
    """Return L2 norm of accelerometer axes per sample."""
    return np.linalg.norm(accel, axis=1)


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average (causal)"""
    if window <= 1:
        return x
    # If the sequence is shorter than the window, return the mean as a flat signal
    if len(x) < window:
        return np.full_like(x, np.mean(x))
    kernel = np.ones(window, dtype=float) / float(window)
    # mode='same' returns an array of the same length as x
    return np.convolve(x, kernel, mode="same")


def compute_jerk(accel: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Compute jerk (derivative of acceleration) magnitude.
    
    Jerk is the rate of change of acceleration and can indicate
    sudden maneuver transitions.
    """
    if len(accel) < 2:
        return np.zeros(len(accel))
    
    dt = np.diff(timestamps)
    dt = np.where(dt > 0, dt, MIN_DT)  # avoid division by zero
    
    # Compute derivative per axis
    jerk = np.zeros_like(accel)
    jerk[1:] = np.diff(accel, axis=0) / dt[:, np.newaxis]
    jerk[0] = jerk[1]  # forward fill first sample
    
    # Return magnitude
    return np.linalg.norm(jerk, axis=1)


def compute_rotational_energy(gyro: np.ndarray) -> np.ndarray:
    """Compute rotational energy from gyroscope data.
    
    Higher rotational energy indicates turning maneuvers.
    """
    # Rotational kinetic energy is proportional to omega^2
    return np.sum(gyro ** 2, axis=1)


def compute_features_from_sequence(seq) -> pd.DataFrame:
    """Compute a small set of features used by the baseline detector.

    Features:
    - accel_mag: magnitude of acceleration
    - accel_mag_smooth: moving-average smoothed magnitude
    - gyro_mag: magnitude of angular rate
    - jerk_mag: magnitude of jerk (acceleration derivative)
    - rot_energy: rotational energy from gyroscope
    - pos_mag: magnitude of position (if available)
    - vel_mag: magnitude of velocity (if available)
    - orient_mag: magnitude of orientation (if available)
    """
    accel_mag = accel_magnitude(seq.accel)
    gyro_mag = np.linalg.norm(seq.gyro, axis=1)
    jerk_mag = compute_jerk(seq.accel, seq.timestamps)
    rot_energy = compute_rotational_energy(seq.gyro)

    # Smooth - pad to keep same length
    smooth = moving_average(accel_mag, window=7)
    pad = np.full(len(accel_mag) - len(smooth), smooth[0])
    accel_smooth = np.concatenate([pad, smooth])

    features = {
        "t": seq.timestamps,
        "accel_mag": accel_mag,
        "accel_smooth": accel_smooth,
        "gyro_mag": gyro_mag,
        "jerk_mag": jerk_mag,
        "rot_energy": rot_energy,
    }

    if seq.pos is not None:
        features["pos_mag"] = np.linalg.norm(seq.pos, axis=1)

    if seq.vel is not None:
        features["vel_mag"] = np.linalg.norm(seq.vel, axis=1)

    if seq.orient is not None:
        features["orient_mag"] = np.linalg.norm(seq.orient, axis=1)

    df = pd.DataFrame(features)
    return df


# --- Windowing helpers --------------------------------------------------------


def windowed_examples_from_sequence(
    seq, window_s: float = 1.0, hop_s: float = 0.5, fs: int | None = None
):
    """Extract sliding windows from a Sequence and return (X, y, windows)

    - X: np.ndarray shaped (n_windows, seq_len, n_channels) where channels are accel (3) then gyro (3) concatenated
    - y: list of labels (str) for each window; label is the GT segment label if window overlaps GT by >=50%, otherwise 'none'
    - windows: list of (start_idx, end_idx) per window

    The sampling rate `fs` is inferred from timestamps if not provided.
    """
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
