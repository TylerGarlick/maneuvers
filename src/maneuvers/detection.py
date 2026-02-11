"""Detection and segmentation algorithms."""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------


def merge_close_segments(
    segments: List[Tuple[int, int]], min_gap: int
) -> List[Tuple[int, int]]:
    """Merge segments whose gap is smaller than *min_gap* samples."""
    if not segments:
        return []
    merged: List[Tuple[int, int]] = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e < min_gap:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def split_long_segments(
    segments: List[Tuple[int, int]], max_len: int
) -> List[Tuple[int, int]]:
    """Split segments longer than *max_len* into equal-sized pieces."""
    if max_len <= 0:
        return list(segments)
    result: List[Tuple[int, int]] = []
    for s, e in segments:
        length = e - s
        if length <= max_len:
            result.append((s, e))
        else:
            n_pieces = int(np.ceil(length / max_len))
            piece_len = int(np.ceil(length / n_pieces))
            for k in range(n_pieces):
                ps = s + k * piece_len
                pe = min(s + (k + 1) * piece_len, e)
                if pe > ps:
                    result.append((ps, pe))
    return result


def filter_short_segments(
    segments: List[Tuple[int, int]], min_len: int
) -> List[Tuple[int, int]]:
    """Remove segments shorter than *min_len*."""
    return [(s, e) for s, e in segments if (e - s) >= min_len]


def apply_post_processing(
    segments: List[Tuple[int, int]],
    *,
    min_gap: int = 0,
    max_len: int = 0,
    min_len: int = 0,
) -> List[Tuple[int, int]]:
    """Apply a standard post-processing pipeline to detected segments.

    1. Merge segments separated by less than *min_gap* samples.
    2. Split segments longer than *max_len* samples.
    3. Drop segments shorter than *min_len* samples.
    """
    segs = list(segments)
    if min_gap > 0:
        segs = merge_close_segments(segs, min_gap)
    if max_len > 0:
        segs = split_long_segments(segs, max_len)
    if min_len > 0:
        segs = filter_short_segments(segs, min_len)
    return segs


# ---------------------------------------------------------------------------
# Threshold detection (original baseline)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CUSUM change-point detection
# ---------------------------------------------------------------------------


def cusum_segment(
    signal: np.ndarray,
    threshold: float = 1.0,
    drift: float = 0.5,
    min_len: int = 5,
) -> List[Tuple[int, int]]:
    """Cumulative Sum (CUSUM) change-point detector.

    Detects upward shifts in *signal* relative to its mean.  The algorithm
    accumulates positive deviations (minus a *drift* allowance) and triggers
    when the cumulative sum exceeds *threshold*.

    Parameters
    ----------
    signal : 1-D array
    threshold : CUSUM decision threshold (higher = fewer, longer detections).
    drift : allowance subtracted each step (controls sensitivity).
    min_len : minimum segment length to keep.
    """
    mu = np.mean(signal)
    N = len(signal)
    g_pos = np.zeros(N)
    in_segment = False
    segments: List[Tuple[int, int]] = []
    start = 0

    for i in range(1, N):
        g_pos[i] = max(0.0, g_pos[i - 1] + (signal[i] - mu) - drift)
        if not in_segment and g_pos[i] > threshold:
            in_segment = True
            start = i
        elif in_segment and g_pos[i] <= 0.0:
            in_segment = False
            if (i - start) >= min_len:
                segments.append((start, i))

    # Close open segment at end of signal
    if in_segment and (N - start) >= min_len:
        segments.append((start, N))

    return segments


# ---------------------------------------------------------------------------
# Energy-based detection
# ---------------------------------------------------------------------------


def energy_segment(
    seq,
    threshold: Optional[float] = None,
    min_len: int = 5,
    fs: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Detect maneuvers from kinetic + potential energy changes.

    Works on Sequence objects that have *vel* and optionally *pos* fields.
    Falls back to accel magnitude if velocity is unavailable.

    The energy signal is the sum of specific kinetic energy (0.5 * |v|^2) and
    specific potential energy (g * z).  Segments where the rate of change of
    total energy exceeds *threshold* are returned.
    """
    g = 9.80665
    if seq.vel is not None:
        ke = 0.5 * np.sum(seq.vel**2, axis=1)
    else:
        ke = 0.5 * np.sum(seq.accel**2, axis=1)

    pe = np.zeros(len(ke))
    if seq.pos is not None and seq.pos.shape[1] >= 3:
        pe = g * seq.pos[:, 2]  # z-up component

    total_energy = ke + pe

    # Rate of change of energy
    if fs is None and len(seq.timestamps) > 1:
        dt = seq.timestamps[1] - seq.timestamps[0]
        if dt > 0:
            fs = int(round(1.0 / dt))
    if fs is None or fs <= 0:
        fs = 100

    energy_rate = np.zeros_like(total_energy)
    energy_rate[1:] = np.abs(np.diff(total_energy)) * fs

    # Smooth the energy rate
    window = max(3, fs // 10)
    kernel = np.ones(window) / window
    energy_rate_smooth = np.convolve(energy_rate, kernel, mode="same")

    if threshold is None:
        threshold = np.mean(energy_rate_smooth) + 2.0 * np.std(energy_rate_smooth)

    return threshold_segment(energy_rate_smooth, thr=threshold, min_len=min_len)


# ---------------------------------------------------------------------------
# Position / velocity / orientation-based detection
# ---------------------------------------------------------------------------


def orientation_segment(
    seq,
    threshold: Optional[float] = None,
    min_len: int = 5,
    fs: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Detect maneuvers from orientation change rates.

    Computes turn rate (heading rate), pitch rate, and roll rate from the
    Sequence *orient* field (roll, pitch, heading).  Falls back to gyro
    magnitude if orientation is unavailable.
    """
    if fs is None and len(seq.timestamps) > 1:
        dt = seq.timestamps[1] - seq.timestamps[0]
        if dt > 0:
            fs = int(round(1.0 / dt))
    if fs is None or fs <= 0:
        fs = 100

    if seq.orient is not None:
        # orient columns: roll, pitch, heading (degrees or radians)
        orient_rate = np.zeros_like(seq.orient)
        orient_rate[1:] = np.abs(np.diff(seq.orient, axis=0)) * fs
        # Handle heading wrap-around (e.g. 359->1 = 2, not 358)
        if seq.orient.shape[1] >= 3:
            heading_diff = np.abs(np.diff(seq.orient[:, 2]))
            heading_diff = np.minimum(heading_diff, 360.0 - heading_diff)
            orient_rate[1:, 2] = heading_diff * fs
        composite = np.linalg.norm(orient_rate, axis=1)
    elif seq.vel is not None:
        # Derive turn rate from velocity heading changes
        heading = np.arctan2(seq.vel[:, 0], seq.vel[:, 1])
        heading_rate = np.zeros_like(heading)
        heading_rate[1:] = np.abs(np.diff(heading)) * fs
        # Wrap
        heading_rate = np.minimum(heading_rate, (2 * np.pi - heading_rate))
        # Climb rate from vertical velocity
        climb_rate = np.abs(seq.vel[:, 2]) if seq.vel.shape[1] >= 3 else 0.0
        composite = heading_rate + climb_rate
    else:
        composite = np.linalg.norm(seq.gyro, axis=1)

    # Smooth
    window = max(3, fs // 10)
    kernel = np.ones(window) / window
    composite = np.convolve(composite, kernel, mode="same")

    if threshold is None:
        threshold = np.mean(composite) + 2.0 * np.std(composite)

    return threshold_segment(composite, thr=threshold, min_len=min_len)


# ---------------------------------------------------------------------------
# Multi-signal fusion
# ---------------------------------------------------------------------------


def fusion_segment(
    seq,
    features=None,
    vote_threshold: int = 2,
    min_len: int = 5,
    fs: Optional[int] = None,
    **kwargs,
) -> List[Tuple[int, int]]:
    """Fuse multiple detection signals via a voting scheme.

    Runs threshold, energy, and orientation detectors independently, then
    keeps time indices where at least *vote_threshold* detectors agree that
    a maneuver is occurring.
    """
    N = len(seq.timestamps)
    votes = np.zeros(N, dtype=int)

    # 1) Threshold on accel
    if features is not None and "accel_smooth" in features.columns:
        thr_kw = float(kwargs.get("threshold", 0.5))
        for s, e in threshold_segment(
            features["accel_smooth"].values, thr=thr_kw, min_len=1
        ):
            votes[s:e] += 1

    # 2) Energy-based
    try:
        for s, e in energy_segment(seq, min_len=1, fs=fs):
            votes[s:e] += 1
    except Exception:
        pass

    # 3) Orientation-based
    try:
        for s, e in orientation_segment(seq, min_len=1, fs=fs):
            votes[s:e] += 1
    except Exception:
        pass

    # 4) CUSUM on accel magnitude
    accel_mag = np.linalg.norm(seq.accel, axis=1)
    try:
        for s, e in cusum_segment(accel_mag, min_len=1):
            votes[s:e] += 1
    except Exception:
        pass

    # Extract segments from vote array
    mask = votes >= vote_threshold
    return threshold_segment(mask.astype(float), thr=0.5, min_len=min_len)


# ---------------------------------------------------------------------------
# Learned (sliding-window) detection
# ---------------------------------------------------------------------------


def learned_segment(
    model,
    features,
    seq,
    window: int = 50,
    hop: int = 25,
    min_len: int = 5,
) -> List[Tuple[int, int]]:
    """Sliding-window binary classifier for maneuver detection.

    *model* should expose a `predict` method that accepts a 2-D array of
    shape (n_windows, n_features) and returns 1 (maneuver) or 0 (no
    maneuver) per window.

    Features per window: mean, std, max, min of each column in *features*
    (a DataFrame) restricted to the window.
    """
    import pandas as pd

    if isinstance(features, pd.DataFrame):
        data = features.select_dtypes(include=[np.number]).values
    else:
        data = np.asarray(features)

    N = data.shape[0]
    window_features = []
    window_positions = []

    for start in range(0, max(1, N - window + 1), hop):
        end = min(start + window, N)
        chunk = data[start:end]
        feat = np.concatenate(
            [
                chunk.mean(axis=0),
                chunk.std(axis=0),
                chunk.max(axis=0),
                chunk.min(axis=0),
            ]
        )
        window_features.append(feat)
        window_positions.append((start, end))

    if not window_features:
        return []

    X = np.vstack(window_features)
    preds = model.predict(X)

    # Translate window-level predictions to a sample-level mask
    mask = np.zeros(N, dtype=int)
    for (s, e), p in zip(window_positions, preds):
        if p == 1:
            mask[s:e] = 1

    return threshold_segment(mask.astype(float), thr=0.5, min_len=min_len)


def train_learned_detector(
    sequences,
    fs: int = 100,
    window_s: float = 0.5,
    hop_s: float = 0.25,
):
    """Train a sliding-window binary detector from labeled Sequences.

    Returns a fitted sklearn model that predicts 1 (maneuver) / 0 (background).
    """
    from sklearn.ensemble import RandomForestClassifier

    from .preprocessing import compute_features_from_sequence

    window = int(window_s * fs)
    hop = int(hop_s * fs)

    X_all = []
    y_all = []

    for seq in sequences:
        feats = compute_features_from_sequence(seq)
        data = feats.select_dtypes(include=[np.number]).values
        N = data.shape[0]
        gt_segs = seq.segments or []

        for start in range(0, max(1, N - window + 1), hop):
            end = min(start + window, N)
            chunk = data[start:end]
            feat = np.concatenate(
                [
                    chunk.mean(axis=0),
                    chunk.std(axis=0),
                    chunk.max(axis=0),
                    chunk.min(axis=0),
                ]
            )
            # Label: 1 if >=50% overlap with any ground-truth segment
            label = 0
            for gs, ge, _ in gt_segs:
                overlap = max(0, min(ge, end) - max(gs, start))
                if overlap >= 0.5 * (end - start):
                    label = 1
                    break
            X_all.append(feat)
            y_all.append(label)

    X = np.vstack(X_all)
    y = np.array(y_all)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def detect_segments(features, method: str = "threshold", seq=None, **kwargs):
    """Dispatch to a detection method.

    Supported methods
    -----------------
    threshold : accel_smooth > thr  (original baseline)
    cusum     : CUSUM change-point on accel magnitude
    energy    : kinetic + potential energy rate of change
    orientation : orientation / turn-rate based
    fusion    : multi-signal voting (combines threshold + cusum + energy + orientation)
    learned   : sliding-window binary classifier (pass ``model`` in kwargs)

    Post-processing kwargs (applied to all methods):
        post_min_gap  : merge segments closer than this (samples)
        post_max_len  : split segments longer than this (samples)
        post_min_len  : drop segments shorter than this (samples)
    """
    min_len = int(kwargs.get("min_len", 5))
    fs = kwargs.get("fs")

    if method == "threshold":
        thr = float(kwargs.get("threshold", 0.5))
        signal = features["accel_smooth"].values
        segments = threshold_segment(signal, thr=thr, min_len=min_len)

    elif method == "cusum":
        thr = float(kwargs.get("threshold", 1.0))
        drift = float(kwargs.get("drift", 0.5))
        signal = features["accel_smooth"].values
        segments = cusum_segment(signal, threshold=thr, drift=drift, min_len=min_len)

    elif method == "energy":
        if seq is None:
            raise ValueError("energy method requires a Sequence (pass seq=)")
        thr = kwargs.get("threshold")
        if thr is not None:
            thr = float(thr)
        segments = energy_segment(seq, threshold=thr, min_len=min_len, fs=fs)

    elif method == "orientation":
        if seq is None:
            raise ValueError("orientation method requires a Sequence (pass seq=)")
        thr = kwargs.get("threshold")
        if thr is not None:
            thr = float(thr)
        segments = orientation_segment(seq, threshold=thr, min_len=min_len, fs=fs)

    elif method == "fusion":
        if seq is None:
            raise ValueError("fusion method requires a Sequence (pass seq=)")
        vote_thr = int(kwargs.get("vote_threshold", 2))
        segments = fusion_segment(
            seq,
            features=features,
            vote_threshold=vote_thr,
            min_len=min_len,
            fs=fs,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in (
                    "min_len",
                    "fs",
                    "vote_threshold",
                    "post_min_gap",
                    "post_max_len",
                    "post_min_len",
                )
            },
        )

    elif method == "learned":
        model = kwargs.get("model")
        if model is None:
            raise ValueError("learned method requires a model (pass model=)")
        if seq is None:
            raise ValueError("learned method requires a Sequence (pass seq=)")
        window = int(kwargs.get("window", 50))
        hop = int(kwargs.get("hop", 25))
        segments = learned_segment(
            model, features, seq, window=window, hop=hop, min_len=min_len
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply optional post-processing
    post_min_gap = int(kwargs.get("post_min_gap", 0))
    post_max_len = int(kwargs.get("post_max_len", 0))
    post_min_len = int(kwargs.get("post_min_len", 0))
    if post_min_gap > 0 or post_max_len > 0 or post_min_len > 0:
        segments = apply_post_processing(
            segments,
            min_gap=post_min_gap,
            max_len=post_max_len,
            min_len=post_min_len,
        )

    return segments
