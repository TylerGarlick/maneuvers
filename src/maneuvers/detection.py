"""Detection and segmentation algorithms (baseline)."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd


# Numerical stability constants
MIN_SIGNAL_RANGE = 1e-9  # Minimum signal range for normalization


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


def merge_nearby_segments(
    segments: List[Tuple[int, int]], max_gap: int = 10
) -> List[Tuple[int, int]]:
    """Merge segments that are close together (likely part of same maneuver).
    
    Args:
        segments: List of (start, end) tuples
        max_gap: Maximum gap between segments to merge
        
    Returns:
        Merged list of segments
    """
    if not segments:
        return []
    
    # Sort by start index
    sorted_segs = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segs[0]]
    
    for curr_start, curr_end in sorted_segs[1:]:
        prev_start, prev_end = merged[-1]
        
        # If gap is small enough, merge
        if curr_start - prev_end <= max_gap:
            merged[-1] = (prev_start, curr_end)
        else:
            merged.append((curr_start, curr_end))
    
    return merged


def adaptive_threshold_segment(
    signal: np.ndarray, 
    window_size: int = 50,
    n_std: float = 2.0,
    min_len: int = 5
) -> List[Tuple[int, int]]:
    """Adaptive threshold segmentation using local statistics.
    
    Uses a sliding window to compute local mean and std, then
    detects segments where signal exceeds mean + n_std * std.
    
    Args:
        signal: 1D signal to segment
        window_size: Size of sliding window for local statistics
        n_std: Number of standard deviations above local mean
        min_len: Minimum segment length
        
    Returns:
        List of (start_idx, end_idx) segments
    """
    if len(signal) < window_size:
        # Fallback to global threshold
        global_mean = np.mean(signal)
        global_std = np.std(signal)
        thr = global_mean + n_std * global_std
        return threshold_segment(signal, thr, min_len)
    
    N = len(signal)
    threshold_curve = np.zeros(N)
    
    # Compute local statistics using sliding window
    for i in range(N):
        start = max(0, i - window_size // 2)
        end = min(N, i + window_size // 2)
        window = signal[start:end]
        local_mean = np.mean(window)
        local_std = np.std(window)
        threshold_curve[i] = local_mean + n_std * local_std
    
    # Detect segments where signal exceeds adaptive threshold
    mask = signal > threshold_curve
    segments: List[Tuple[int, int]] = []
    i = 0
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


def multi_signal_fusion_detect(
    features: pd.DataFrame,
    accel_weight: float = 0.6,
    gyro_weight: float = 0.3,
    jerk_weight: float = 0.1,
    threshold: float = 0.5,
    min_len: int = 5,
    merge_gap: int = 10
) -> List[Tuple[int, int]]:
    """Detect segments by fusing multiple sensor signals.
    
    Combines normalized acceleration, gyro, and jerk magnitudes
    with configurable weights to create a composite detection signal.
    
    Args:
        features: DataFrame with accel_mag, gyro_mag, jerk_mag columns
        accel_weight: Weight for acceleration signal
        gyro_weight: Weight for gyro signal  
        jerk_weight: Weight for jerk signal
        threshold: Detection threshold on composite signal
        min_len: Minimum segment length
        merge_gap: Maximum gap for merging nearby segments
        
    Returns:
        List of detected segments
    """
    # Normalize each signal to [0, 1] range
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < MIN_SIGNAL_RANGE:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    accel_norm = normalize(features['accel_mag'].values)
    gyro_norm = normalize(features['gyro_mag'].values)
    
    # Use jerk if available
    if 'jerk_mag' in features.columns:
        jerk_norm = normalize(features['jerk_mag'].values)
    else:
        jerk_norm = np.zeros_like(accel_norm)
    
    # Weighted fusion
    composite = (
        accel_weight * accel_norm + 
        gyro_weight * gyro_norm + 
        jerk_weight * jerk_norm
    )
    
    # Smooth the composite signal
    from maneuvers.preprocessing import moving_average
    composite_smooth = moving_average(composite, window=5)
    
    # Detect segments
    segments = threshold_segment(composite_smooth, threshold, min_len)
    
    # Merge nearby segments
    segments = merge_nearby_segments(segments, max_gap=merge_gap)
    
    return segments


def variance_based_changepoint_detect(
    signal: np.ndarray,
    window_size: int = 20,
    var_threshold: float = 2.0,
    min_len: int = 5
) -> List[Tuple[int, int]]:
    """Detect maneuvers using variance-based change-point detection.
    
    Computes sliding window variance and detects regions where
    variance significantly increases (indicating dynamic maneuvers).
    
    Args:
        signal: 1D signal to analyze
        window_size: Size of sliding window for variance computation
        var_threshold: Variance threshold multiplier
        min_len: Minimum segment length
        
    Returns:
        List of detected segments
    """
    N = len(signal)
    if N < window_size:
        return []
    
    # Compute sliding window variance
    variances = np.zeros(N)
    for i in range(N):
        start = max(0, i - window_size // 2)
        end = min(N, i + window_size // 2)
        window = signal[start:end]
        variances[i] = np.var(window)
    
    # Detect high variance regions
    var_mean = np.mean(variances)
    var_std = np.std(variances)
    thr = var_mean + var_threshold * var_std
    
    return threshold_segment(variances, thr, min_len)


def detect_segments(features, method: str = "threshold", **kwargs):
    """Dispatch to a detection method.

    Available methods:
    - 'threshold': uses accel_smooth > thr (original baseline)
    - 'adaptive': adaptive threshold based on local statistics
    - 'fusion': multi-signal fusion (accel + gyro + jerk)
    - 'variance': variance-based change-point detection
    """
    if method == "threshold":
        thr = float(kwargs.get("threshold", 0.5))
        min_len = int(kwargs.get("min_len", 5))
        signal = features["accel_smooth"].values
        segments = threshold_segment(signal, thr=thr, min_len=min_len)
        
        # Optionally merge nearby segments
        merge_gap = int(kwargs.get("merge_gap", 0))
        if merge_gap > 0:
            segments = merge_nearby_segments(segments, max_gap=merge_gap)
        
        return segments
    
    elif method == "adaptive":
        window_size = int(kwargs.get("window_size", 50))
        n_std = float(kwargs.get("n_std", 2.0))
        min_len = int(kwargs.get("min_len", 5))
        signal = features["accel_smooth"].values
        return adaptive_threshold_segment(signal, window_size, n_std, min_len)
    
    elif method == "fusion":
        return multi_signal_fusion_detect(
            features,
            accel_weight=float(kwargs.get("accel_weight", 0.6)),
            gyro_weight=float(kwargs.get("gyro_weight", 0.3)),
            jerk_weight=float(kwargs.get("jerk_weight", 0.1)),
            threshold=float(kwargs.get("threshold", 0.5)),
            min_len=int(kwargs.get("min_len", 5)),
            merge_gap=int(kwargs.get("merge_gap", 10))
        )
    
    elif method == "variance":
        window_size = int(kwargs.get("window_size", 20))
        var_threshold = float(kwargs.get("var_threshold", 2.0))
        min_len = int(kwargs.get("min_len", 5))
        signal = features["accel_smooth"].values
        return variance_based_changepoint_detect(signal, window_size, var_threshold, min_len)
    
    else:
        raise ValueError(f"Unknown method: {method}")
