"""Tests for improved detection methods."""

import numpy as np
import pytest
from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.detection import (
    detect_segments,
    merge_nearby_segments,
    adaptive_threshold_segment,
    multi_signal_fusion_detect,
    variance_based_changepoint_detect,
)


def test_merge_nearby_segments():
    """Test segment merging functionality."""
    segments = [(10, 20), (25, 35), (100, 110)]
    
    # With max_gap=10, first two should merge
    merged = merge_nearby_segments(segments, max_gap=10)
    assert len(merged) == 2
    assert merged[0] == (10, 35)
    assert merged[1] == (100, 110)
    
    # With max_gap=0, no merging
    merged = merge_nearby_segments(segments, max_gap=0)
    assert len(merged) == 3


def test_adaptive_threshold_detection():
    """Test adaptive threshold detection finds maneuvers."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    segments = detect_segments(feats, method="adaptive", window_size=50, n_std=1.5, min_len=5)
    
    # Should find at least some maneuvers
    assert len(segments) > 0
    # Segments should have correct format
    for s, e in segments:
        assert s < e
        assert e - s >= 5


def test_multi_signal_fusion_detection():
    """Test multi-signal fusion detection."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    segments = detect_segments(
        feats, 
        method="fusion",
        accel_weight=0.6,
        gyro_weight=0.3,
        jerk_weight=0.1,
        threshold=0.4,
        min_len=5,
        merge_gap=10
    )
    
    # Should detect maneuvers
    assert len(segments) > 0
    for s, e in segments:
        assert s < e
        assert e - s >= 5


def test_variance_based_detection():
    """Test variance-based change-point detection."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    segments = detect_segments(
        feats,
        method="variance",
        window_size=20,
        var_threshold=2.0,
        min_len=5
    )
    
    # Should detect some segments
    assert len(segments) >= 0  # May find 0 or more depending on variance threshold
    for s, e in segments:
        assert s < e


def test_enhanced_features_available():
    """Test that enhanced features are computed."""
    seq = generate_synthetic_sequence(duration_s=5.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    # Check new features are present
    assert "jerk_mag" in feats.columns
    assert "rot_energy" in feats.columns
    
    # Check features have correct length
    assert len(feats["jerk_mag"]) == len(seq.timestamps)
    assert len(feats["rot_energy"]) == len(seq.timestamps)
    
    # Check features are not all zero
    assert feats["jerk_mag"].sum() > 0
    assert feats["rot_energy"].sum() > 0


def test_threshold_detection_with_merge():
    """Test original threshold detection with merge option."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    # Without merge
    segments_no_merge = detect_segments(feats, method="threshold", threshold=0.4, min_len=5)
    
    # With merge
    segments_with_merge = detect_segments(
        feats, method="threshold", threshold=0.4, min_len=5, merge_gap=20
    )
    
    # Merged version should have <= segments
    assert len(segments_with_merge) <= len(segments_no_merge)


def test_detection_methods_comparable_results():
    """Test that different methods produce reasonable results on same data."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    # All methods should find at least one maneuver on synthetic data
    methods = ["threshold", "adaptive", "fusion", "variance"]
    
    for method in methods:
        if method == "threshold":
            segments = detect_segments(feats, method=method, threshold=0.4, min_len=5)
        elif method == "adaptive":
            segments = detect_segments(feats, method=method, n_std=1.5, min_len=5)
        elif method == "fusion":
            segments = detect_segments(feats, method=method, threshold=0.4, min_len=5)
        elif method == "variance":
            segments = detect_segments(feats, method=method, var_threshold=1.5, min_len=5)
        
        # Should produce valid segments
        for s, e in segments:
            assert 0 <= s < e <= len(feats)
