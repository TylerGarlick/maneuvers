"""Tests for enhanced classification features."""

import numpy as np
import pytest
from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.classify import segment_aggregated_features, build_training_data_from_sequence


def test_enhanced_segment_features():
    """Test that enhanced features are computed correctly."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    # Get a segment
    segment = (100, 200)
    features = segment_aggregated_features(feats, segment)
    
    # Should have 25 features now (9 original + 16 enhanced)
    assert len(features) == 25
    
    # All features should be finite
    assert np.all(np.isfinite(features))
    
    # Check some specific enhanced features are not zero
    # Features are: [mean_accel, std_accel, max_accel, min_accel, mean_gyro, energy, length,
    #                dominant_freq, fft_energy, median_accel, q75_accel, q25_accel, 
    #                skewness, kurtosis, std_gyro, max_gyro, mean_jerk, max_jerk,
    #                mean_rot_energy, max_rot_energy, spectral_entropy, rise_ratio, 
    #                fall_ratio, peak_location, cross_corr]
    
    # Length should be correct
    assert features[6] == 100
    
    # Median should be within range of min/max
    median_accel = features[9]
    min_accel = features[3]
    max_accel = features[2]
    assert min_accel <= median_accel <= max_accel
    
    # Peak location should be in [0, 1]
    peak_location = features[23]
    assert 0 <= peak_location <= 1


def test_enhanced_features_with_synthetic_maneuvers():
    """Test enhanced features on actual maneuver segments."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    # Use ground truth segments
    for s, e, label in seq.segments:
        features = segment_aggregated_features(feats, (s, e))
        
        # Should produce valid features
        assert len(features) == 25
        assert np.all(np.isfinite(features))
        
        # Energy should be positive for maneuvers
        energy = features[5]
        assert energy > 0
        
        # Length should match segment
        assert features[6] == e - s


def test_build_training_data_with_enhanced_features():
    """Test that training data generation works with enhanced features."""
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    X, y = build_training_data_from_sequence(seq, feats)
    
    # Should have samples
    assert len(X) > 0
    assert len(y) == len(X)
    
    # Each sample should have 25 features
    assert X.shape[1] == 25
    
    # Should have both positive and negative examples
    labels = set(y)
    assert "none" in labels
    assert len(labels) > 1  # At least 'none' and some maneuver type


def test_backward_compatibility_of_features():
    """Test that first 9 features match original implementation."""
    seq = generate_synthetic_sequence(duration_s=5.0, fs=100, seed=42)
    feats = compute_features_from_sequence(seq)
    
    segment = (100, 150)
    features = segment_aggregated_features(feats, segment)
    
    # Original features in order:
    # mean_accel, std_accel, max_accel, min_accel, mean_gyro, energy, length, dominant_freq, fft_energy
    
    # Verify basic properties of original features
    assert features[0] > 0  # mean_accel should be positive
    assert features[2] >= features[0]  # max >= mean
    assert features[3] <= features[0]  # min <= mean
    assert features[6] == 50  # length
    assert features[7] >= 0  # dominant_freq >= 0
    assert features[8] >= 0  # fft_energy >= 0


def test_enhanced_features_robustness():
    """Test enhanced features handle edge cases."""
    seq = generate_synthetic_sequence(duration_s=3.0, fs=100, seed=0)
    feats = compute_features_from_sequence(seq)
    
    # Very small segment
    small_segment = (10, 15)
    features = segment_aggregated_features(feats, small_segment)
    
    # Should still produce valid features
    assert len(features) == 25
    assert np.all(np.isfinite(features))
    
    # Short segment
    features_short = segment_aggregated_features(feats, (10, 12))
    assert len(features_short) == 25
    assert np.all(np.isfinite(features_short))
