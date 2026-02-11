"""Tests for advanced detection methods and post-processing."""

import numpy as np
import pytest
from maneuvers.data.loader import Sequence, generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.detection import (
    merge_close_segments,
    split_long_segments,
    filter_short_segments,
    apply_post_processing,
    cusum_segment,
    energy_segment,
    orientation_segment,
    fusion_segment,
    learned_segment,
    train_learned_detector,
    detect_segments,
)
from maneuvers.eval import evaluate_detection

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


class TestMergeCloseSegments:
    def test_empty(self):
        assert merge_close_segments([], 10) == []

    def test_no_merge_needed(self):
        segs = [(0, 10), (30, 40)]
        assert merge_close_segments(segs, 5) == segs

    def test_merge_close_pair(self):
        segs = [(0, 10), (12, 20)]
        merged = merge_close_segments(segs, 5)
        assert merged == [(0, 20)]

    def test_merge_chain(self):
        segs = [(0, 10), (12, 20), (22, 30)]
        merged = merge_close_segments(segs, 5)
        assert merged == [(0, 30)]

    def test_partial_merge(self):
        segs = [(0, 10), (12, 20), (50, 60)]
        merged = merge_close_segments(segs, 5)
        assert merged == [(0, 20), (50, 60)]


class TestSplitLongSegments:
    def test_no_split_needed(self):
        segs = [(0, 10)]
        assert split_long_segments(segs, 20) == segs

    def test_split_in_two(self):
        result = split_long_segments([(0, 20)], 10)
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[-1][1] == 20

    def test_zero_max_len_returns_copy(self):
        segs = [(0, 100)]
        assert split_long_segments(segs, 0) == segs


class TestFilterShortSegments:
    def test_removes_short(self):
        segs = [(0, 3), (10, 30), (40, 42)]
        assert filter_short_segments(segs, 5) == [(10, 30)]

    def test_keeps_exact_min(self):
        segs = [(0, 5)]
        assert filter_short_segments(segs, 5) == [(0, 5)]


class TestApplyPostProcessing:
    def test_full_pipeline(self):
        segs = [(0, 10), (12, 20), (50, 55), (100, 200)]
        result = apply_post_processing(segs, min_gap=5, max_len=50, min_len=10)
        # (0,10)+(12,20) merged -> (0,20), (50,55) too short after min_len,
        # (100,200) split into chunks <= 50
        assert all((e - s) >= 10 for s, e in result)
        assert all((e - s) <= 50 for s, e in result)


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


class TestCusumSegment:
    def test_detects_spike(self):
        signal = np.zeros(200)
        signal[50:100] = 3.0  # strong spike
        segs = cusum_segment(signal, threshold=0.5, drift=0.1, min_len=5)
        assert len(segs) >= 1
        # The detected segment should overlap with the spike
        found = any(s < 100 and e > 50 for s, e in segs)
        assert found

    def test_no_detection_flat(self):
        signal = np.ones(100)
        segs = cusum_segment(signal, threshold=5.0, drift=0.5, min_len=5)
        assert len(segs) == 0

    def test_min_len_respected(self):
        signal = np.zeros(100)
        signal[50:53] = 5.0  # very short spike
        segs = cusum_segment(signal, threshold=0.1, drift=0.01, min_len=10)
        assert all((e - s) >= 10 for s, e in segs)


# ---------------------------------------------------------------------------
# Energy-based detection
# ---------------------------------------------------------------------------


def _make_sequence_with_vel():
    """Create a Sequence with velocity/position data and a maneuver burst."""
    N = 500
    fs = 100
    t = np.linspace(0, N / fs, N, endpoint=False)
    accel = 0.05 * np.random.RandomState(0).standard_normal((N, 3))
    gyro = 0.01 * np.random.RandomState(1).standard_normal((N, 3))
    vel = np.zeros((N, 3))
    vel[:, 0] = 50.0  # constant forward speed
    pos = np.zeros((N, 3))
    pos[:, 0] = np.cumsum(vel[:, 0]) / fs

    # Inject a maneuver: climbing turn from sample 150-250
    vel[150:250, 2] = 20.0  # climb
    vel[150:250, 0] += np.linspace(0, 30, 100)  # accelerate
    pos[150:250, 2] = np.cumsum(vel[150:250, 2]) / fs
    accel[150:250, 2] += 2.0

    orient = np.zeros((N, 3))
    orient[:, 2] = np.linspace(0, 90, N)  # slow heading change
    orient[150:250, 0] = 30.0  # bank angle during maneuver
    orient[150:250, 1] = 10.0  # pitch up

    return Sequence(
        timestamps=t,
        accel=accel,
        gyro=gyro,
        segments=[(150, 250, "climbing_turn")],
        pos=pos,
        vel=vel,
        orient=orient,
    )


class TestEnergySegment:
    def test_detects_with_velocity(self):
        seq = _make_sequence_with_vel()
        segs = energy_segment(seq, min_len=5, fs=100)
        assert len(segs) >= 1

    def test_falls_back_to_accel(self):
        seq = generate_synthetic_sequence(duration_s=5.0, fs=100, seed=0)
        # seq has no vel — energy_segment should still work using accel
        segs = energy_segment(seq, min_len=5, fs=100)
        assert isinstance(segs, list)

    def test_custom_threshold(self):
        seq = _make_sequence_with_vel()
        segs_low = energy_segment(seq, threshold=0.01, min_len=5, fs=100)
        segs_high = energy_segment(seq, threshold=1e6, min_len=5, fs=100)
        assert len(segs_low) >= len(segs_high)


# ---------------------------------------------------------------------------
# Orientation-based detection
# ---------------------------------------------------------------------------


class TestOrientationSegment:
    def test_detects_with_orient(self):
        seq = _make_sequence_with_vel()
        segs = orientation_segment(seq, min_len=5, fs=100)
        assert len(segs) >= 1

    def test_falls_back_to_velocity(self):
        seq = _make_sequence_with_vel()
        # Remove orient, keep vel
        seq_no_orient = Sequence(
            timestamps=seq.timestamps,
            accel=seq.accel,
            gyro=seq.gyro,
            segments=seq.segments,
            pos=seq.pos,
            vel=seq.vel,
            orient=None,
        )
        segs = orientation_segment(seq_no_orient, min_len=5, fs=100)
        assert isinstance(segs, list)

    def test_falls_back_to_gyro(self):
        seq = generate_synthetic_sequence(duration_s=5.0, fs=100, seed=0)
        # No orient, no vel -> uses gyro
        segs = orientation_segment(seq, min_len=5, fs=100)
        assert isinstance(segs, list)


# ---------------------------------------------------------------------------
# Fusion detection
# ---------------------------------------------------------------------------


class TestFusionSegment:
    def test_fuses_multiple_signals(self):
        seq = _make_sequence_with_vel()
        feats = compute_features_from_sequence(seq)
        segs = fusion_segment(seq, features=feats, vote_threshold=2, min_len=5, fs=100)
        assert isinstance(segs, list)

    def test_high_vote_threshold_reduces_detections(self):
        seq = _make_sequence_with_vel()
        feats = compute_features_from_sequence(seq)
        segs_low = fusion_segment(
            seq, features=feats, vote_threshold=1, min_len=5, fs=100
        )
        segs_high = fusion_segment(
            seq, features=feats, vote_threshold=4, min_len=5, fs=100
        )
        assert len(segs_low) >= len(segs_high)


# ---------------------------------------------------------------------------
# Learned detection
# ---------------------------------------------------------------------------


class TestLearnedSegment:
    def test_train_and_detect(self):
        seqs = [
            generate_synthetic_sequence(duration_s=5.0, fs=50, seed=i) for i in range(3)
        ]
        clf = train_learned_detector(seqs, fs=50, window_s=0.5, hop_s=0.25)

        # Use trained model to detect on new data
        test_seq = generate_synthetic_sequence(duration_s=5.0, fs=50, seed=99)
        feats = compute_features_from_sequence(test_seq)
        segs = learned_segment(clf, feats, test_seq, window=25, hop=12, min_len=5)
        assert isinstance(segs, list)

    def test_dummy_model(self):
        """Test with a mock model that always predicts 'maneuver'."""

        class AlwaysManeuver:
            def predict(self, X):
                return np.ones(len(X), dtype=int)

        seq = generate_synthetic_sequence(duration_s=3.0, fs=50, seed=0)
        feats = compute_features_from_sequence(seq)
        segs = learned_segment(
            AlwaysManeuver(), feats, seq, window=25, hop=12, min_len=5
        )
        # Should detect at least one segment since model says everything is maneuver
        assert len(segs) >= 1


# ---------------------------------------------------------------------------
# Dispatcher (detect_segments) with new methods
# ---------------------------------------------------------------------------


class TestDetectSegmentsDispatcher:
    def test_cusum_via_dispatcher(self):
        seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
        feats = compute_features_from_sequence(seq)
        segs = detect_segments(
            feats, method="cusum", seq=seq, threshold=0.5, drift=0.1, min_len=5
        )
        assert isinstance(segs, list)

    def test_energy_via_dispatcher(self):
        seq = _make_sequence_with_vel()
        feats = compute_features_from_sequence(seq)
        segs = detect_segments(feats, method="energy", seq=seq, min_len=5)
        assert isinstance(segs, list)

    def test_orientation_via_dispatcher(self):
        seq = _make_sequence_with_vel()
        feats = compute_features_from_sequence(seq)
        segs = detect_segments(feats, method="orientation", seq=seq, min_len=5)
        assert isinstance(segs, list)

    def test_fusion_via_dispatcher(self):
        seq = _make_sequence_with_vel()
        feats = compute_features_from_sequence(seq)
        segs = detect_segments(
            feats, method="fusion", seq=seq, vote_threshold=2, min_len=5
        )
        assert isinstance(segs, list)

    def test_energy_requires_seq(self):
        seq = generate_synthetic_sequence(duration_s=5.0, fs=100, seed=0)
        feats = compute_features_from_sequence(seq)
        with pytest.raises(ValueError, match="requires a Sequence"):
            detect_segments(feats, method="energy")

    def test_post_processing_via_dispatcher(self):
        seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
        feats = compute_features_from_sequence(seq)
        segs = detect_segments(
            feats,
            method="threshold",
            threshold=0.2,
            min_len=3,
            post_min_gap=20,
            post_min_len=10,
        )
        # All segments should be >= 10 samples after post-processing
        assert all((e - s) >= 10 for s, e in segs)

    def test_unknown_method_still_raises(self):
        seq = generate_synthetic_sequence(duration_s=5.0, fs=50, seed=1)
        feats = compute_features_from_sequence(seq)
        with pytest.raises(ValueError, match="Unknown method"):
            detect_segments(feats, method="nonexistent")


# ---------------------------------------------------------------------------
# Integration: new methods on synthetic data with eval
# ---------------------------------------------------------------------------


class TestDetectionIntegration:
    def test_cusum_finds_maneuvers(self):
        seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=2)
        feats = compute_features_from_sequence(seq)
        preds = detect_segments(
            feats, method="cusum", seq=seq, threshold=0.3, drift=0.05, min_len=5
        )
        gt = [(s, e) for s, e, _ in seq.segments]
        res = evaluate_detection(gt, preds, iou_thresh=0.2)
        assert res["tp"] >= 1

    def test_energy_on_seq_with_vel(self):
        seq = _make_sequence_with_vel()
        feats = compute_features_from_sequence(seq)
        preds = detect_segments(feats, method="energy", seq=seq, min_len=5)
        # Energy detector should find something in the maneuver region
        assert len(preds) >= 1
        # At least one prediction should overlap the GT maneuver (150-250)
        found_overlap = any(s < 250 and e > 150 for s, e in preds)
        assert found_overlap

    def test_fusion_on_seq_with_vel(self):
        seq = _make_sequence_with_vel()
        feats = compute_features_from_sequence(seq)
        preds = detect_segments(
            feats, method="fusion", seq=seq, vote_threshold=2, min_len=5
        )
        gt = [(s, e) for s, e, _ in seq.segments]
        res = evaluate_detection(gt, preds, iou_thresh=0.1)
        assert res["tp"] >= 1
