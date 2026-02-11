"""Tests for data quality checks including trajectory continuity."""

import numpy as np
import pytest
from maneuvers.data.loader import Sequence
from maneuvers.data.quality import (
    check_minimum_requirements,
    check_timestamp_gaps,
    check_position_jumps,
    check_velocity_discontinuities,
    check_orientation_discontinuities,
    check_trajectory_continuity,
)


def _make_sequence(
    N=500,
    fs=100,
    pos=None,
    vel=None,
    orient=None,
    accel=None,
    gyro=None,
    timestamps=None,
):
    """Helper to build a Sequence with sensible defaults."""
    if timestamps is None:
        timestamps = np.linspace(0, N / fs, N, endpoint=False)
    if accel is None:
        accel = 0.1 * np.random.default_rng(0).standard_normal((N, 3))
    if gyro is None:
        gyro = 0.01 * np.random.default_rng(1).standard_normal((N, 3))
    return Sequence(
        timestamps=timestamps,
        accel=accel,
        gyro=gyro,
        pos=pos,
        vel=vel,
        orient=orient,
    )


# ---------------------------------------------------------------------------
# check_timestamp_gaps
# ---------------------------------------------------------------------------


class TestTimestampGaps:
    def test_uniform_timestamps_no_gaps(self):
        seq = _make_sequence(N=200, fs=100)
        result = check_timestamp_gaps(seq)
        assert result["valid"] is True
        assert result["gaps"] == []

    def test_single_gap(self):
        N = 200
        t = np.linspace(0, 2, N, endpoint=False)
        # Insert a 0.5 s gap at index 100 (normal dt ~0.01)
        t[100:] += 0.5
        seq = _make_sequence(N=N, timestamps=t)
        result = check_timestamp_gaps(seq)
        assert result["valid"] is False
        assert len(result["gaps"]) >= 1
        # The gap should be at index 99 (diff between t[99] and t[100])
        gap_indices = [g[0] for g in result["gaps"]]
        assert 99 in gap_indices

    def test_multiple_gaps(self):
        N = 300
        t = np.linspace(0, 3, N, endpoint=False)
        t[100:] += 0.5
        t[200:] += 0.5
        seq = _make_sequence(N=N, timestamps=t)
        result = check_timestamp_gaps(seq)
        assert result["valid"] is False
        assert len(result["gaps"]) >= 2

    def test_short_sequence(self):
        seq = _make_sequence(N=1, fs=100)
        result = check_timestamp_gaps(seq)
        assert result["valid"] is True

    def test_custom_gap_factor(self):
        N = 200
        t = np.linspace(0, 2, N, endpoint=False)
        # Insert a mild gap (2x median dt)
        t[100:] += 0.015
        seq = _make_sequence(N=N, timestamps=t)
        # With default factor 3.0, this shouldn't trigger
        assert check_timestamp_gaps(seq, gap_factor=3.0)["valid"] is True
        # With factor 1.5, it should trigger
        assert check_timestamp_gaps(seq, gap_factor=1.5)["valid"] is False


# ---------------------------------------------------------------------------
# check_position_jumps
# ---------------------------------------------------------------------------


class TestPositionJumps:
    def test_no_pos_data_skipped(self):
        seq = _make_sequence(N=200)
        result = check_position_jumps(seq)
        assert result["valid"] is True
        assert result["skipped"] is True

    def test_smooth_trajectory_no_jumps(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        # Smooth trajectory: aircraft moving at 100 m/s in x
        pos = np.zeros((N, 3))
        pos[:, 0] = 100.0 * t
        seq = _make_sequence(N=N, pos=pos)
        result = check_position_jumps(seq)
        assert result["valid"] is True
        assert result["skipped"] is False

    def test_teleportation_detected(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        pos = np.zeros((N, 3))
        pos[:, 0] = 100.0 * t
        # Insert a 10 km teleportation at index 250
        pos[250:, 0] += 10000.0
        seq = _make_sequence(N=N, pos=pos)
        result = check_position_jumps(seq)
        assert result["valid"] is False
        assert len(result["jumps"]) >= 1
        jump_indices = [j[0] for j in result["jumps"]]
        assert 249 in jump_indices

    def test_custom_speed_threshold(self):
        N = 200
        t = np.linspace(0, 2, N, endpoint=False)
        pos = np.zeros((N, 3))
        pos[:, 0] = 200.0 * t  # 200 m/s constant speed
        seq = _make_sequence(N=N, pos=pos)
        # Default threshold 350 m/s — should pass
        assert check_position_jumps(seq, max_speed_m_s=350.0)["valid"] is True
        # Lower threshold — should fail
        assert (
            check_position_jumps(seq, max_speed_m_s=100.0)["valid"] is False
        )


# ---------------------------------------------------------------------------
# check_velocity_discontinuities
# ---------------------------------------------------------------------------


class TestVelocityDiscontinuities:
    def test_no_vel_data_skipped(self):
        seq = _make_sequence(N=200)
        result = check_velocity_discontinuities(seq)
        assert result["valid"] is True
        assert result["skipped"] is True

    def test_smooth_velocity_no_discontinuities(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        # Smooth acceleration: velocity increasing linearly at 5 m/s^2
        vel = np.zeros((N, 3))
        vel[:, 0] = 50.0 + 5.0 * t
        seq = _make_sequence(N=N, vel=vel)
        result = check_velocity_discontinuities(seq)
        assert result["valid"] is True
        assert result["skipped"] is False

    def test_velocity_jump_detected(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        vel = np.zeros((N, 3))
        vel[:, 0] = 100.0
        # Sudden 500 m/s velocity change at index 250
        vel[250:, 0] += 500.0
        seq = _make_sequence(N=N, vel=vel)
        result = check_velocity_discontinuities(seq)
        assert result["valid"] is False
        assert len(result["discontinuities"]) >= 1


# ---------------------------------------------------------------------------
# check_orientation_discontinuities
# ---------------------------------------------------------------------------


class TestOrientationDiscontinuities:
    def test_no_orient_data_skipped(self):
        seq = _make_sequence(N=200)
        result = check_orientation_discontinuities(seq)
        assert result["valid"] is True
        assert result["skipped"] is True

    def test_smooth_orientation_no_discontinuities(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        orient = np.zeros((N, 3))
        # Heading rotating at 30 deg/s
        orient[:, 2] = 30.0 * t
        seq = _make_sequence(N=N, orient=orient)
        result = check_orientation_discontinuities(seq)
        assert result["valid"] is True

    def test_heading_jump_detected(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        orient = np.zeros((N, 3))
        orient[:, 2] = 90.0  # constant heading
        # Sudden 170-degree heading jump at index 250
        orient[250:, 2] += 170.0
        seq = _make_sequence(N=N, orient=orient)
        result = check_orientation_discontinuities(seq)
        assert result["valid"] is False
        assert len(result["discontinuities"]) >= 1

    def test_angle_wrapping_not_flagged(self):
        """Normal heading crossing 360/0 boundary should not be flagged."""
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        orient = np.zeros((N, 3))
        # Heading going from 350 to 10 degrees smoothly at 20 deg/s
        orient[:, 2] = (350.0 + 20.0 * t) % 360.0
        seq = _make_sequence(N=N, orient=orient)
        result = check_orientation_discontinuities(seq)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# check_trajectory_continuity (combined)
# ---------------------------------------------------------------------------


class TestTrajectoryContinuityCombined:
    def test_clean_sequence_valid(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        pos = np.zeros((N, 3))
        pos[:, 0] = 100.0 * t
        vel = np.zeros((N, 3))
        vel[:, 0] = 100.0
        orient = np.zeros((N, 3))
        orient[:, 2] = 10.0 * t
        seq = _make_sequence(N=N, pos=pos, vel=vel, orient=orient)
        result = check_trajectory_continuity(seq)
        assert result["valid"] is True
        assert result["issues"] == []
        assert "timestamp_gaps" in result["details"]
        assert "position_jumps" in result["details"]
        assert "velocity_discontinuities" in result["details"]
        assert "orientation_discontinuities" in result["details"]

    def test_multiple_issues_reported(self):
        N = 500
        t = np.linspace(0, 5, N, endpoint=False)
        # Timestamp gap
        t[250:] += 1.0
        # Position jump
        pos = np.zeros((N, 3))
        pos[:, 0] = 100.0 * t
        pos[300:, 0] += 50000.0
        seq = _make_sequence(N=N, timestamps=t, pos=pos)
        result = check_trajectory_continuity(seq)
        assert result["valid"] is False
        assert len(result["issues"]) >= 2

    def test_accel_gyro_only_sequence(self):
        """A sequence with no pos/vel/orient should still pass continuity."""
        seq = _make_sequence(N=500)
        result = check_trajectory_continuity(seq)
        assert result["valid"] is True
