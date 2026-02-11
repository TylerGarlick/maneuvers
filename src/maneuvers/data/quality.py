"""Quality checks for flight sequences.

Provides checks for the Maneuver-ID challenge data quality sorting task:
- Minimum requirements (sampling rate, duration, column presence)
- Trajectory continuity (position jumps, velocity discontinuities, timestamp gaps)
- Comprehensive quality assessment combining all checks
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from maneuvers.data.loader import Sequence


def check_minimum_requirements(seq: Sequence) -> dict:
    """Check if sequence meets minimum quality requirements.

    Returns dict with 'valid': bool and 'issues': list of str.
    """
    issues = []
    valid = True

    # Check sampling rate
    if len(seq.timestamps) < 2:
        issues.append("Sequence too short (<2 samples)")
        valid = False
    else:
        dt = np.diff(seq.timestamps)
        fs = 1.0 / np.mean(dt)
        if fs < 50:
            issues.append(f"Sampling rate too low: {fs:.1f} Hz < 50 Hz")
            valid = False

    # Check sequence length
    duration = seq.timestamps[-1] - seq.timestamps[0]
    if duration < 10.0:
        issues.append(f"Sequence too short: {duration:.1f} s < 10 s")
        valid = False

    # Check column presence
    if seq.accel is None or seq.accel.shape[1] != 3:
        issues.append("Missing or invalid accel data")
        valid = False
    if seq.gyro is None or seq.gyro.shape[1] != 3:
        issues.append("Missing or invalid gyro data")
        valid = False

    # Physical plausibility
    if seq.accel is not None:
        accel_mag = np.linalg.norm(seq.accel, axis=1)
        if np.max(accel_mag) > 10 * 9.81:  # 10g
            issues.append("Acceleration exceeds 10g")
            valid = False

    if seq.vel is not None:
        vel_mag = np.linalg.norm(seq.vel, axis=1)
        if np.max(vel_mag) > 300:  # Mach 1 approx
            issues.append("Velocity exceeds 300 m/s")
            valid = False

    return {"valid": valid, "issues": issues}


# ---------------------------------------------------------------------------
# Trajectory continuity checks
# ---------------------------------------------------------------------------


def check_timestamp_gaps(
    seq: Sequence,
    gap_factor: float = 3.0,
) -> dict:
    """Detect gaps in timestamps where dt exceeds gap_factor * median(dt).

    Args:
        seq: Flight sequence to check.
        gap_factor: A timestep is flagged as a gap when it exceeds
            gap_factor * median(dt).  Default 3.0.

    Returns:
        dict with:
            'gaps': list of (index, dt_value) for each flagged gap
            'valid': True if no gaps detected
    """
    if len(seq.timestamps) < 2:
        return {"gaps": [], "valid": True}

    dt = np.diff(seq.timestamps)
    median_dt = np.median(dt)
    if median_dt <= 0:
        return {
            "gaps": [(0, float(dt[0]))],
            "valid": False,
        }

    threshold = gap_factor * median_dt
    gap_indices = np.where(dt > threshold)[0]

    gaps = [(int(i), float(dt[i])) for i in gap_indices]
    return {"gaps": gaps, "valid": len(gaps) == 0}


def check_position_jumps(
    seq: Sequence,
    max_speed_m_s: float = 350.0,
) -> dict:
    """Detect teleportation jumps in position data.

    A jump is flagged when the implied speed between consecutive samples
    exceeds max_speed_m_s (default ~Mach 1).

    Args:
        seq: Flight sequence with optional pos field.
        max_speed_m_s: Maximum plausible speed in m/s.

    Returns:
        dict with:
            'jumps': list of (index, implied_speed) for each flagged jump
            'valid': True if no jumps and pos data present
            'skipped': True if pos data is not available
    """
    if seq.pos is None or len(seq.timestamps) < 2:
        return {"jumps": [], "valid": True, "skipped": True}

    dt = np.diff(seq.timestamps)
    dt = np.where(dt > 0, dt, 1e-6)
    dpos = np.diff(seq.pos, axis=0)
    dist = np.linalg.norm(dpos, axis=1)
    speed = dist / dt

    jump_indices = np.where(speed > max_speed_m_s)[0]
    jumps = [(int(i), float(speed[i])) for i in jump_indices]
    return {"jumps": jumps, "valid": len(jumps) == 0, "skipped": False}


def check_velocity_discontinuities(
    seq: Sequence,
    max_accel_m_s2: float = 150.0,
) -> dict:
    """Detect discontinuities in velocity data.

    A discontinuity is flagged when the implied acceleration between
    consecutive samples exceeds max_accel_m_s2 (default ~15g, generous
    for any maneuvering aircraft).

    Args:
        seq: Flight sequence with optional vel field.
        max_accel_m_s2: Maximum plausible acceleration in m/s^2.

    Returns:
        dict with:
            'discontinuities': list of (index, implied_accel)
            'valid': True if none found and vel data present
            'skipped': True if vel data is not available
    """
    if seq.vel is None or len(seq.timestamps) < 2:
        return {"discontinuities": [], "valid": True, "skipped": True}

    dt = np.diff(seq.timestamps)
    dt = np.where(dt > 0, dt, 1e-6)
    dvel = np.diff(seq.vel, axis=0)
    dvel_mag = np.linalg.norm(dvel, axis=1)
    implied_accel = dvel_mag / dt

    disc_indices = np.where(implied_accel > max_accel_m_s2)[0]
    discs = [(int(i), float(implied_accel[i])) for i in disc_indices]
    return {
        "discontinuities": discs,
        "valid": len(discs) == 0,
        "skipped": False,
    }


def check_orientation_discontinuities(
    seq: Sequence,
    max_rate_deg_s: float = 500.0,
) -> dict:
    """Detect discontinuities in orientation data.

    A discontinuity is flagged when the implied angular rate between
    consecutive samples exceeds max_rate_deg_s.

    Args:
        seq: Flight sequence with optional orient field (degrees).
        max_rate_deg_s: Maximum plausible angular rate in deg/s.

    Returns:
        dict with:
            'discontinuities': list of (index, implied_rate_deg_s)
            'valid': True if none found and orient data present
            'skipped': True if orient data is not available
    """
    if seq.orient is None or len(seq.timestamps) < 2:
        return {"discontinuities": [], "valid": True, "skipped": True}

    dt = np.diff(seq.timestamps)
    dt = np.where(dt > 0, dt, 1e-6)
    dorient = np.diff(seq.orient, axis=0)

    # Handle angle wrapping: differences should be in [-180, 180]
    dorient = (dorient + 180.0) % 360.0 - 180.0

    rate = np.abs(dorient) / dt[:, np.newaxis]
    max_rate_per_step = np.max(rate, axis=1)

    disc_indices = np.where(max_rate_per_step > max_rate_deg_s)[0]
    discs = [(int(i), float(max_rate_per_step[i])) for i in disc_indices]
    return {
        "discontinuities": discs,
        "valid": len(discs) == 0,
        "skipped": False,
    }


def check_trajectory_continuity(seq: Sequence, **kwargs) -> dict:
    """Run all continuity checks and return a combined result.

    Keyword arguments are forwarded to the individual check functions
    (gap_factor, max_speed_m_s, max_accel_m_s2, max_rate_deg_s).

    Returns:
        dict with:
            'valid': True only if all sub-checks pass
            'issues': list of human-readable issue strings
            'details': dict of individual check results keyed by name
    """
    ts_result = check_timestamp_gaps(
        seq, gap_factor=kwargs.get("gap_factor", 3.0)
    )
    pos_result = check_position_jumps(
        seq, max_speed_m_s=kwargs.get("max_speed_m_s", 350.0)
    )
    vel_result = check_velocity_discontinuities(
        seq, max_accel_m_s2=kwargs.get("max_accel_m_s2", 150.0)
    )
    orient_result = check_orientation_discontinuities(
        seq, max_rate_deg_s=kwargs.get("max_rate_deg_s", 500.0)
    )

    issues: List[str] = []
    if not ts_result["valid"]:
        issues.append(
            f"Timestamp gaps detected: {len(ts_result['gaps'])} gap(s)"
        )
    if not pos_result["valid"]:
        issues.append(
            f"Position jumps detected: {len(pos_result['jumps'])} jump(s)"
        )
    if not vel_result["valid"]:
        issues.append(
            f"Velocity discontinuities: "
            f"{len(vel_result['discontinuities'])} discontinuity(ies)"
        )
    if not orient_result["valid"]:
        issues.append(
            f"Orientation discontinuities: "
            f"{len(orient_result['discontinuities'])} discontinuity(ies)"
        )

    valid = all(
        r["valid"]
        for r in [ts_result, pos_result, vel_result, orient_result]
    )

    return {
        "valid": valid,
        "issues": issues,
        "details": {
            "timestamp_gaps": ts_result,
            "position_jumps": pos_result,
            "velocity_discontinuities": vel_result,
            "orientation_discontinuities": orient_result,
        },
    }
