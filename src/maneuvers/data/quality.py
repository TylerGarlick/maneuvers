"""Quality checks for flight sequences."""

from __future__ import annotations
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
