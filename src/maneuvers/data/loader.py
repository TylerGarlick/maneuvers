"""Data loader and synthetic data generator for maneuvers demo."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Sequence:
    timestamps: np.ndarray  # shape (N,)
    accel: np.ndarray  # shape (N, 3)
    gyro: np.ndarray  # shape (N, 3)
    segments: List[Tuple[int, int, str]] = None  # list of (start_idx, end_idx, label)


def generate_synthetic_sequence(
    duration_s: float = 10.0, fs: int = 100, seed: int | None = 0
) -> Sequence:
    """Generate a synthetic flight sequence with a few maneuvers.

    The sequence contains background noise and a few maneuvers represented as
    increases in acceleration magnitude.
    """
    rng = np.random.default_rng(seed)
    N = int(duration_s * fs)
    t = np.linspace(0.0, duration_s, N, endpoint=False)

    # Background sensor noise
    accel = 0.1 * rng.standard_normal((N, 3))
    gyro = 0.01 * rng.standard_normal((N, 3))

    # Insert a few synthetic maneuvers (bursts of acceleration)
    segments = []
    # Two maneuvers at predefined intervals
    maneuvers = [
        (int(1.0 * fs), int(2.2 * fs), "left_roll"),
        (int(4.0 * fs), int(5.0 * fs), "right_roll"),
        (int(7.0 * fs), int(7.7 * fs), "climb"),
    ]
    for s, e, label in maneuvers:
        # skip maneuvers that start after the sequence
        if s >= N:
            continue
        # clip maneuvers that extend past the end
        e = min(e, N)
        length = e - s
        if length <= 0:
            continue
        # make an accel bump in x and z axes
        bump = np.hanning(length)
        accel[s:e, 0] += 1.0 * bump
        accel[s:e, 2] += 0.8 * bump
        gyro[s:e, 1] += 0.3 * bump
        segments.append((s, e, label))

    return Sequence(t, accel, gyro, segments)


# --- Dataset generation helpers ------------------------------------------------
def _write_sequence_csv(path: str | "Path", seq: Sequence) -> None:
    import csv
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
        for i, tt in enumerate(seq.timestamps):
            ax, ay, az = seq.accel[i]
            gx, gy, gz = seq.gyro[i]
            writer.writerow(
                [f"{tt:.6f}", f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}", f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}"]
            )


def generate_maneuvers_dataset(outdir: str | "Path" = "examples/datasets/maneuvers_small", maneuvers: list | None = None, duration_s: float = 10.0, fs: int = 100, seed: int | None = 0) -> list:
    """Generate a small dataset with one CSV per maneuver and return a manifest.

    Returns manifest: list of dicts {"file": str(path), "segments": [(s,e,label)]}
    """
    from pathlib import Path
    import json
    from maneuvers.data.maneuvers_catalog import MANEUVERS, synthesize_maneuver

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if maneuvers is None:
        maneuvers = MANEUVERS

    rng = np.random.default_rng(seed)
    manifest = []

    for i, name in enumerate(maneuvers):
        # create base sequence and place the maneuver in the center
        N = int(duration_s * fs)
        t = np.linspace(0.0, duration_s, N, endpoint=False)
        accel = 0.05 * rng.standard_normal((N, 3))
        gyro = 0.01 * rng.standard_normal((N, 3))

        man_len_s = 1.0 + 1.0 * rng.random()
        man_len = int(man_len_s * fs)
        start = max(0, N // 2 - man_len // 2)
        end = min(N, start + man_len)

        a, g = synthesize_maneuver(name, man_len_s, fs, rng=rng)
        accel[start:end] += a[: end - start]
        gyro[start:end] += g[: end - start]

        seq = Sequence(timestamps=t, accel=accel, gyro=gyro, segments=[(start, end, name)])
        fname = outdir / f"{i:02d}_{name}.csv"
        _write_sequence_csv(fname, seq)
        manifest.append({"file": str(fname), "segments": seq.segments})

    with open(outdir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest


def from_csv(path: str) -> Sequence:
    """Minimal CSV loader expected to have columns: t, ax,ay,az, gx,gy,gz"""
    import csv

    import numpy as np

    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("CSV contains no rows")

    t = np.array([float(r["t"]) for r in rows], dtype=float)
    accel = np.vstack([[float(r["ax"]), float(r["ay"]), float(r["az"])] for r in rows])
    gyro = np.vstack([[float(r["gx"]), float(r["gy"]), float(r["gz"])] for r in rows])

    return Sequence(timestamps=t, accel=accel, gyro=gyro, segments=None)


# --- Flight-simulator specific loaders --------------------------------------

def _load_csv_rows(path: str):
    """Load CSV into a list of dict rows (case-insensitive keys)."""
    import csv

    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # normalize keys to lower-case
            rows.append({k.lower(): v for k, v in r.items()})
    return rows


def _get_columns(rows, candidates):
    """Return a list of values for the first matching candidate column name."""
    if not rows:
        raise ValueError("CSV contains no rows")
    keys = set(rows[0].keys())
    for cand in candidates:
        if cand.lower() in keys:
            return np.array([float(r[cand.lower()]) for r in rows], dtype=float)
    # try exact lower-case candidates
    for cand in candidates:
        cl = cand.lower()
        if cl in keys:
            return np.array([float(r[cl]) for r in rows], dtype=float)
    raise KeyError(f"None of the candidate columns were found: {candidates}")


def _assemble_sequence_from_rows(rows, time_cols, accel_cols, gyro_cols):
    """Build a Sequence from CSV rows given candidate column names for time, accel, gyro."""
    t = _get_columns(rows, time_cols)
    ax = _get_columns(rows, accel_cols)
    ay = _get_columns(rows, [c.replace("x", "y") for c in accel_cols]) if any(c.lower().endswith("x") for c in accel_cols) else None
    az = _get_columns(rows, [c.replace("x", "z") for c in accel_cols]) if any(c.lower().endswith("z") or c.lower().endswith("z") for c in accel_cols) else None

    # More robust: collect per-axis candidates
    def _axis(arr_candidates):
        for names in arr_candidates:
            try:
                return _get_columns(rows, names)
            except KeyError:
                continue
        raise KeyError("No axis column found among candidates: %s" % arr_candidates)

    # accel candidates as lists of per-axis name lists
    accel_x_candidates = [["ax", "accx", "accelerationx", "acceleration_x", "accel_x", "udot"]]
    accel_y_candidates = [["ay", "accy", "accelerationy", "acceleration_y", "accel_y", "vdot"]]
    accel_z_candidates = [["az", "accz", "accelerationz", "acceleration_z", "accel_z", "wdot"]]

    # gyro candidates (angular rates)
    gyro_x_candidates = [["p", "roll_rate", "roll_rate_deg_s", "angularvelocityx", "angular_velocity_x"]]
    gyro_y_candidates = [["q", "pitch_rate", "pitch_rate_deg_s", "angularvelocityy", "angular_velocity_y"]]
    gyro_z_candidates = [["r", "yaw_rate", "yaw_rate_deg_s", "angularvelocityz", "angular_velocity_z"]]

    # Helper to try candidate groups for a single axis
    def _axis_single(group_candidates):
        for names in group_candidates:
            try:
                return _get_columns(rows, names)
            except KeyError:
                continue
        raise KeyError("No axis column found among candidates: %s" % group_candidates)

    # Get accel axes
    ax = _axis_single(accel_x_candidates)
    ay = _axis_single(accel_y_candidates)
    az = _axis_single(accel_z_candidates)

    # Get gyro axes
    gx = _axis_single(gyro_x_candidates)
    gy = _axis_single(gyro_y_candidates)
    gz = _axis_single(gyro_z_candidates)

    accel = np.vstack([ax, ay, az]).T
    gyro = np.vstack([gx, gy, gz]).T

    return Sequence(timestamps=t, accel=accel, gyro=gyro, segments=None)


def from_xplane_csv(path: str) -> Sequence:
    """Load telemetry exported by X-Plane-like CSV files.

    Expected columns (case-insensitive): time, ax, ay, az, p, q, r or variants.
    """
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(rows, time_cols=["time", "t"], accel_cols=["ax", "ay", "az"], gyro_cols=["p", "q", "r"])


def from_flightgear_csv(path: str) -> Sequence:
    """Load telemetry exported by FlightGear-like CSV files.

    FlightGear often exports body-axis accelerations as udot/vdot/wdot and angular rates as p/q/r.
    """
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(rows, time_cols=["time", "t"], accel_cols=["udot", "vdot", "wdot"], gyro_cols=["p", "q", "r"])


def from_simconnect_csv(path: str) -> Sequence:
    """Load telemetry exported via SimConnect/MSFS-like CSV files.

    Typical column names include AccelerationX/Y/Z and AngularVelocityX/Y/Z.
    """
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(rows, time_cols=["time", "t", "timestamp"], accel_cols=["accelerationx", "accelerationy", "accelerationz", "accx", "accy", "accz"], gyro_cols=["angularvelocityx", "angularvelocityy", "angularvelocityz", "p", "q", "r"])


def from_jsbsim_csv(path: str) -> Sequence:
    """Load telemetry exported by JSBSim-like CSV files."""
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(rows, time_cols=["time", "t"], accel_cols=["ax", "ay", "az", "udot", "vdot", "wdot"], gyro_cols=["p", "q", "r"]) 
