"""Data loader and synthetic data generator for maneuvers demo."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Sequence:
    timestamps: np.ndarray  # shape (N,)
    accel: np.ndarray  # shape (N, 3)
    gyro: np.ndarray  # shape (N, 3)
    segments: List[Tuple[int, int, str]] = None  # list of (start_idx, end_idx, label)
    pos: Optional[np.ndarray] = None  # shape (N, 3) - positions x,y,z
    vel: Optional[np.ndarray] = None  # shape (N, 3) - velocities vx,vy,vz
    orient: Optional[np.ndarray] = None  # shape (N, 3) - orientations roll,pitch,yaw


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
def _write_sequence_csv(path: str | Path, seq: Sequence) -> None:
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
        for i, tt in enumerate(seq.timestamps):
            ax, ay, az = seq.accel[i]
            gx, gy, gz = seq.gyro[i]
            writer.writerow(
                [
                    f"{tt:.6f}",
                    f"{ax:.6f}",
                    f"{ay:.6f}",
                    f"{az:.6f}",
                    f"{gx:.6f}",
                    f"{gy:.6f}",
                    f"{gz:.6f}",
                ]
            )


def generate_maneuvers_dataset(
    outdir: str | Path = "examples/datasets/maneuvers_small",
    maneuvers: list | None = None,
    duration_s: float = 10.0,
    fs: int = 100,
    seed: int | None = 0,
) -> list:
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

        seq = Sequence(
            timestamps=t, accel=accel, gyro=gyro, segments=[(start, end, name)]
        )
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


def _detect_units_from_column_names(rows):
    """Heuristic detection of units from column names and values.

    Returns dict with keys: gyro_units ('deg/s'|'rad/s'|None), accel_units ('g'|'m/s2'|None)
    """
    keys = set(rows[0].keys()) if rows else set()
    lower_keys = {k.lower() for k in keys}

    res = {"gyro_units": None, "accel_units": None}

    # detect gyro columns with explicit unit hints
    if (
        any("deg" in k for k in lower_keys)
        or any("deg_s" in k for k in lower_keys)
        or any("deg/s" in k for k in lower_keys)
    ):
        res["gyro_units"] = "deg/s"
    if any("rad" in k or "rad_s" in k or "rad/s" in k for k in lower_keys):
        res["gyro_units"] = "rad/s"

    # detect accel columns in g or m/s^2
    if any(
        "g" in k and ("acc" in k or "accel" in k or "udot" in k or "wdot" in k)
        for k in lower_keys
    ):
        res["accel_units"] = "g"
    if any("m/s" in k or "m_s" in k or "mps" in k for k in lower_keys):
        res["accel_units"] = "m/s2"

    # numeric heuristic: check magnitudes of gyro values if available
    try:
        # collect any gyro-like columns
        gyro_vals = []
        for k in lower_keys:
            if (
                k in ("p", "q", "r")
                or "angularvelocity" in k
                or "roll_rate" in k
                or "pitch_rate" in k
                or "yaw_rate" in k
            ):
                gyro_vals.append([float(r[k]) for r in rows])
        if gyro_vals:
            import numpy as _np

            gvals = _np.vstack(gyro_vals)
            mean_abs = _np.mean(_np.abs(gvals))
            # typical deg/s values often >> 10; rad/s values are small (<10)
            if mean_abs > 20:
                res["gyro_units"] = res.get("gyro_units") or "deg/s"
            elif mean_abs < 10 and res.get("gyro_units") is None:
                res["gyro_units"] = "rad/s"
    except Exception:
        # ignore
        pass

    # numeric heuristic for accel: if typical magnitudes ~1, might be g
    try:
        acc_vals = []
        for k in lower_keys:
            if k.startswith("a") or k in ("udot", "vdot", "wdot") or "acc" in k:
                acc_vals.append([float(r[k]) for r in rows])
        if acc_vals:
            import numpy as _np

            avals = _np.vstack(acc_vals)
            mean_abs = _np.mean(_np.abs(avals))
            if mean_abs > 5:
                # large accelerations in m/s^2
                res["accel_units"] = res.get("accel_units") or "m/s2"
            elif mean_abs < 5:
                # maybe in gs
                res["accel_units"] = res.get("accel_units") or "g"
    except Exception:
        pass

    return res


def _apply_unit_conversions(
    accel,
    gyro,
    detect_info=None,
    convert_gyro_deg_to_rad=False,
    convert_accel_g_to_m_s2=False,
):
    """Apply unit conversions according to flags or detected units."""
    import numpy as _np

    if detect_info is None:
        detect_info = {}

    # Gyro: deg/s -> rad/s
    gyro_units = detect_info.get("gyro_units")
    if convert_gyro_deg_to_rad or (
        convert_gyro_deg_to_rad is None and gyro_units == "deg/s"
    ):
        gyro = gyro * (_np.pi / 180.0)

    # Accel: g -> m/s^2
    accel_units = detect_info.get("accel_units")
    if convert_accel_g_to_m_s2 or (
        convert_accel_g_to_m_s2 is None and accel_units == "g"
    ):
        accel = accel * 9.80665

    return accel, gyro


def _assemble_sequence_from_rows(
    rows,
    time_cols,
    accel_cols,
    gyro_cols,
    *,
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
):
    """Build a Sequence from CSV rows given candidate column names for time, accel, gyro.

    Parameters
    - convert_gyro_deg_to_rad: False (default) | True to convert deg/s to rad/s | None to auto-detect and convert if needed
    - convert_accel_g_to_m_s2: same for acceleration in g -> m/s^2
    """
    t = _get_columns(rows, time_cols)

    # More robust: collect per-axis candidates
    def _axis(arr_candidates):
        for names in arr_candidates:
            try:
                return _get_columns(rows, names)
            except KeyError:
                continue
        raise KeyError("No axis column found among candidates: %s" % arr_candidates)

    # accel candidates as lists of per-axis name lists
    accel_x_candidates = [
        ["ax", "accx", "accelerationx", "acceleration_x", "accel_x", "udot"]
    ]
    accel_y_candidates = [
        ["ay", "accy", "accelerationy", "acceleration_y", "accel_y", "vdot"]
    ]
    accel_z_candidates = [
        ["az", "accz", "accelerationz", "acceleration_z", "accel_z", "wdot"]
    ]

    # gyro candidates (angular rates)
    gyro_x_candidates = [
        ["p", "roll_rate", "roll_rate_deg_s", "angularvelocityx", "angular_velocity_x"]
    ]
    gyro_y_candidates = [
        [
            "q",
            "pitch_rate",
            "pitch_rate_deg_s",
            "angularvelocityy",
            "angular_velocity_y",
        ]
    ]
    gyro_z_candidates = [
        ["r", "yaw_rate", "yaw_rate_deg_s", "angularvelocityz", "angular_velocity_z"]
    ]

    # Get accel axes
    ax = _axis(accel_x_candidates)
    ay = _axis(accel_y_candidates)
    az = _axis(accel_z_candidates)

    # Get gyro axes
    gx = _axis(gyro_x_candidates)
    gy = _axis(gyro_y_candidates)
    gz = _axis(gyro_z_candidates)

    accel = np.vstack([ax, ay, az]).T
    gyro = np.vstack([gx, gy, gz]).T

    # Unit detection heuristic
    detect_info = _detect_units_from_column_names(rows)

    # Apply conversions according to flags
    accel, gyro = _apply_unit_conversions(
        accel,
        gyro,
        detect_info,
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )

    return Sequence(timestamps=t, accel=accel, gyro=gyro, segments=None)


def from_xplane_csv(
    path: str,
    *,
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load telemetry exported by X-Plane-like CSV files.

    Expected columns (case-insensitive): time, ax, ay, az, p, q, r or variants.

    Parameters
    - convert_gyro_deg_to_rad: False | True | None (auto-detect)
    - convert_accel_g_to_m_s2: False | True | None
    """
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(
        rows,
        time_cols=["time", "t"],
        accel_cols=["ax", "ay", "az"],
        gyro_cols=["p", "q", "r"],
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )


def from_flightgear_csv(
    path: str,
    *,
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load telemetry exported by FlightGear-like CSV files.

    FlightGear often exports body-axis accelerations as udot/vdot/wdot and angular rates as p/q/r.
    """
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(
        rows,
        time_cols=["time", "t"],
        accel_cols=["udot", "vdot", "wdot"],
        gyro_cols=["p", "q", "r"],
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )


def from_simconnect_csv(
    path: str,
    *,
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load telemetry exported via SimConnect/MSFS-like CSV files.

    Typical column names include AccelerationX/Y/Z and AngularVelocityX/Y/Z.
    """
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(
        rows,
        time_cols=["time", "t", "timestamp"],
        accel_cols=[
            "accelerationx",
            "accelerationy",
            "accelerationz",
            "accx",
            "accy",
            "accz",
        ],
        gyro_cols=[
            "angularvelocityx",
            "angularvelocityy",
            "angularvelocityz",
            "p",
            "q",
            "r",
        ],
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )


def from_jsbsim_csv(
    path: str,
    *,
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load telemetry exported by JSBSim-like CSV files."""
    rows = _load_csv_rows(path)
    return _assemble_sequence_from_rows(
        rows,
        time_cols=["time", "t"],
        accel_cols=["ax", "ay", "az", "udot", "vdot", "wdot"],
        gyro_cols=["p", "q", "r"],
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )


def from_csv_filelike(
    f,
    *,
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load CSV from a file-like object (open file or stream)."""
    import csv

    rows = []
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({k.lower(): v for k, v in r.items()})
    if not rows:
        raise ValueError("CSV contains no rows")
    # attempt to detect which loader to use by inspecting columns
    lower_keys = set(rows[0].keys())
    # simple heuristic: use udot->flightgear, accelerationx->simconnect, else xplane/jsbsim
    if any(k.startswith("udot") for k in lower_keys):
        return _assemble_sequence_from_rows(
            rows,
            time_cols=["time", "t"],
            accel_cols=["udot", "vdot", "wdot"],
            gyro_cols=["p", "q", "r"],
            convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
            convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
        )
    if any("accelerationx" in k for k in lower_keys) or any(
        k.startswith("acc") and k.endswith("x") for k in lower_keys
    ):
        return _assemble_sequence_from_rows(
            rows,
            time_cols=["time", "t", "timestamp"],
            accel_cols=[
                "accelerationx",
                "accelerationy",
                "accelerationz",
                "accx",
                "accy",
                "accz",
            ],
            gyro_cols=[
                "angularvelocityx",
                "angularvelocityy",
                "angularvelocityz",
                "p",
                "q",
                "r",
            ],
            convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
            convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
        )
    # fallback to xplane-style
    return _assemble_sequence_from_rows(
        rows,
        time_cols=["time", "t"],
        accel_cols=["ax", "ay", "az"],
        gyro_cols=["p", "q", "r"],
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )


def from_json(
    path: str,
    *,
    time_key: str = "time",
    accel_key: str = "accel",
    gyro_key: str = "gyro",
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load telemetry from a simple JSON file with arrays.

    Expected formats supported:
      {"time": [...], "accel": [[ax,ay,az], ...], "gyro": [[gx,gy,gz], ...]}
      {"time": [...], "ax": [...], "ay": [...], "az": [...], "p": [...], "q": [...], "r": [...]}  (per-axis)
    """
    import json

    with open(path, "r") as fh:
        data = json.load(fh)

    # per-axis arrays
    if all(k in data for k in ("ax", "ay", "az", "p", "q", "r")):
        t = np.array(data.get(time_key) or data.get("t"), dtype=float)
        accel = np.vstack(
            [
                np.array(data["ax"], dtype=float),
                np.array(data["ay"], dtype=float),
                np.array(data["az"], dtype=float),
            ]
        ).T
        gyro = np.vstack(
            [
                np.array(data["p"], dtype=float),
                np.array(data["q"], dtype=float),
                np.array(data["r"], dtype=float),
            ]
        ).T
        detect_info = {"gyro_units": None, "accel_units": None}
        accel, gyro = _apply_unit_conversions(
            accel,
            gyro,
            detect_info,
            convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
            convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
        )
        return Sequence(timestamps=t, accel=accel, gyro=gyro, segments=None)

    # nested accel/gyro arrays
    if accel_key in data and gyro_key in data:
        t = np.array(data.get(time_key) or data.get("t"), dtype=float)
        accel = np.array(data[accel_key], dtype=float)
        gyro = np.array(data[gyro_key], dtype=float)
        detect_info = {"gyro_units": None, "accel_units": None}
        accel, gyro = _apply_unit_conversions(
            accel,
            gyro,
            detect_info,
            convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
            convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
        )
        return Sequence(timestamps=t, accel=accel, gyro=gyro, segments=None)

    raise ValueError(
        "JSON format not recognized; expected per-axis arrays or nested accel/gyro arrays"
    )


def load_t6a_sequence(
    path: str,
    *,
    convert_gyro_deg_to_rad: bool | None = False,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load T-6A Texan II maneuver data from CSV.

    Expected columns: time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
    orient_roll, orient_pitch, orient_yaw, accel_x, accel_y, accel_z,
    gyro_p, gyro_q, gyro_r

    Positions in meters, velocities in m/s, orientations in radians (or degrees if converted),
    accel in m/s² or g, gyro in rad/s or deg/s.
    """
    import csv

    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("CSV contains no rows")

    # Extract data
    t = np.array([float(r["time"]) for r in rows], dtype=float)
    pos = np.vstack([
        [float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])] for r in rows
    ])
    vel = np.vstack([
        [float(r["vel_x"]), float(r["vel_y"]), float(r["vel_z"])] for r in rows
    ])
    orient = np.vstack([
        [float(r["orient_roll"]), float(r["orient_pitch"]), float(r["orient_yaw"])] for r in rows
    ])
    accel = np.vstack([
        [float(r["accel_x"]), float(r["accel_y"]), float(r["accel_z"])] for r in rows
    ])
    gyro = np.vstack([
        [float(r["gyro_p"]), float(r["gyro_q"]), float(r["gyro_r"])] for r in rows
    ])

    # Apply unit conversions if needed
    detect_info = {"gyro_units": None, "accel_units": None}
    accel, gyro = _apply_unit_conversions(
        accel,
        gyro,
        detect_info,
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )

    return Sequence(
        timestamps=t,
        accel=accel,
        gyro=gyro,
        segments=None,
        pos=pos,
        vel=vel,
        orient=orient,
    )


def from_maneuver_id_tsv(
    path: str,
    *,
    convert_gyro_deg_to_rad: bool | None = None,
    convert_accel_g_to_m_s2: bool | None = None,
) -> Sequence:
    """Load Maneuver-ID dataset from TSV format.

    The Maneuver-ID dataset (MIT) contains flight simulator telemetry in TSV format
    with positional and orientational data.

    Expected columns: time, xEast, yNorth, zUp, vxEast, vyNorth, vzUp, heading, pitch, roll

    Args:
        path: Path to TSV file
        convert_gyro_deg_to_rad: Convert angular rates from deg/s to rad/s (auto-detect if None)
        convert_accel_g_to_m_s2: Convert accelerations from g to m/s^2 (auto-detect if None)

    Returns:
        Sequence object with timestamps, accel, gyro, and optional pos/vel/orient fields
    """
    import csv
    import numpy as np

    rows = []
    with open(path, "r", newline="") as fh:
        # TSV format uses tab delimiter
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("TSV contains no rows")

    # Extract time
    t = np.array([float(r.get("time", r.get("Time", 0))) for r in rows], dtype=float)

    # Extract position (meters)
    pos = None
    if all(k in rows[0] for k in ("xEast", "yNorth", "zUp")):
        pos = np.vstack([
            [float(r["xEast"]), float(r["yNorth"]), float(r["zUp"])] for r in rows
        ])

    # Extract velocity (m/s)
    vel = None
    if all(k in rows[0] for k in ("vxEast", "vyNorth", "vzUp")):
        vel = np.vstack([
            [float(r["vxEast"]), float(r["vyNorth"]), float(r["vzUp"])] for r in rows
        ])

    # Extract orientation (degrees)
    orient = None
    if all(k in rows[0] for k in ("roll", "pitch", "heading")):
        orient = np.vstack([
            [float(r["roll"]), float(r["pitch"]), float(r["heading"])] for r in rows
        ])
        # Note: Orientation in Maneuver-ID is in degrees, keep as-is for now
        # Angular rates (gyro) will be computed as derivatives and already in rad/s

    # Compute accelerations from velocity if available
    accel = np.zeros((len(t), 3))
    if vel is not None and len(t) > 1:
        dt = np.diff(t)
        dt = np.where(dt > 0, dt, 1e-6)  # avoid division by zero
        # Central difference for acceleration
        accel[1:-1] = (vel[2:] - vel[:-2]) / (dt[1:, np.newaxis] + dt[:-1, np.newaxis])
        # Forward/backward difference at boundaries
        accel[0] = (vel[1] - vel[0]) / dt[0]
        accel[-1] = (vel[-1] - vel[-2]) / dt[-1]
    else:
        # If no velocity data, use small random noise as placeholder
        accel = 0.01 * np.random.standard_normal((len(t), 3))

    # Compute angular rates from orientation if available
    gyro = np.zeros((len(t), 3))
    if orient is not None and len(t) > 1:
        dt = np.diff(t)
        dt = np.where(dt > 0, dt, 1e-6)
        # Central difference for angular rates
        gyro[1:-1] = (orient[2:] - orient[:-2]) / (dt[1:, np.newaxis] + dt[:-1, np.newaxis])
        # Forward/backward difference at boundaries
        gyro[0] = (orient[1] - orient[0]) / dt[0]
        gyro[-1] = (orient[-1] - orient[-2]) / dt[-1]
    else:
        # If no orientation data, use small random noise as placeholder
        gyro = 0.01 * np.random.standard_normal((len(t), 3))

    # Apply unit conversions if needed (accel from velocity is already in m/s^2)
    detect_info = {"gyro_units": "rad/s", "accel_units": "m/s2"}
    # Note: accel and gyro are already derived in SI units, so conversions mostly N/A

    return Sequence(
        timestamps=t,
        accel=accel,
        gyro=gyro,
        segments=None,
        pos=pos,
        vel=vel,
        orient=orient,
    )


def from_garmin_g1000_csv(
    path: str,
    *,
    convert_gyro_deg_to_rad: bool | None = True,
    convert_accel_g_to_m_s2: bool | None = False,
) -> Sequence:
    """Load Garmin G1000 telemetry from CSV format.

    This loader handles CSV files exported from Garmin G1000 avionics or similar
    General Aviation glass cockpit systems.

    Expected columns: time, latitude, longitude, altitude_ft, airspeed_kts,
    groundspeed_kts, vertical_speed_fpm, roll_deg, pitch_deg, heading_deg,
    accel_x, accel_y, accel_z, gyro_p, gyro_q, gyro_r

    Args:
        path: Path to CSV file
        convert_gyro_deg_to_rad: Convert angular rates from deg/s to rad/s (default True)
        convert_accel_g_to_m_s2: Convert accelerations from g to m/s^2 (default False)

    Returns:
        Sequence object with timestamps, accel, gyro, and optional pos/vel/orient fields
    """
    import csv

    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("CSV contains no rows")

    # Extract time
    t = np.array([float(r["time"]) for r in rows], dtype=float)

    # Extract accelerations (m/s^2 or g, depending on source)
    accel = np.vstack([
        [float(r["accel_x"]), float(r["accel_y"]), float(r["accel_z"])] for r in rows
    ])

    # Extract gyro (deg/s or rad/s, depending on source)
    gyro = np.vstack([
        [float(r["gyro_p"]), float(r["gyro_q"]), float(r["gyro_r"])] for r in rows
    ])

    # Extract orientation (degrees)
    orient = None
    if all(k in rows[0] for k in ("roll_deg", "pitch_deg", "heading_deg")):
        orient = np.vstack([
            [float(r["roll_deg"]), float(r["pitch_deg"]), float(r["heading_deg"])] for r in rows
        ])

    # Position: lat/lon/alt (note: simplified, not full conversion to meters)
    pos = None
    if all(k in rows[0] for k in ("latitude", "longitude", "altitude_ft")):
        # Store as lat, lon, alt_meters
        pos = np.vstack([
            [float(r["latitude"]), float(r["longitude"]), float(r["altitude_ft"]) * 0.3048]
            for r in rows
        ])

    # Velocity: airspeed, groundspeed, vertical_speed (convert to m/s)
    vel = None
    if all(k in rows[0] for k in ("airspeed_kts", "groundspeed_kts", "vertical_speed_fpm")):
        # Store as airspeed_m/s, groundspeed_m/s, vs_m/s
        vel = np.vstack([
            [
                float(r["airspeed_kts"]) * 0.514444,  # knots to m/s
                float(r["groundspeed_kts"]) * 0.514444,
                float(r["vertical_speed_fpm"]) * 0.00508,  # fpm to m/s
            ]
            for r in rows
        ])

    # Apply unit conversions
    # Note: Assume Garmin G1000 CSV has gyro in deg/s and accel in m/s^2 by default
    # User can override with parameters
    detect_info = {"gyro_units": "deg/s", "accel_units": "m/s2"}
    
    accel, gyro = _apply_unit_conversions(
        accel,
        gyro,
        detect_info,
        convert_gyro_deg_to_rad=convert_gyro_deg_to_rad,
        convert_accel_g_to_m_s2=convert_accel_g_to_m_s2,
    )

    return Sequence(
        timestamps=t,
        accel=accel,
        gyro=gyro,
        segments=None,
        pos=pos,
        vel=vel,
        orient=orient,
    )
