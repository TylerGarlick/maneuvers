# Data & Synthetic Examples

This project includes a small synthetic data generator and a CSV loader to make experimenting easy. It also supports external datasets from real flight sources.

## Key files

- `src/maneuvers/data/loader.py` — main data utilities
- `examples/synthetic.csv` — a sample synthetic sequence
- `examples/synthetic_gt.json` — corresponding ground-truth segments
- `examples/data/maneuver_id_sample.csv` — sample from Garmin G1000 format
- `data/external/` — directory for external datasets (see `data/external/README.md`)

## Sequence structure

The code defines a `Sequence` object (a simple dataclass) with fields:

- `timestamps`: array of times (seconds)
- `accel`: Nx3 array of accelerometer measurements (ax, ay, az)
- `gyro`: Nx3 array of gyroscope measurements (gx, gy, gz)
- `segments`: list of `(start_idx, end_idx, label)` indicating maneuver intervals (end index is exclusive)
- `pos`: (optional) Nx3 array of positions (x, y, z)
- `vel`: (optional) Nx3 array of velocities (vx, vy, vz)
- `orient`: (optional) Nx3 array of orientations (roll, pitch, yaw)

## Generating synthetic data

- Use `generate_synthetic_sequence(duration_s=10.0, fs=100)` to create reproducible demo data.
- Use `from_csv(path)` to load CSV files with columns: `t, ax, ay, az, gx, gy, gz`.

## External datasets

### Maneuver-ID (MIT)

The Maneuver-ID dataset contains labeled flight maneuver data from high-fidelity flight simulators used in Air Force pilot training.

**Source**: https://maneuver-id.mit.edu/data/

**Format**: TSV (tab-separated values) with columns like `time, xEast, yNorth, zUp, vxEast, vyNorth, vzUp, heading, pitch, roll`

**Loader**: Use `from_maneuver_id_tsv(path)` to load Maneuver-ID TSV files

**Sample**: A small CSV sample in Garmin G1000 format is available at `examples/data/maneuver_id_sample.csv`

**Fetching**: Run `python -m maneuvers.data.fetch_datasets --dataset maneuver-id` to create the dataset directory and instructions

### NASA DASHlink

NASA DASHlink provides flight test data from various aircraft with sensor telemetry.

**Source**: https://c3.ndc.nasa.gov/dashlink/projects/85/

**Format**: Various CSV/MAT formats depending on the aircraft

**Fetching**: Run `python -m maneuvers.data.fetch_datasets --dataset dashlink` to create the dataset directory and instructions

### Using external datasets

```python
from maneuvers.data.loader import from_garmin_g1000_csv, from_maneuver_id_tsv

# Load Garmin G1000 format (sample data)
seq = from_garmin_g1000_csv("examples/data/maneuver_id_sample.csv")

# Load Maneuver-ID TSV format
seq = from_maneuver_id_tsv("data/external/maneuver-id/flight_001.tsv")

# Process as usual
from maneuvers.preprocessing import compute_features_from_sequence
feats = compute_features_from_sequence(seq)
```

### Training with external datasets

```bash
# Train using external data directory
maneuvers train --data-dir data/external/maneuver-id --out model_maneuver_id.joblib

# Train using sample Garmin G1000 data
maneuvers train --data-dir examples/data --data-format garmin-g1000 --out model_g1000.joblib
```

## Simulator export formats

This project also provides convenience loaders for common flight-simulator telemetry exports:

- `from_xplane_csv(path)` — X-Plane-like CSVs (e.g., `time, ax, ay, az, p, q, r`)
- `from_flightgear_csv(path)` — FlightGear-like CSVs (e.g., `time, udot, vdot, wdot, p, q, r`)
- `from_simconnect_csv(path)` — SimConnect / MSFS CSVs (e.g., `Time, AccelerationX, AccelerationY, AccelerationZ, AngularVelocityX, AngularVelocityY, AngularVelocityZ`)
- `from_jsbsim_csv(path)` — JSBSim CSVs (similar to X-Plane)
- `from_csv_filelike(fileobj, ...)` — heuristically load from a file-like CSV stream (useful for sockets/streams)
- `from_json(path)` — load JSON telemetry with per-axis arrays or nested accel/gyro arrays
- `from_garmin_g1000_csv(path)` — Garmin G1000 avionics CSV format
- `from_maneuver_id_tsv(path)` — Maneuver-ID dataset TSV format

Example CSV and JSON files for these formats are included under `examples/simulators/` to help you try the loaders quickly.

## Unit detection & conversion

Loaders accept options to convert detected units to normalized SI units:

- `convert_gyro_deg_to_rad`: False|True|None (None = auto-detect deg/s and convert to rad/s)
- `convert_accel_g_to_m_s2`: False|True|None (None = auto-detect g and convert to m/s^2)

Use example:

```python
from maneuvers.data.loader import from_simconnect_csv
seq = from_simconnect_csv("examples/simulators/simconnect_example.csv", convert_gyro_deg_to_rad=None)
```

## Data quality checks

Location: `src/maneuvers/data/quality.py`

Before training or evaluating on real data, run quality checks to filter out bad sorties (the Maneuver-ID challenge explicitly requires separating good from bad data).

### Minimum requirements

`check_minimum_requirements(seq)` validates:
- Sampling rate >= 50 Hz
- Duration >= 10 seconds
- Accel and gyro arrays present with shape (N, 3)
- Acceleration does not exceed 10g
- Velocity does not exceed 300 m/s (Mach 1)

### Trajectory continuity

`check_trajectory_continuity(seq)` runs four sub-checks and returns a combined result:

| Check | Function | What it detects | Key parameter |
|-------|----------|----------------|---------------|
| Timestamp gaps | `check_timestamp_gaps()` | Missing data / recording pauses | `gap_factor` (default 3.0x median dt) |
| Position jumps | `check_position_jumps()` | Teleportation in position data | `max_speed_m_s` (default 350 m/s) |
| Velocity discontinuities | `check_velocity_discontinuities()` | Sudden velocity changes | `max_accel_m_s2` (default 150 m/s^2) |
| Orientation discontinuities | `check_orientation_discontinuities()` | Sudden heading/pitch/roll jumps | `max_rate_deg_s` (default 500 deg/s) |

The orientation check handles angle wrapping correctly — a heading smoothly crossing 360/0 degrees is not flagged.

```python
from maneuvers.data.loader import from_maneuver_id_tsv
from maneuvers.data.quality import check_minimum_requirements, check_trajectory_continuity

seq = from_maneuver_id_tsv("flight_001.tsv")

# Basic checks
basic = check_minimum_requirements(seq)
print(basic["valid"], basic["issues"])

# Continuity checks
cont = check_trajectory_continuity(seq)
print(cont["valid"], cont["issues"])

# Detailed results per check
for name, detail in cont["details"].items():
    print(f"  {name}: valid={detail['valid']}")
```

Each check returns a dict with `valid` (bool) and details about flagged locations (index + value). Checks that require optional data (pos, vel, orient) are skipped gracefully when that data is absent, returning `skipped: True`.

## Why synthetic data?

It helps you:
- Test and debug algorithms quickly
- Build repeatable unit tests and CI fixtures
- Explore detection thresholds and classifier behavior without downloading external datasets

## Moving from synthetic to real data

1. Start with synthetic data and the sample CSV to validate your pipeline
2. Use `fetch_datasets.py` to set up external dataset directories
3. Download real datasets (requires registration for some sources)
4. Train models on real data using the `train` CLI command
5. Evaluate and compare performance between synthetic and real data