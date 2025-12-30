# Data & Synthetic Examples

This project includes a small synthetic data generator and a CSV loader to make experimenting easy.

Key files:

- `src/maneuvers/data/loader.py` — main data utilities
- `examples/synthetic.csv` — a sample synthetic sequence
- `examples/synthetic_gt.json` — corresponding ground-truth segments

Sequence structure

The code defines a `Sequence` object (a simple dataclass) with fields:

- `timestamps`: array of times (seconds)
- `accel`: Nx3 array of accelerometer measurements (ax, ay, az)
- `gyro`: Nx3 array of gyroscope measurements (gx, gy, gz)
- `segments`: list of `(start_idx, end_idx, label)` indicating maneuver intervals (end index is exclusive)

Generating synthetic data

- Use `generate_synthetic_sequence(duration_s=10.0, fs=100)` to create reproducible demo data.
- Use `from_csv(path)` to load CSV files with columns: `t, ax, ay, az, gx, gy, gz`.

Simulator export formats

This project also provides convenience loaders for common flight-simulator telemetry exports:

- `from_xplane_csv(path)` — X-Plane-like CSVs (e.g., `time, ax, ay, az, p, q, r`)
- `from_flightgear_csv(path)` — FlightGear-like CSVs (e.g., `time, udot, vdot, wdot, p, q, r`)
- `from_simconnect_csv(path)` — SimConnect / MSFS CSVs (e.g., `Time, AccelerationX, AccelerationY, AccelerationZ, AngularVelocityX, AngularVelocityY, AngularVelocityZ`)
- `from_jsbsim_csv(path)` — JSBSim CSVs (similar to X-Plane)
- `from_csv_filelike(fileobj, ...)` — heuristically load from a file-like CSV stream (useful for sockets/streams)
- `from_json(path)` — load JSON telemetry with per-axis arrays or nested accel/gyro arrays

Example CSV and JSON files for these formats are included under `examples/simulators/` to help you try the loaders quickly.

Unit detection & conversion

Loaders accept options to convert detected units to normalized SI units:

- `convert_gyro_deg_to_rad`: False|True|None (None = auto-detect deg/s and convert to rad/s)
- `convert_accel_g_to_m_s2`: False|True|None (None = auto-detect g and convert to m/s^2)

Use example:

```python
from maneuvers.data.loader import from_simconnect_csv
seq = from_simconnect_csv("examples/simulators/simconnect_example.csv", convert_gyro_deg_to_rad=None)
```

Why synthetic data? It helps you:
- Test and debug algorithms quickly
- Build repeatable unit tests and CI fixtures
- Explore detection thresholds and classifier behavior without downloading external datasets