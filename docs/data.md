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

Example CSV files for each of these formats are included under `examples/simulators/` to help you try the loaders quickly.

Why synthetic data? It helps you:
- Test and debug algorithms quickly
- Build repeatable unit tests and CI fixtures
- Explore detection thresholds and classifier behavior without downloading external datasets