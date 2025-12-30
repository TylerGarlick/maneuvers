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

Why synthetic data? It helps you:
- Test and debug algorithms quickly
- Build repeatable unit tests and CI fixtures
- Explore detection thresholds and classifier behavior without downloading external datasets