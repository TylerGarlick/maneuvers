# CLI (Command-line interface)

Location: `src/maneuvers/cli.py`

This project exposes a small CLI (built with `typer`) for quick experiments.

Available commands

- `detect-synthetic`
  - Generates a synthetic sequence and runs detection.
  - Options: `--duration`, `--fs`, `--threshold`, `--min-len`, `--out` (write predictions), `--model` (path to a saved model to label detected segments).

  Example:
  ```bash
  maneuvers detect-synthetic --duration 10 --fs 100 --threshold 0.4
  maneuvers detect-synthetic --model model.joblib
  ```

- `eval-synthetic`
  - Runs detection on a synthetic sequence and prints TP/FP/FN/precision/recall/F1.

  Example:
  ```bash
  maneuvers eval-synthetic --duration 10 --fs 100 --threshold 0.4
  ```

- `train-synthetic`
  - Trains a classifier on multiple synthetic sequences and optionally maneuvers_small dataset, with hyperparameter tuning.
  - Options: `--duration`, `--fs`, `--out`, `--model-type` (rf, logistic, mlp, xgb), `--num-sequences` (default 10), `--use-maneuvers-small` (default True).

  Example:
  ```bash
  maneuvers train-synthetic --out model.joblib --duration 10 --fs 100
  maneuvers train-synthetic --out model.joblib --num-sequences 20 --use-maneuvers-small true
  ```

- `train`
  - Trains a classifier on external datasets from real flight sources.
  - Options: `--data-dir` (required, path to dataset directory), `--data-format` (auto, csv, tsv, maneuver-id, garmin-g1000), `--out`, `--model-type` (rf, logistic, mlp, xgb), `--fs`.

  Example:
  ```bash
  maneuvers train --data-dir data/external/maneuver-id --out model_maneuver_id.joblib
  maneuvers train --data-dir examples/data --data-format garmin-g1000 --out model_g1000.joblib
  ```

- `export-detect`
  - Generates synthetic data, detects maneuvers, labels with model, computes scores, and exports to JSON/CSV.
  - Options: `--duration`, `--fs`, `--threshold`, `--min-len`, `--out`, `--model`.

  Example:
  ```bash
  maneuvers export-detect --out maneuvers_export.json --model model.joblib
  ```

- `plot-3d`
  - Generates synthetic data, detects maneuvers, and saves a 3D plot of the flight path with maneuver markers.
  - Options: `--duration`, `--fs`, `--threshold`, `--min-len`, `--out`, `--model`, `--interactive` (for optional HTML).

  Example:
  ```bash
  maneuvers plot-3d --out flight_path_3d.png --model model.joblib
  ```

Notes

- CLI commands are useful for quick checks and CI; for production or large experiments you will want to write scripts or notebooks that batch data and keep track of models/metrics.
- The new `train` command enables training on real external datasets like Maneuver-ID and NASA DASHlink, moving beyond synthetic data for production-quality models.
