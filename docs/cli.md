# CLI (Command-line interface)

Location: `src/maneuvers/cli.py`

This project exposes a small CLI (built with `typer`) for quick experiments.

Available commands

- `detect-synthetic`
  - Generates a synthetic sequence and runs detection.
  - Options: `--duration`, `--fs`, `--threshold`, `--min-len`, `--out` (write predictions), `--model` (path to a saved model to label detected segments).

  Example:
  ```bash
  python -m maneuvers.cli detect-synthetic --duration 10 --fs 100 --threshold 0.4
  python -m maneuvers.cli detect-synthetic --model model.joblib
  ```

- `eval-synthetic`
  - Runs detection on a synthetic sequence and prints TP/FP/FN/precision/recall/F1.

  Example:
  ```bash
  python -m maneuvers.cli eval-synthetic --duration 10 --fs 100 --threshold 0.4
  ```

- `train-synthetic`
  - Trains a simple classifier on a synthetic sequence and saves the trained object.

  Example:
  ```bash
  python -m maneuvers.cli train-synthetic --out model.joblib --duration 10 --fs 100
  ```

Notes

- CLI commands are useful for quick checks and CI; for production or large experiments you will want to write scripts or notebooks that batch data and keep track of models/metrics.