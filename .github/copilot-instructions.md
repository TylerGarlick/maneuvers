# Copilot / AI Agent Instructions for the `maneuvers` repo

Purpose: give an AI coding agent exactly the repository knowledge needed to be productive.

- Quick summary: This project is a small Python package that generates synthetic flight sensor sequences, extracts features, runs a threshold-based segmentation baseline, trains a simple classifier, and evaluates results. Key components: CLI (`src/maneuvers/cli.py`), data generator/loader (`src/maneuvers/data/loader.py`), feature extraction (`src/maneuvers/preprocessing.py`), detector (`src/maneuvers/detection.py`), classifier utilities (`src/maneuvers/classify.py`), evaluation (`src/maneuvers/eval.py`).

- How the data flows: `generate_synthetic_sequence()` -> `compute_features_from_sequence()` -> `detect_segments()` -> optional `classify.predict_segment_labels()` -> `evaluate_detection()`.

- Important file anchors (read these first):
  - `README.md` — project overview and quick commands.
  - `pyproject.toml` — packaging and console-script entry (`maneuvers`).
  - `src/maneuvers/cli.py` — Typer CLI commands and examples of typical workflows.
  - `src/maneuvers/data/loader.py` — `Sequence` dataclass, `generate_synthetic_sequence()` and CSV I/O.
  - `src/maneuvers/preprocessing.py` — canonical feature names: `accel_mag`, `accel_smooth` (used by detector), `gyro_mag`.
  - `src/maneuvers/detection.py` — baseline segmentation (`threshold`), segments are `(start, end)` with `end` exclusive.
  - `src/maneuvers/classify.py` — training helpers; returns model dict saved with `joblib.dump()` and expected shape for `predict_segment_labels()`.

- Naming / conventions to follow when editing or adding code:
  - Features DataFrame columns are expected exactly as in `compute_features_from_sequence()` (`accel_mag`, `accel_smooth`, `gyro_mag`). Detection code indexes `features['accel_smooth'].values`.
  - Segments are index ranges `(start, end)` where `end` is exclusive. Many helpers and tests assume this convention.
  - `Sequence` objects (from `data/loader.py`) have `timestamps`, `accel` (N,3), `gyro` (N,3), and `segments` list of `(s,e,label)` for ground truth.
  - Model objects saved/loaded via `classify.save_model()` / `load_model()` are dicts with keys: `model`, `encoder`, `cv_scores`.
  - Be conservative about cross-validation: `train_classifier()` already contains small-sample fallbacks; avoid changing CV logic without adjusting related tests.

- Developer workflows and commands (copy-paste):
  - Setup virtualenv and editable install:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
    pip install -r requirements.txt  # for test/dev deps
    ```

  - Run CLI examples (same examples appear in `README.md`):

    ```bash
    maneuvers detect-synthetic --duration 10 --fs 100 --threshold 0.4
    maneuvers train-synthetic --out model.joblib --duration 10 --fs 100
    maneuvers eval-synthetic --duration 10 --fs 100 --threshold 0.4
    ```

  - Run tests: `pytest` (CI uses a similar flow in `.github/workflows/ci.yml`).

- Patterns to preserve when contributing:
  - Keep synthetic data deterministic where seed=0 is used; tests rely on reproducible sequences.
  - Unit tests exercise small examples (see `tests/`); prefer minimal, focused changes to avoid breaking small-sample training logic.
  - CLI commands use `typer` in `src/maneuvers/cli.py`; prefer adding new commands there rather than ad-hoc scripts.

- Integration & external dependencies:
  - Dependencies are declared in `pyproject.toml` and `requirements.txt`. Notable libs: `numpy`, `pandas`, `scipy`, `typer`, `sklearn`, `joblib`.
  - The repository ships example notebooks under `examples/` and dataset helpers under `examples/datasets/` and `src/maneuvers/data/`.
  - CI executes the demo notebook in `.github/workflows/ci.yml`. If you add heavy notebook cells, make them optional or gated.

- Helpful code snippets (use these exact forms when writing examples/tests):
  - Generate and process a sequence:

    ```py
    from maneuvers.data.loader import generate_synthetic_sequence
    from maneuvers.preprocessing import compute_features_from_sequence

    seq = generate_synthetic_sequence(duration_s=5.0, fs=100)
    feats = compute_features_from_sequence(seq)
    ```

  - Run detection and interpret segments:

    ```py
    from maneuvers.detection import detect_segments
    preds = detect_segments(feats, method='threshold', threshold=0.5, min_len=5)
    # preds is a list of (start, end) with end exclusive
    ```

- When unsure, open these files first: `src/maneuvers/cli.py`, `src/maneuvers/data/loader.py`, `src/maneuvers/preprocessing.py`, `src/maneuvers/detection.py`, `src/maneuvers/classify.py`, and `README.md`.

If anything in this guidance is unclear or you want additional examples (unit test templates, PR checklist, or CI notes), tell me which section to expand and I will iterate.
