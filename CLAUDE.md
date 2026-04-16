# CLAUDE.md

## Project Overview

**maneuvers** is a Python library for detecting and classifying aircraft flight maneuvers from time-series sensor data. It targets the [MIT/USAF Maneuver Identification Challenge](https://maneuver-id.mit.edu/) — a challenge to automatically identify and label each maneuver flown from a catalog of ~30 maneuver types using flight simulator telemetry from the Air Force's Pilot Training Next (PTN) program.

## Quick Reference

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run tests (65 tests, all should pass)
pytest

# Lint and format
black --check src/ tests/
flake8 src/ tests/

# CLI entry point
maneuvers --help
```

## Architecture

```
src/maneuvers/
  cli.py              # Typer CLI — primary interface (detect, train, eval, export, plot)
  detection.py         # Segmentation algorithms (currently threshold-only)
  classify.py          # Training & prediction (RF, Logistic, MLP, XGBoost, CNN)
  preprocessing.py     # Feature extraction (accel/gyro magnitude, smoothing, FFT)
  eval.py              # IoU-based segment matching, precision/recall/F1
  export.py            # JSON/CSV export of labeled segments
  visualize.py         # 3D scatter plots, confusion matrices, PR curves
  data/
    loader.py          # Sequence dataclass + loaders (CSV, TSV, Garmin G1000, T-6A)
    maneuvers_catalog.py  # ~30 maneuver type definitions + parametric synthesis
    fetch_datasets.py  # Dataset download instructions (Maneuver-ID, DASHlink)
    quality.py         # Basic data quality validation
  models/
    cnn.py             # Keras 1D CNN and multi-branch CNN classifiers
```

## Key Concepts

- **Sequence**: Core dataclass (`data/loader.py`) holding timestamps, accel (N,3), gyro (N,3), optional pos/vel/orient, and ground-truth segments `[(start, end, label), ...]`.
- **Detection**: Find temporal boundaries of maneuvers. Currently only threshold on smoothed accel magnitude.
- **Classification**: Label each detected segment as one of ~30 maneuver types. Uses aggregate features (mean/std/max/min accel, gyro mag, energy, FFT dominant freq) fed to sklearn models.
- **Evaluation**: Greedy IoU matching between predicted and ground-truth segments, then precision/recall/F1.

## Maneuver-ID Challenge

The challenge (https://maneuver-id.mit.edu/) has three sub-tasks:

1. **Data Quality Sorting**: Separate physically feasible (good) from infeasible (bad) sorties. Good data has realistic, continuous trajectories with identifiable maneuvers. Bad data has jumps, straight lines, or physics violations.
2. **Maneuver Detection**: Place start/stop boundaries for each maneuver in the time series.
3. **Maneuver Classification**: Label each detected segment from a catalog of ~30 maneuver types (aileron rolls, steep turns, barrel rolls, loops, spins, lazy 8s, split-S, ILS approaches, etc.).

### Challenge Data Format

Maneuver-ID TSV files contain: `time, xEast, yNorth, zUp, vxEast, vyNorth, vzUp, heading, pitch, roll` — positions, velocities, and orientations (not raw accel/gyro). Loader: `Sequence.from_maneuver_id_tsv()`.

## What Is Implemented

- Synthetic data generation for ~30 maneuver types (`maneuvers_catalog.py`)
- Data loaders for CSV, TSV (Maneuver-ID), Garmin G1000, T-6A, flight simulators
- Threshold-based segment detection on smoothed acceleration magnitude
- Feature extraction: accel/gyro magnitude, moving average, FFT energy/dominant freq
- Classification: RandomForest, LogisticRegression, MLP, XGBoost, 1D CNN, multi-branch CNN
- IoU-based evaluation with precision/recall/F1
- CLI with commands: `detect-synthetic`, `eval-synthetic`, `train-synthetic`, `detect-real`, `train`, `plot-3d`, `export-detect`
- 65 passing tests, CI with Python 3.10/3.11 matrix (with/without optional deps)
- 4 Jupyter demo notebooks

## What Is Left to Implement

### 1. Data Quality Sorting (Challenge Sub-task)
`quality.py` only checks basic thresholds (sampling rate, duration, accel < 10g). Missing:
- **Trajectory continuity detection**: identify jumps/teleportation in position data
- **Straight-line / idle detection**: flag sorties with no meaningful maneuvers
- **Physics violation detection**: check that position, velocity, and orientation are consistent (e.g., integrate velocity and compare to position)
- **Good/bad sortie classifier**: a binary classifier or rule-based system to sort entire sorties
- **Batch processing CLI command** for sorting a full dataset directory

### 2. Detection Beyond Threshold
`detection.py` has only `threshold_segment()` on smoothed accel magnitude. Missing:
- **Position/velocity/orientation-based detection**: the real Maneuver-ID data has pos/vel/orient, not accel/gyro — detection must work on these signals
- **Change-point detection**: CUSUM, PELT, or Bayesian changepoint methods for finding segment boundaries
- **Energy-based detection**: detect maneuver onset from kinetic/potential energy changes
- **Learned detection**: sliding-window binary classifier (is this window part of a maneuver?)
- **Multi-signal fusion**: combine accel, gyro, position, velocity, orientation signals for robust detection
- **Post-processing**: merge close segments, split long segments, non-maximum suppression, minimum gap enforcement

### 3. Classification on Real Data Features
Feature engineering in `classify.py` uses only accel/gyro aggregate stats. Missing:
- **Position/velocity/orientation features**: curvature, turn rate from heading, climb rate from zUp, bank angle from roll — critical for real Maneuver-ID data
- **Temporal shape features**: DTW distance to reference maneuver templates, shape descriptors
- **Sequence-level features**: context from preceding/following segments
- **Spectral features on all channels**: FFT on position, velocity, orientation signals
- **Normalized features**: altitude-independent, speed-independent representations

### 4. End-to-End Real Data Pipeline
- `fetch_datasets.py` creates README placeholders only — no actual data download automation
- No batch training/evaluation pipeline on full Maneuver-ID dataset (directory of TSV files)
- No **challenge submission format** generation
- No **benchmark evaluation** against published baselines from the challenge papers
- No pipeline for `maneuvers train --data-dir <maneuver-id-dir>` with proper train/val/test splits on real data

### 5. Model Improvements
- No **ensemble methods** (combining RF + CNN + XGBoost predictions)
- No **hyperparameter optimization** workflow for real data (beyond basic grid search)
- No **data augmentation** for real data (time warping, noise injection, crop/pad)
- No **confidence calibration** for predicted labels
- No **class imbalance handling** for rare maneuver types in real data (beyond basic over/undersampling in CNN)

### 6. Evaluation Gaps
- No **per-maneuver-type performance breakdown** (confusion analysis by maneuver)
- No **detection + classification joint evaluation** (end-to-end F1 where both boundaries and label must be correct)
- No **comparison with published baselines** from the challenge papers

## Development Guidelines

- Python >= 3.10 required
- Format with `black`, lint with `flake8` (config in `.flake8`)
- Tests in `tests/` — run `pytest`. All tests must pass before committing.
- Optional dependencies: `xgboost`, `tensorflow`, `matplotlib` — code must degrade gracefully without them
- CI runs matrix tests on Python 3.10 + 3.11 with/without optional deps
- CLI built with Typer (`cli.py`); add new commands there
- Notebooks in `examples/notebooks/` are executed in CI via nbval/nbconvert


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
