# maneuvers

Detect and classify flight maneuvers from time-series sensor data. The project includes a synthetic data generator, baseline detection and classification pipelines, a Typer-powered CLI, and notebooks for experimentation. **Now supports real-world training data from Maneuver-ID (MIT) and NASA DASHlink datasets.**

## Installation
- Requires Python >= 3.10
- Create and activate a virtual environment, then install dependencies and the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run tests to verify the setup:

```bash
pytest
```

## Quickstart
Common end-to-end actions (see [docs/quickstart.md](docs/quickstart.md) for details):

```bash
# Detect maneuvers on a synthetic sequence
maneuvers detect-synthetic --duration 10 --fs 100 --threshold 0.4

# Evaluate detection vs. ground truth
maneuvers eval-synthetic --duration 10 --fs 100 --threshold 0.4

# Train and save a classifier (enhanced with multiple sequences and maneuvers_small)
maneuvers train-synthetic --out model.joblib --duration 10 --fs 100 --num-sequences 20

# Use a saved model to label detected segments
maneuvers detect-synthetic --duration 10 --fs 100 --threshold 0.4 --model model.joblib

# Export labeled segments with scores
maneuvers export-detect --out maneuvers_export.json --model model.joblib

# Generate 3D flight path plot
maneuvers plot-3d --out flight_path_3d.png --model model.joblib

# Execute the demo notebook (requires Jupyter)
jupyter nbconvert --execute examples/notebooks/detection_classification_demo.ipynb --to html
```

## External Datasets

This project now supports training on real-world flight data from external sources:

### Fetching Datasets

```bash
# Set up dataset directories and download instructions
python -m maneuvers.data.fetch_datasets

# Or fetch specific datasets
python -m maneuvers.data.fetch_datasets --dataset maneuver-id
python -m maneuvers.data.fetch_datasets --dataset dashlink
```

### Training with External Data

```bash
# Train on Maneuver-ID dataset (after downloading)
maneuvers train --data-dir data/external/maneuver-id --out model_maneuver_id.joblib

# Train on sample Garmin G1000 data
maneuvers train --data-dir examples/data --data-format garmin-g1000 --out model_g1000.joblib
```

### Available Datasets

- **Maneuver-ID (MIT)**: High-fidelity flight simulator data with ~30 labeled maneuver types
  - Source: https://maneuver-id.mit.edu/data/
  - Requires registration and Data Sharing Agreement
  - Sample CSV available at `examples/data/maneuver_id_sample.csv`

- **NASA DASHlink**: Flight test data from various NASA aircraft
  - Source: https://c3.ndc.nasa.gov/dashlink/projects/85/
  - Requires registration for full dataset

See [docs/data.md](docs/data.md) for detailed information on dataset formats and usage.

## Documentation
All docs live under [docs/index.md](docs/index.md). Key pages:
- [docs/index.md](docs/index.md) — overview and navigation
- [docs/installation.md](docs/installation.md) — installation and environment setup
- [docs/quickstart.md](docs/quickstart.md) — quickstart commands
- [docs/improvements.md](docs/improvements.md) — **NEW**: improved detection methods and enhanced features
- [docs/data.md](docs/data.md) — data formats, external datasets, and simulator CSV loaders
- [docs/maneuvers_catalog.md](docs/maneuvers_catalog.md) — maneuver catalog and labels
- [docs/preprocessing.md](docs/preprocessing.md) — preprocessing and feature extraction
- [docs/detection.md](docs/detection.md) — baseline detector
- [docs/classification.md](docs/classification.md) — training and evaluation
- [docs/evaluation.md](docs/evaluation.md) — metrics
- [docs/cli.md](docs/cli.md) — command reference
- [docs/notebook.md](docs/notebook.md) — notebooks and CI execution
- [docs/ci.md](docs/ci.md) — CI details
- [docs/contributing.md](docs/contributing.md) — contribution guidelines; see [docs/CHANGELOG.md](docs/CHANGELOG.md) for notable changes

## Examples and data
- Synthetic demo artifacts: [examples/synthetic.csv](examples/synthetic.csv) and [examples/synthetic_gt.json](examples/synthetic_gt.json)
- Flight-simulator CSV samples: [examples/simulators/](examples/simulators/) for X-Plane, FlightGear, SimConnect/MSFS, and JSBSim
- External dataset sample: [examples/data/maneuver_id_sample.csv](examples/data/maneuver_id_sample.csv) — Garmin G1000 format
- Notebooks: [examples/notebooks/](examples/notebooks/) including [examples/notebooks/detection_classification_demo.ipynb](examples/notebooks/detection_classification_demo.ipynb) plus optional XGBoost and CNN demos

## Testing
Run the full suite:

```bash
pytest
```

Notebook execution in CI is covered by nbconvert/nbval; see [docs/ci.md](docs/ci.md).

## Contributing
See [docs/contributing.md](docs/contributing.md) for workflow, style, and release notes (with [docs/CHANGELOG.md](docs/CHANGELOG.md)).
# maneuvers

