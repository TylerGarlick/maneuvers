# maneuvers

Find and identify maneuvers from Flight Data

## Getting Started

This is a minimal Python application that demonstrates a small CLI called `maneuvers`.

### Prerequisites
- Python 3.8 or newer
- pip (bundled with Python)

### Install locally (editable)
1. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate
```

2. Install the package in editable mode with pip:

```bash
pip install -e .
```

This installs a console script `maneuvers` pointing to the package entry point.

### Running the app
Run the CLI directly:

```bash
maneuvers --name Alice
```

Or run the package with Python:

```bash
python -m maneuvers --name Bob
```

### Development notes
- The project uses `pyproject.toml` (setuptools) for packaging metadata.
- Add runtime dependencies to `requirements.txt` or to `pyproject.toml`.

---

### Running tests ✅
We include a tiny smoke test using `pytest`.

Install test dependencies and run tests:

```bash
pip install pytest
pytest
```

If you plan to publish this package, update the metadata in `pyproject.toml` (author, description, version) and consider adding more tests and CI workflows.

## Resources

### Challenges 
- https://arxiv.org/abs/2108.11503
- https://arxiv.org/abs/2211.15552

[Manuever Patterns](https://maneuver-id.mit.edu/maneuvers-0/)

## Tasks

- [X] Setup Python Project
- [ ] Implement data loader and synthetic data generator
- [ ] Implement preprocessing & feature extraction
- [ ] Implement threshold-based detector baseline
- [ ] Implement evaluation metrics and tests

---

## Baseline scaffold (implemented)

I added a minimal baseline to help get started quickly:

- Synthetic data generator: `src/maneuvers/data/loader.py` (function `generate_synthetic_sequence`)
- Preprocessing & features: `src/maneuvers/preprocessing.py` (`compute_features_from_sequence`)
- Threshold detector: `src/maneuvers/detection.py` (`detect_segments`)
- Evaluation utilities: `src/maneuvers/eval.py` (`segment_iou`, `evaluate_detection`)
- CLI commands: `src/maneuvers/cli.py` (`detect-synthetic`, `eval-synthetic`)
- Unit tests: `tests/test_loader.py`, `tests/test_detection.py`, `tests/test_eval.py`, `tests/test_cli.py`

Quickstart (after creating a virtualenv and installing requirements):

```bash
pip install -r requirements.txt
pytest
```

Run detection & evaluation on synthetic data with the baseline:

```bash
python -m maneuvers.cli detect-synthetic --duration 10 --fs 100 --threshold 0.4
python -m maneuvers.cli eval-synthetic --duration 10 --fs 100 --threshold 0.4
```

Train a simple classifier on a synthetic sequence and save it:

```bash
python -m maneuvers.cli train-synthetic --out model.joblib --duration 10 --fs 100
```

Use a saved model to label detected segments:

```bash
python -m maneuvers.cli detect-synthetic --duration 10 --fs 100 --threshold 0.4 --model model.joblib
```

Notebook demo:

- `examples/notebooks/detection_classification_demo.ipynb` demonstrates the full pipeline (generate → detect → train → evaluate) and includes CI-snippets to execute the notebook in CI.

These additions provide a repeatable baseline for running detection experiments and iterating toward the challenge objectives (segmentation, classification, evaluation).

---

## Documentation

Comprehensive documentation is available in the `docs/` folder — start at `docs/index.md`. Key pages include:

- `docs/index.md` — overview and navigation
- `docs/installation.md` — installation and environment setup
- `docs/quickstart.md` — quickstart examples and commands
- `docs/data.md` — data formats and synthetic dataset generation
- `docs/maneuvers_catalog.md` — maneuver catalog and labels
- `docs/preprocessing.md` — preprocessing and feature extraction
- `docs/detection.md` — detection algorithm and usage
- `docs/classification.md` — classifier training and evaluation
- `docs/evaluation.md` — metrics and evaluation workflows
- `docs/cli.md` — CLI commands and examples
- `docs/notebook.md` — demo notebook usage and CI execution
- `docs/ci.md` — CI configuration and how to run tests/notebooks in CI
- `docs/contributing.md` — contribution guidelines and code style

View the docs locally with any Markdown viewer or host them as a static site (e.g., GitHub Pages).

---

## Continuous Integration ✅

A GitHub Actions workflow has been added: `.github/workflows/ci.yml`.

This workflow runs on push and pull requests to `main` and performs the following checks:

- Installs dependencies and the package in editable mode
- Runs the test suite with `pytest`
- Runs `black --check` and `flake8` as basic style checks
- Executes the demo notebook `examples/notebooks/detection_classification_demo.ipynb`, converts it to HTML, and uploads the HTML as a CI artifact so results and figures are inspectable from the workflow run

This should help keep the project testable and maintainable when collaborating via PRs, and ensures the demo notebook remains executable in CI.

