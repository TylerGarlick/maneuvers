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

