# Installation (Beginner-friendly)

This project uses Python. A virtual environment keeps dependencies isolated and is recommended.

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install the package in editable mode so local edits are available automatically:

```bash
pip install -e .
```

4. Run tests:

```bash
pytest
```

Notes:
- If you want to run the demo notebook, install `nbconvert` and `nbval` (already in `requirements.txt`) and use `jupyter nbconvert --execute` or run the notebook with `pytest --nbval` in CI.
- If your install fails, try upgrading `pip` first: `python -m pip install --upgrade pip`.