# Changelog

All notable changes to this project are documented in this file.

## Unreleased

- Fix: Corrected IoU calculation for segment overlap in `src/maneuvers/eval.py` (union computed as len(a)+len(b)-intersection).
- Fix: Improved `moving_average` to use `np.convolve(..., mode='same')` and handle very short inputs without indexing errors (`src/maneuvers/preprocessing.py`).
- Fix: Robust CSV loader axis/column detection to support varied simulator outputs and case-insensitive headers (`src/maneuvers/data/loader.py`).
- Tidy: Cleaned duplicate imports in `src/maneuvers/classify.py`.
- Docs: Updated `README.md` to list `docs/` pages and added this changelog entry.
- Meta: Bumped minimum Python to `>=3.10` and added developer extras in `pyproject.toml` (`[project.optional-dependencies]`) for easier local testing of optional backends.


