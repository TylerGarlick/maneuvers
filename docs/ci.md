# Continuous Integration (CI)

Workflow file: `.github/workflows/ci.yml`

What the workflow does:

- Checks out the repository
- Sets up Python (matrix across versions)
- Installs dependencies (`pip install -r requirements.txt` and `pip install -e .`)
- Runs `pytest` to execute unit tests
- Runs `black --check` and `flake8` for formatting and linting
- Executes the demo notebook (via `nbval` and `nbconvert`) and uploads an HTML artifact (`artifacts/detection_classification_demo.html`) so you can inspect figures & outputs from the CI run

How to update

- To run the notebook in a separate job, copy the `nbval` step into its own job and add caching for `pip` or the virtualenv to speed up subsequent runs.
- To collect more artifacts (e.g., JSON results, images) add steps to write artifacts into the `artifacts/` directory and use `actions/upload-artifact`.

Why we include notebook execution in CI

- Prevents regressions in example notebooks
- Ensures demo code runs with the current package and environment
- Allows reviewers to inspect a reproducible HTML snapshot of the notebook outputs