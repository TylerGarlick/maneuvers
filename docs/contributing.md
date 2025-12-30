# Contributing

Thanks for wanting to contribute! This page explains how to run tests, add features, and prepare a clean PR.

Development workflow

1. Fork / branch from `main` and create a topic branch for your change.
2. Run tests locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

3. Format and lint before opening a PR:

```bash
black .
flake8 .
```

4. Add tests for new functionality in `tests/`. Prefer small focused unit tests and an integration test if the change spans modules.

Guidelines

- Keep functions small and well-documented.
- Add docstrings to APIs you create and update the docs under `docs/`.
- If you add data or notebooks, put them under `examples/` and add a test that the notebook executes if appropriate.

Submitting a PR

- Open a PR against `main`, include a short description, and reference any relevant issues.
- CI will run tests and checks; address comments and push fixes to the branch.

If you'd like help designing a feature or tests, open an issue or ask for review in the PR — happy to help.