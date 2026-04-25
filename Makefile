
all: install test lint notebooks clean-artifacts

install:
	@echo "Installing dependencies..."
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:
	@echo "Running tests..."
	.venv/bin/pytest tests

lint:
	@echo "Running lint checks..."
	.venv/bin/black --check .
	.venv/bin/flake8 .

notebooks:
	@echo "Running notebooks..."
	.venv/bin/pip install jupyter nbconvert nbval
	.venv/bin/pytest --nbval-lax examples/notebooks/detection_classification_demo.ipynb

clean-artifacts:
	@echo "Cleaning artifacts..."
	rm -rf artifacts/

.PHONY: all install test lint notebooks clean-artifacts
