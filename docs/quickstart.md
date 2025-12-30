# Quickstart

A short sequence of commands to run the baseline pipeline locally.

1. Create virtualenv and install dependencies (see `docs/installation.md`).

2. Run detection on synthetic demo data:

```bash
python -m maneuvers.cli detect-synthetic --duration 10 --fs 100 --threshold 0.4
```

3. Evaluate detection on a synthetic sequence:

```bash
python -m maneuvers.cli eval-synthetic --duration 10 --fs 100 --threshold 0.4
```

4. Train a simple classifier on synthetic data and save the model:

```bash
python -m maneuvers.cli train-synthetic --out model.joblib --duration 10 --fs 100
```

5. Use a saved model to label detected segments:

```bash
python -m maneuvers.cli detect-synthetic --duration 10 --fs 100 --threshold 0.4 --model model.joblib
```

6. Run the demo notebook (requires Jupyter):

```bash
jupyter nbconvert --execute examples/notebooks/detection_classification_demo.ipynb --to html
```

If you want to quickly inspect one of the generated example files, see `examples/synthetic.csv` and `examples/synthetic_gt.json`.