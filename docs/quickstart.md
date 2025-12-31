# Quickstart

A short sequence of commands to run the baseline pipeline locally.

1. Create virtualenv and install dependencies (see `docs/installation.md`).

2. Run detection on synthetic demo data:

```bash
maneuvers detect-synthetic --duration 10 --fs 100 --threshold 0.4
```

3. Evaluate detection on a synthetic sequence:

```bash
maneuvers eval-synthetic --duration 10 --fs 100 --threshold 0.4
```

4. Train a classifier on synthetic data (with enhanced training using multiple sequences and maneuvers_small dataset):

```bash
maneuvers train-synthetic --out model.joblib --duration 10 --fs 100 --num-sequences 20
```

5. Use a saved model to label detected segments:

```bash
maneuvers detect-synthetic --duration 10 --fs 100 --threshold 0.4 --model model.joblib
```

6. Export detected maneuvers with labels and quality scores:

```bash
maneuvers export-detect --out maneuvers_export.json --model model.joblib
```

7. Generate a 3D plot of the flight path with maneuver markers:

```bash
maneuvers plot-3d --out flight_path_3d.png --model model.joblib
```

8. Run the demo notebook (requires Jupyter):

```bash
jupyter nbconvert --execute examples/notebooks/detection_classification_demo.ipynb --to html
```

If you want to quickly inspect one of the generated example files, see `examples/synthetic.csv` and `examples/synthetic_gt.json`. Generated outputs like `maneuvers_export.json` and `flight_path_3d.png` are saved in the `examples/` directory.
````
