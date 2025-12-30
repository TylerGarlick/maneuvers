# Notebook Demo

File: `examples/notebooks/detection_classification_demo.ipynb`

This Jupyter Notebook walks through a complete example: generate synthetic data, compute features, detect segments, train a classifier, and evaluate results. It is intended for beginners and helps you try changes interactively.

Notebook sections

1. Install & setup — confirms `maneuvers` can be imported
2. Generate sequences — shows generating and saving synthetic examples
3. Visualize — plots signals and ground-truth segments
4. Detect — runs threshold detection and visualizes predictions
5. Classify — trains a small classifier on aggregated segment features
6. Evaluate — computes IoU/precision/recall/F1 and prints classification reports
7. Run pipeline via CLI — demonstrates the `maneuvers` CLI from a notebook
8. CI snippet — shows the GitHub Actions steps that execute the notebook in CI

Running the notebook from your machine

```bash
# execute and export to HTML
jupyter nbconvert --to html --execute examples/notebooks/detection_classification_demo.ipynb --output demo.html
```

CI note

- The CI workflow runs `nbval` and `nbconvert` to ensure the notebook remains executable. See `.github/workflows/ci.yml` for details.