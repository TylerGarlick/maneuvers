# Classification

Location: `src/maneuvers/classify.py`

The baseline classifier works at the *segment* level: it aggregates features over detected segments and trains a standard classifier (RandomForest or LogisticRegression) on these aggregated features.

Key functions & concepts

- `segment_aggregated_features(features_df, segment)`
  - Computes simple aggregated features for a segment: mean/std/max/min acceleration magnitude, mean gyro, energy, and length.

- `build_training_data_from_sequence(seq, features_df)`
  - Converts a `Sequence` and a features DataFrame into `X` (feature matrix) and `y` (labels). It uses ground-truth `seq.segments` to build positive examples and samples background windows labeled `'none'`.

- `train_classifier(X, y, model_type='rf', cv=5)`
  - Returns a fitted pipeline (`StandardScaler` + classifier), label encoder, and cross-validation scores.

- `save_model(obj, path)` / `load_model(path)`
  - Save and load trained models using `joblib`.

Practical tips

- For robust performance, use cross-validation and more data than the tiny synthetic examples provide.
- Add more features (time-domain statistics, spectral features, or learned features from a 1D-CNN) for better classification.

Example

```python
X, y = build_training_data_from_sequence(seq, feats)
res = train_classifier(X, y, model_type='rf', cv=3)
model_obj = {'model': res['model'], 'encoder': res['encoder']}
save_model(model_obj, 'model.joblib')
```