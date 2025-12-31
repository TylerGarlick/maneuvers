# Classification

Location: `src/maneuvers/classify.py`

The baseline classifier works at the *segment* level: it aggregates features over detected segments and trains a standard classifier (RandomForest or LogisticRegression) on these aggregated features.

Key functions & concepts

- `segment_aggregated_features(features_df, segment)`
  - Computes aggregated features for a segment: mean/std/max/min acceleration magnitude, mean gyro, energy, length, dominant frequency, and FFT energy.

- `build_training_data_from_sequence(seq, features_df)`
  - Converts a `Sequence` and a features DataFrame into `X` (feature matrix) and `y` (labels). It uses ground-truth `seq.segments` to build positive examples and samples background windows labeled `'none'`.

- `train_classifier(X, y, model_type='rf', cv=5, param_grid=None)`
  - Returns a fitted pipeline (`StandardScaler` + classifier), label encoder, and cross-validation scores. Supports hyperparameter tuning via `param_grid` for GridSearchCV.

- `save_model(obj, path)` / `load_model(path)`
  - Save and load trained models using `joblib`. Models include metadata like training params and CV scores.

Practical tips

- For robust performance, use cross-validation, hyperparameter tuning, and combined training data (synthetic + maneuvers_small dataset).
- Add more features (time-domain statistics, spectral features, or learned features from a 1D-CNN) for better classification.

Example

```python
# Enhanced training with grid search
param_grid = {'clf__n_estimators': [50, 100], 'clf__max_depth': [None, 10]}
res = train_classifier(X, y, model_type='rf', cv=5, param_grid=param_grid)
model_obj = {'model': res['model'], 'encoder': res['encoder'], 'cv_scores': res['cv_scores']}
save_model(model_obj, 'model.joblib')
```

For CLI training with combined data:

```bash
maneuvers train-synthetic --out model.joblib --num-sequences 20 --use-maneuvers-small true
```
