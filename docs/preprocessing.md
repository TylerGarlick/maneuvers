# Preprocessing & Feature Extraction

Location: `src/maneuvers/preprocessing.py`

The baseline implements a small set of preprocessing helpers that are easy to understand and useful for detection:

- `accel_magnitude(accel)` — L2 norm of the 3-axis accelerometer signal (per-sample)
- `moving_average(x, window=7)` — simple moving average smoothing
- `compute_features_from_sequence(seq)` — returns a pandas DataFrame with columns:
  - `t` — timestamps
  - `accel_mag` — raw acceleration magnitude
  - `accel_smooth` — smoothed magnitude (used by the threshold detector)
  - `gyro_mag` — gyroscope magnitude

Tips for experimentation

- Try different smoothing windows (larger windows reduce noise but blur short maneuvers).
- Add rolling statistics (mean, std), derivatives (jerk), or spectral features (FFT energy) for more robust detection and classification.
- Keep feature shapes consistent between training and inference (same window lengths, same normalization).

Example usage

```python
from maneuvers.preprocessing import compute_features_from_sequence
feats = compute_features_from_sequence(seq)
print(feats.head())
```