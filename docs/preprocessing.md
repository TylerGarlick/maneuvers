# Preprocessing & Feature Extraction

Location: `src/maneuvers/preprocessing.py`

The baseline implements a small set of preprocessing helpers that are easy to understand and useful for detection. **New in v0.1.0: Enhanced features including jerk and rotational energy** - see [improvements.md](improvements.md) for details.

- `accel_magnitude(accel)` — L2 norm of the 3-axis accelerometer signal (per-sample)
- `moving_average(x, window=7)` — simple moving average smoothing
- `compute_jerk(accel, timestamps)` — **NEW**: Compute jerk (rate of change of acceleration)
- `compute_rotational_energy(gyro)` — **NEW**: Compute rotational energy from gyroscope
- `compute_features_from_sequence(seq)` — returns a pandas DataFrame with columns:
  - `t` — timestamps
  - `accel_mag` — raw acceleration magnitude
  - `accel_smooth` — smoothed magnitude (used by the threshold detector)
  - `gyro_mag` — gyroscope magnitude
  - `jerk_mag` — **NEW**: jerk magnitude (indicates sudden transitions)
  - `rot_energy` — **NEW**: rotational energy (indicates turning intensity)
  - `pos_mag`, `vel_mag`, `orient_mag` — optional, if available in sequence

Tips for experimentation

- Try different smoothing windows (larger windows reduce noise but blur short maneuvers).
- Add rolling statistics (mean, std), derivatives (jerk), or spectral features (FFT energy) for more robust detection and classification. **✓ Jerk now included**
- Keep feature shapes consistent between training and inference (same window lengths, same normalization).
- Use the new jerk and rotational energy features for better maneuver characterization.

Example usage

```python
from maneuvers.preprocessing import compute_features_from_sequence
feats = compute_features_from_sequence(seq)
print(feats.head())
# Output includes: t, accel_mag, accel_smooth, gyro_mag, jerk_mag, rot_energy
```

For comprehensive documentation on enhanced features, see [improvements.md](improvements.md).