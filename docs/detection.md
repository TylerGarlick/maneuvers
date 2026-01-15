# Detection (Baseline)

Location: `src/maneuvers/detection.py`

The baseline detector in this repository is intentionally simple so you can understand and extend it quickly. **New in v0.1.0: Multiple detection methods are now available for improved accuracy** - see [improvements.md](improvements.md) for details.

Key functions

- `threshold_segment(signal, thr, min_len=5)`
  - Finds contiguous regions where `signal > thr` and returns segments as `(start_idx, end_idx)` (end exclusive).
  - `min_len` filters short noisy detections.

- `detect_segments(features, method='threshold', **kwargs)`
  - Dispatch function supporting multiple detection methods:
    - `method='threshold'`: Uses `features['accel_smooth']` with fixed threshold (original baseline)
    - `method='adaptive'`: Adaptive thresholding based on local statistics
    - `method='fusion'`: Multi-signal fusion combining accel, gyro, and jerk
    - `method='variance'`: Variance-based change-point detection
  
- `merge_nearby_segments(segments, max_gap=10)`
  - Merges segments that are close together (likely part of same maneuver)

How to tune

- Increase `threshold` to reduce false positives (miss weak maneuvers).
- Increase `min_len` to ignore short spikes or sensor glitches.
- Use `merge_gap` parameter to consolidate fragmented detections.
- Try different detection methods for better results on your data.

Ideas for improvement

- Add change-point detection (e.g., using the `ruptures` package) to find segments automatically. **✓ Now available as `method='variance'`**
- Use a sliding-window classifier to detect maneuver start/stop with learned features.
- Use an HMM or run-length model for smoother segmentation that takes temporal context into account.

Example

```python
# Basic threshold detection
preds = detect_segments(feats, method='threshold', threshold=0.4, min_len=8)

# Threshold with merging
preds = detect_segments(feats, method='threshold', threshold=0.4, min_len=5, merge_gap=15)

# Adaptive thresholding
preds = detect_segments(feats, method='adaptive', window_size=50, n_std=2.0, min_len=5)

# Multi-signal fusion
preds = detect_segments(feats, method='fusion', 
                       accel_weight=0.6, gyro_weight=0.3, jerk_weight=0.1,
                       threshold=0.4, merge_gap=10)

# Variance-based change-point detection
preds = detect_segments(feats, method='variance', window_size=20, var_threshold=2.0, min_len=5)
```

For comprehensive documentation on new detection methods, see [improvements.md](improvements.md).