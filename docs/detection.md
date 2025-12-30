# Detection (Baseline)

Location: `src/maneuvers/detection.py`

The baseline detector in this repository is intentionally simple so you can understand and extend it quickly.

Key functions

- `threshold_segment(signal, thr, min_len=5)`
  - Finds contiguous regions where `signal > thr` and returns segments as `(start_idx, end_idx)` (end exclusive).
  - `min_len` filters short noisy detections.

- `detect_segments(features, method='threshold', **kwargs)`
  - Dispatch function: for `method='threshold'` it uses `features['accel_smooth']` and the `threshold` and `min_len` kwargs.

How to tune

- Increase `threshold` to reduce false positives (miss weak maneuvers).
- Increase `min_len` to ignore short spikes or sensor glitches.

Ideas for improvement

- Add change-point detection (e.g., using the `ruptures` package) to find segments automatically.
- Use a sliding-window classifier to detect maneuver start/stop with learned features.
- Use an HMM or run-length model for smoother segmentation that takes temporal context into account.

Example

```python
preds = detect_segments(feats, method='threshold', threshold=0.4, min_len=8)
# preds is a list of (start_idx, end_idx)
```