# Maneuver Detection Improvements

This document describes the improvements made to maneuver detection and classification in the `maneuvers` package.

## Overview

The enhancements focus on three key areas:
1. **Detection Methods**: Multiple algorithms for more robust maneuver identification
2. **Feature Extraction**: Richer signal features for better characterization
3. **Classification Features**: Enhanced features for improved maneuver type discrimination

All improvements maintain backward compatibility with existing code.

## Enhanced Detection Methods

### 1. Adaptive Thresholding (`method="adaptive"`)

**What it does**: Uses local statistics (mean and standard deviation) in a sliding window to compute adaptive thresholds, making detection more robust to varying signal characteristics.

**When to use**: When your flight data has varying baseline levels or noise characteristics across the sequence.

**Parameters**:
- `window_size`: Size of sliding window for local statistics (default: 50)
- `n_std`: Number of standard deviations above local mean (default: 2.0)
- `min_len`: Minimum segment length (default: 5)

**Example**:
```python
from maneuvers.detection import detect_segments
from maneuvers.preprocessing import compute_features_from_sequence

features = compute_features_from_sequence(seq)
segments = detect_segments(
    features, 
    method="adaptive",
    window_size=50,
    n_std=2.0,
    min_len=5
)
```

**Advantages**:
- Handles non-stationary signals better than fixed threshold
- Adapts to local noise levels
- Reduces false positives in noisy regions

### 2. Multi-Signal Fusion (`method="fusion"`)

**What it does**: Combines normalized acceleration, gyroscope, and jerk signals with configurable weights to create a composite detection signal.

**When to use**: When maneuvers have distinct signatures in multiple sensor channels (e.g., rolls show strong gyro response, climbs show strong acceleration).

**Parameters**:
- `accel_weight`: Weight for acceleration signal (default: 0.6)
- `gyro_weight`: Weight for gyroscope signal (default: 0.3)
- `jerk_weight`: Weight for jerk signal (default: 0.1)
- `threshold`: Detection threshold on composite signal (default: 0.5)
- `min_len`: Minimum segment length (default: 5)
- `merge_gap`: Maximum gap for merging nearby segments (default: 10)

**Example**:
```python
segments = detect_segments(
    features,
    method="fusion",
    accel_weight=0.5,
    gyro_weight=0.4,
    jerk_weight=0.1,
    threshold=0.4,
    merge_gap=10
)
```

**Advantages**:
- Leverages multiple sensor modalities
- Tunable to emphasize different maneuver characteristics
- Automatically merges fragmented detections

### 3. Variance-Based Change-Point Detection (`method="variance"`)

**What it does**: Identifies maneuvers by detecting regions where signal variance significantly increases, indicating dynamic motion.

**When to use**: When maneuvers are characterized by increased variability rather than absolute signal levels.

**Parameters**:
- `window_size`: Size of sliding window for variance computation (default: 20)
- `var_threshold`: Variance threshold multiplier (default: 2.0)
- `min_len`: Minimum segment length (default: 5)

**Example**:
```python
segments = detect_segments(
    features,
    method="variance",
    window_size=20,
    var_threshold=2.0,
    min_len=5
)
```

**Advantages**:
- Good for detecting onset of dynamic maneuvers
- Complements magnitude-based detection
- Less sensitive to calibration offsets

### 4. Enhanced Threshold Method with Segment Merging

The original threshold method now supports merging nearby segments:

**Example**:
```python
segments = detect_segments(
    features,
    method="threshold",
    threshold=0.4,
    min_len=5,
    merge_gap=15  # Merge segments within 15 samples
)
```

**When to use**: When a single maneuver may be split into multiple fragments due to signal fluctuations.

## Enhanced Feature Extraction

### New Preprocessing Features

The `compute_features_from_sequence()` function now computes:

1. **Jerk Magnitude** (`jerk_mag`): Rate of change of acceleration
   - Indicates sudden transitions and maneuver aggressiveness
   - Computed as derivative of acceleration magnitude

2. **Rotational Energy** (`rot_energy`): Sum of squared angular rates
   - Indicates turning intensity
   - Higher for rolls, spins, and coordinated turns

**Example**:
```python
from maneuvers.preprocessing import compute_features_from_sequence

features = compute_features_from_sequence(seq)
print(features.columns)
# Output: ['t', 'accel_mag', 'accel_smooth', 'gyro_mag', 'jerk_mag', 'rot_energy', ...]
```

## Enhanced Classification Features

The `segment_aggregated_features()` function now computes **25 features** (up from 9):

### Original Features (maintained for compatibility):
1. Mean acceleration magnitude
2. Standard deviation of acceleration
3. Max acceleration
4. Min acceleration
5. Mean gyroscope magnitude
6. Total energy (sum of squared accelerations)
7. Segment length
8. Dominant frequency (FFT)
9. FFT energy

### New Statistical Features:
10. Median acceleration
11. 75th percentile acceleration
12. 25th percentile acceleration
13. Skewness (distribution asymmetry)
14. Kurtosis (distribution peakedness)
15. Gyroscope standard deviation
16. Max gyroscope magnitude

### New Motion Features:
17. Mean jerk
18. Max jerk
19. Mean rotational energy
20. Max rotational energy

### New Spectral Feature:
21. Spectral entropy (frequency complexity)

### New Temporal Pattern Features:
22. Rise ratio (how quickly maneuver starts)
23. Fall ratio (how quickly maneuver ends)
24. Peak location (normalized position of maximum)

### New Cross-Signal Feature:
25. Acceleration-gyroscope cross-correlation

**Impact**: These features provide much richer characterization for classification, enabling:
- Better discrimination between maneuver types
- More robust models with complex patterns
- Improved generalization to real-world data

## Usage Guide

### Basic Usage with New Methods

```python
from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.detection import detect_segments
from maneuvers.eval import evaluate_detection

# Generate test data
seq = generate_synthetic_sequence(duration_s=20.0, fs=100, seed=0)
features = compute_features_from_sequence(seq)

# Try different detection methods
methods = {
    "threshold": {"threshold": 0.4, "min_len": 5, "merge_gap": 10},
    "adaptive": {"n_std": 2.0, "min_len": 5},
    "fusion": {"accel_weight": 0.6, "gyro_weight": 0.3, "jerk_weight": 0.1, "threshold": 0.4},
    "variance": {"var_threshold": 2.0, "min_len": 5}
}

for method_name, params in methods.items():
    segments = detect_segments(features, method=method_name, **params)
    
    # Extract ground truth for evaluation
    gt_segments = [(s, e) for s, e, _ in seq.segments]
    results = evaluate_detection(gt_segments, segments)
    
    print(f"{method_name}: precision={results['precision']:.2f}, recall={results['recall']:.2f}, f1={results['f1']:.2f}")
```

### Training with Enhanced Features

```python
from maneuvers.classify import build_training_data_from_sequence, train_classifier

# Build training data (now uses 25 features per segment)
X, y = build_training_data_from_sequence(seq, features)

# Train classifier
model_obj = train_classifier(X, y, model_type='rf', cv=5)

# The model automatically handles the enhanced feature set
```

## Performance Considerations

### Detection Method Selection Guide

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| Clean synthetic data | `threshold` | Simplest, fast, well-calibrated |
| Varying noise levels | `adaptive` | Adapts to local characteristics |
| Multi-modal maneuvers | `fusion` | Leverages all sensor channels |
| Highly dynamic flight | `variance` | Detects changes in motion patterns |
| Fragmented detections | Any with `merge_gap > 0` | Consolidates split segments |

### Computational Cost

- **threshold**: O(n) - fastest
- **adaptive**: O(n × window_size) - moderate
- **fusion**: O(n) - fast (after normalization)
- **variance**: O(n × window_size) - moderate

Where n = sequence length.

### Feature Computation Cost

- Enhanced features add minimal overhead (~20% increase)
- Dominated by FFT computation (present in original)
- Parallelizable across segments

## Backward Compatibility

All improvements are **fully backward compatible**:

1. Original `method="threshold"` behavior unchanged (when `merge_gap` not specified)
2. Original feature columns still present in same order
3. Existing models continue to work (trained on 9 features)
4. New models automatically use all 25 features

## Testing

Comprehensive test coverage ensures reliability:

- `tests/test_detection_improvements.py`: 7 tests for new detection methods
- `tests/test_classification_improvements.py`: 5 tests for enhanced features
- All existing tests pass without modification

Run tests:
```bash
pytest tests/test_detection_improvements.py -v
pytest tests/test_classification_improvements.py -v
```

## Future Enhancements

Potential areas for further improvement:

1. **Deep Learning**: 1D CNN or LSTM for end-to-end detection
2. **Ensemble Methods**: Combine multiple detection methods via voting
3. **Online Detection**: Streaming algorithms for real-time applications
4. **Uncertainty Quantification**: Confidence scores for detections
5. **Multi-Scale Analysis**: Wavelet-based features for hierarchical patterns

## References

- Original detection baseline: `docs/detection.md`
- Original preprocessing: `docs/preprocessing.md`
- Classification guide: `docs/classification.md`
