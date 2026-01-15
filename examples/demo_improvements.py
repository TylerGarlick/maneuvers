#!/usr/bin/env python
"""Demonstration of improved maneuver detection methods.

This script compares different detection methods on synthetic flight data
and shows the enhanced features available for classification.
"""

import numpy as np
from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.detection import detect_segments
from maneuvers.eval import evaluate_detection
from maneuvers.classify import segment_aggregated_features


def compare_detection_methods():
    """Compare all detection methods on synthetic data."""
    print("=" * 70)
    print("Maneuver Detection Method Comparison")
    print("=" * 70)
    
    # Generate synthetic sequence with known maneuvers
    seq = generate_synthetic_sequence(duration_s=20.0, fs=100, seed=42)
    features = compute_features_from_sequence(seq)
    
    print(f"\nGenerated sequence: {len(seq.timestamps)} samples, {len(seq.segments)} maneuvers")
    print(f"Maneuvers: {[(s, e, label) for s, e, label in seq.segments]}")
    
    # Ground truth segments (without labels)
    gt_segments = [(s, e) for s, e, _ in seq.segments]
    
    print("\n" + "-" * 70)
    print("Detection Results:")
    print("-" * 70)
    
    # Define detection methods and their parameters
    methods = {
        "Threshold (baseline)": {
            "method": "threshold",
            "params": {"threshold": 0.4, "min_len": 5}
        },
        "Threshold + Merge": {
            "method": "threshold",
            "params": {"threshold": 0.4, "min_len": 5, "merge_gap": 20}
        },
        "Adaptive Threshold": {
            "method": "adaptive",
            "params": {"window_size": 50, "n_std": 1.8, "min_len": 5}
        },
        "Multi-Signal Fusion": {
            "method": "fusion",
            "params": {
                "accel_weight": 0.5,
                "gyro_weight": 0.4,
                "jerk_weight": 0.1,
                "threshold": 0.35,
                "min_len": 5,
                "merge_gap": 15
            }
        },
        "Variance-Based": {
            "method": "variance",
            "params": {"window_size": 25, "var_threshold": 1.5, "min_len": 5}
        }
    }
    
    results_table = []
    
    for method_name, config in methods.items():
        segments = detect_segments(
            features, 
            method=config["method"],
            **config["params"]
        )
        
        # Evaluate detection
        eval_results = evaluate_detection(gt_segments, segments)
        
        results_table.append({
            "Method": method_name,
            "Detected": len(segments),
            "TP": eval_results["tp"],
            "FP": eval_results["fp"],
            "FN": eval_results["fn"],
            "Precision": eval_results["precision"],
            "Recall": eval_results["recall"],
            "F1": eval_results["f1"]
        })
    
    # Print results table
    print(f"\n{'Method':<25} {'Det':<5} {'TP':<4} {'FP':<4} {'FN':<4} {'Prec':<6} {'Rec':<6} {'F1':<6}")
    print("-" * 70)
    for r in results_table:
        print(f"{r['Method']:<25} {r['Detected']:<5} {r['TP']:<4} {r['FP']:<4} {r['FN']:<4} "
              f"{r['Precision']:<6.3f} {r['Recall']:<6.3f} {r['F1']:<6.3f}")
    
    print("\n" + "=" * 70)


def show_enhanced_features():
    """Demonstrate enhanced feature extraction."""
    print("\n" + "=" * 70)
    print("Enhanced Feature Extraction")
    print("=" * 70)
    
    # Generate small sequence
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=0)
    features = compute_features_from_sequence(seq)
    
    print(f"\nAvailable features per time step: {len(features.columns)}")
    print(f"Feature names: {list(features.columns)}")
    
    print("\n" + "-" * 70)
    print("New features added:")
    print("-" * 70)
    
    new_features = ["jerk_mag", "rot_energy"]
    for feat_name in new_features:
        if feat_name in features.columns:
            feat_data = features[feat_name].values
            print(f"\n{feat_name}:")
            print(f"  Mean: {np.mean(feat_data):.4f}")
            print(f"  Std:  {np.std(feat_data):.4f}")
            print(f"  Max:  {np.max(feat_data):.4f}")
            print(f"  Min:  {np.min(feat_data):.4f}")
    
    print("\n" + "-" * 70)
    print("Segment-level features (for classification):")
    print("-" * 70)
    
    # Get a maneuver segment
    if seq.segments:
        s, e, label = seq.segments[0]
        segment_feats = segment_aggregated_features(features, (s, e))
        
        print(f"\nManeuver: {label} (samples {s}-{e})")
        print(f"Total classification features: {len(segment_feats)}")
        
        feature_names = [
            "mean_accel", "std_accel", "max_accel", "min_accel", "mean_gyro",
            "energy", "length", "dominant_freq", "fft_energy",
            "median_accel", "q75_accel", "q25_accel", "skewness", "kurtosis",
            "std_gyro", "max_gyro", "mean_jerk", "max_jerk",
            "mean_rot_energy", "max_rot_energy", "spectral_entropy",
            "rise_ratio", "fall_ratio", "peak_location", "cross_corr"
        ]
        
        print("\nKey features:")
        indices_to_show = [0, 2, 4, 6, 10, 13, 16, 17, 20, 23, 24]
        for idx in indices_to_show:
            if idx < len(segment_feats):
                print(f"  {feature_names[idx]:<20}: {segment_feats[idx]:.4f}")
    
    print("\n" + "=" * 70)


def demonstrate_parameter_tuning():
    """Show how to tune parameters for different scenarios."""
    print("\n" + "=" * 70)
    print("Parameter Tuning Examples")
    print("=" * 70)
    
    seq = generate_synthetic_sequence(duration_s=15.0, fs=100, seed=123)
    features = compute_features_from_sequence(seq)
    gt_segments = [(s, e) for s, e, _ in seq.segments]
    
    print("\nScenario 1: Minimize False Positives (High Precision)")
    print("-" * 70)
    
    configs = [
        ("Conservative threshold", {"method": "threshold", "threshold": 0.6, "min_len": 10}),
        ("Strict adaptive", {"method": "adaptive", "n_std": 2.5, "min_len": 10}),
        ("Strict fusion", {"method": "fusion", "threshold": 0.6, "min_len": 10}),
    ]
    
    for name, config in configs:
        method = config.pop("method")
        segments = detect_segments(features, method=method, **config)
        eval_res = evaluate_detection(gt_segments, segments)
        print(f"{name:<25}: Prec={eval_res['precision']:.3f}, Rec={eval_res['recall']:.3f}, Det={len(segments)}")
    
    print("\nScenario 2: Maximize Recall (Catch All Maneuvers)")
    print("-" * 70)
    
    configs = [
        ("Loose threshold", {"method": "threshold", "threshold": 0.3, "min_len": 3, "merge_gap": 20}),
        ("Permissive adaptive", {"method": "adaptive", "n_std": 1.2, "min_len": 3}),
        ("Sensitive fusion", {"method": "fusion", "threshold": 0.3, "min_len": 3, "merge_gap": 20}),
    ]
    
    for name, config in configs:
        method = config.pop("method")
        segments = detect_segments(features, method=method, **config)
        eval_res = evaluate_detection(gt_segments, segments)
        print(f"{name:<25}: Prec={eval_res['precision']:.3f}, Rec={eval_res['recall']:.3f}, Det={len(segments)}")
    
    print("\nScenario 3: Balanced Performance")
    print("-" * 70)
    
    configs = [
        ("Balanced threshold", {"method": "threshold", "threshold": 0.4, "min_len": 5, "merge_gap": 10}),
        ("Balanced adaptive", {"method": "adaptive", "n_std": 2.0, "min_len": 5}),
        ("Balanced fusion", {"method": "fusion", "threshold": 0.4, "min_len": 5}),
    ]
    
    for name, config in configs:
        method = config.pop("method")
        segments = detect_segments(features, method=method, **config)
        eval_res = evaluate_detection(gt_segments, segments)
        print(f"{name:<25}: Prec={eval_res['precision']:.3f}, Rec={eval_res['recall']:.3f}, F1={eval_res['f1']:.3f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MANEUVER DETECTION IMPROVEMENTS DEMONSTRATION")
    print("=" * 70)
    
    # Run demonstrations
    compare_detection_methods()
    show_enhanced_features()
    demonstrate_parameter_tuning()
    
    print("\n" + "=" * 70)
    print("For more details, see: docs/improvements.md")
    print("=" * 70)
    print()
