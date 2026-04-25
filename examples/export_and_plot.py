"""Example script: Generate synthetic data, detect maneuvers, label, score, export, and plot."""

from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.detection import detect_segments
from maneuvers.classify import load_model, predict_segment_labels
from maneuvers.export import export_segments, score_segment
from maneuvers.visualize import plot_3d_segments


def main():
    # Generate data
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100)
    feats = compute_features_from_sequence(seq)

    # Detect
    preds = detect_segments(feats, method="threshold", threshold=0.5, min_len=5)

    # Label and score (if model available)
    labels = None
    scores = None
    try:
        mobj = load_model("model.joblib")
        labels = predict_segment_labels(mobj, feats, preds)
        scores = [score_segment(seq, seg, mobj) for seg in preds]
    except:
        print("No model found, skipping labeling/scoring")

    # Export
    export_segments(seq, preds, labels, scores, "examples/example_export.json")
    print(f"Exported {len(preds)} segments")

    # Plot
    plot_3d_segments(seq, preds, labels, "examples/example_3d.png")
    print("Saved 3D plot")


if __name__ == "__main__":
    main()
