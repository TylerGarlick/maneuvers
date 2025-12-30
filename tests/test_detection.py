from maneuvers.data.loader import generate_synthetic_sequence
from maneuvers.preprocessing import compute_features_from_sequence
from maneuvers.detection import detect_segments
from maneuvers.eval import evaluate_detection


def test_threshold_detect_finds_maneuvers():
    seq = generate_synthetic_sequence(duration_s=10.0, fs=100, seed=2)
    feats = compute_features_from_sequence(seq)
    preds = detect_segments(feats, method="threshold", threshold=0.3, min_len=5)
    gt = [(s, e) for s, e, _ in seq.segments]
    res = evaluate_detection(gt, preds, iou_thresh=0.2)
    # should detect at least one true positive
    assert res["tp"] >= 1
    # predictions should be non-empty
    assert len(preds) >= 1
