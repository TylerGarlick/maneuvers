from maneuvers.eval import segment_iou, evaluate_detection


def test_segment_iou():
    a = (10, 20)
    b = (15, 25)
    assert abs(segment_iou(a, b) - 5 / 15) < 1e-8


def test_evaluate_detection_counts():
    gt = [(0, 10), (20, 30)]
    preds = [(0, 10), (21, 29), (40, 50)]
    res = evaluate_detection(gt, preds, iou_thresh=0.5)
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 0
