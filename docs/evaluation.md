# Evaluation

Location: `src/maneuvers/eval.py`

This repo uses simple, explainable metrics to measure detection and classification quality:

- `segment_iou(a, b)` — Intersection over Union (IoU) for two integer-indexed segments. IoU is:

  ```text
  IoU(P, T) = |P ∩ T| / |P ∪ T|
  ```

- `evaluate_detection(gt_segments, pred_segments, iou_thresh=0.5)`
  - Greedy matching by IoU with a threshold to count True Positives (TP), False Positives (FP), and False Negatives (FN).
  - Returns `precision`, `recall`, and `f1` computed from TP/FP/FN.

How to interpret

- A higher IoU threshold (e.g., 0.5) requires better localization for a predicted segment to be counted as a TP.
- Precision is the proportion of predicted segments that match a ground-truth segment.
- Recall is the proportion of ground-truth segments that were detected.
- F1 balances precision and recall: `F1 = 2 * precision * recall / (precision + recall)`.

Extensions

- Use per-class metrics after matching to assess classification performance.
- Consider evaluating at multiple IoU thresholds (like object detection benchmarks) to understand localization sensitivity.