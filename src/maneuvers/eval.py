"""Evaluation utilities for segment detection."""

from __future__ import annotations
from typing import List, Tuple


def segment_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """IoU between two integer-indexed segments. end indices are exclusive.

    IoU = intersection_length / union_length, where union_length = len(a) + len(b) - intersection_length.
    """
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    len_a = max(0, e1 - s1)
    len_b = max(0, e2 - s2)
    union = len_a + len_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_detection(
    gt_segments: List[Tuple[int, int]],
    pred_segments: List[Tuple[int, int]],
    iou_thresh: float = 0.5,
) -> dict:
    """Simple evaluation: counts TP, FP, FN using greedy matching by IoU."""
    matches = []
    used_pred = set()

    for gi, g in enumerate(gt_segments):
        best_i = -1
        best_iou = 0.0
        for pi, p in enumerate(pred_segments):
            if pi in used_pred:
                continue
            iou = segment_iou(g, p)
            if iou > best_iou:
                best_iou = iou
                best_i = pi
        if best_i >= 0 and best_iou >= iou_thresh:
            matches.append((gi, best_i, best_iou))
            used_pred.add(best_i)

    tp = len(matches)
    fp = len(pred_segments) - len(used_pred)
    fn = len(gt_segments) - len(matches)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
