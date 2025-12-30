import pytest
from maneuvers.eval import segment_iou


def test_iou_overlap():
    a = (0, 10)
    b = (5, 15)
    # intersection = 5, union = 15 -> iou = 5/15
    assert pytest.approx(segment_iou(a, b), rel=1e-6) == 5.0 / 15.0


def test_iou_contained():
    a = (0, 10)
    b = (2, 8)
    # intersection 6, union = 10 -> 6/10
    assert pytest.approx(segment_iou(a, b), rel=1e-6) == 6.0 / 10.0


def test_iou_adjacent():
    a = (0, 5)
    b = (5, 10)
    # intersection 0, union = 10 -> 0.0
    assert segment_iou(a, b) == 0.0


def test_iou_disjoint():
    a = (0, 3)
    b = (5, 8)
    # intersection 0, union = 6 -> 0.0
    assert segment_iou(a, b) == 0.0
