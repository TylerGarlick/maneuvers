"""Small plotting helpers for classifier evaluation."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
from typing import Sequence


def plot_confusion(true_labels: Sequence[str], pred_labels: Sequence[str], labels: list | None = None):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax)
    plt.tight_layout()
    return fig


def plot_pr_curve_binary(y_true, y_score, pos_label=1):
    p, r, _ = precision_recall_curve(y_true == pos_label, y_score[:, pos_label])
    disp = PrecisionRecallDisplay(precision=p, recall=r)
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax)
    plt.tight_layout()
    return fig
