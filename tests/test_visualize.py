"""Tests for visualization functions."""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
from maneuvers.visualize import plot_confusion, plot_pr_curve_binary


def test_plot_confusion():
    """Test confusion matrix plotting."""
    true_labels = ['A', 'B', 'A', 'B', 'A']
    pred_labels = ['A', 'B', 'B', 'B', 'A']

    fig = plot_confusion(true_labels, pred_labels)
    assert fig is not None
    plt.close(fig)


def test_plot_confusion_with_labels():
    """Test confusion matrix plotting with custom labels."""
    true_labels = ['A', 'B', 'A', 'B', 'A']
    pred_labels = ['A', 'B', 'B', 'B', 'A']
    labels = ['A', 'B']

    fig = plot_confusion(true_labels, pred_labels, labels=labels)
    assert fig is not None
    plt.close(fig)


def test_plot_pr_curve_binary():
    """Test precision-recall curve plotting."""
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_score = np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2], [0.65, 0.35], [0.9, 0.1]])

    fig = plot_pr_curve_binary(y_true, y_score, pos_label=1)
    assert fig is not None
    plt.close(fig)
