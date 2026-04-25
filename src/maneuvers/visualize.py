"""Small plotting helpers for classifier evaluation."""

from __future__ import annotations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
)
from typing import Sequence, List, Tuple
import numpy as np


def plot_confusion(
    true_labels: Sequence[str], pred_labels: Sequence[str], labels: list | None = None
):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax)
    plt.tight_layout()
    return fig


def plot_pr_curve_binary(y_true, y_score, pos_label=1):
    p, r, _ = precision_recall_curve(y_true == pos_label, y_score[:, pos_label])
    disp = PrecisionRecallDisplay(precision=p, recall=r)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax)
    plt.tight_layout()
    return fig


def plot_3d_segments(
    seq,
    segments: List[Tuple[int, int]],
    labels: List[str] = None,
    out_path: str = "flight_path_3d.png",
    interactive: bool = False,
):
    """Plot 3D flight path with maneuver segments marked.

    Since we have accel/gyro, plot integrated position (approximate) or just accel in 3D.
    For simplicity, plot accel x,y,z as 3D scatter with time as color, mark segments.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot accel as 3D points colored by time
    t_norm = (seq.timestamps - seq.timestamps.min()) / (
        seq.timestamps.max() - seq.timestamps.min()
    )
    ax.scatter(
        seq.accel[:, 0],
        seq.accel[:, 1],
        seq.accel[:, 2],
        c=t_norm,
        cmap="viridis",
        alpha=0.6,
        s=1,
    )
    ax.set_xlabel("Accel X")
    ax.set_ylabel("Accel Y")
    ax.set_zlabel("Accel Z")
    ax.set_title("3D Flight Path (Accel Space)")

    # Mark segments
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for i, (s, e) in enumerate(segments):
        label = labels[i] if labels and i < len(labels) else f"Segment {i}"
        ax.scatter(
            seq.accel[s:e, 0],
            seq.accel[s:e, 1],
            seq.accel[s:e, 2],
            color=colors[i % len(colors)],
            s=10,
            label=label,
        )
        # Mark start and end
        ax.scatter(
            seq.accel[s, 0],
            seq.accel[s, 1],
            seq.accel[s, 2],
            color="red",
            marker="^",
            s=50,
        )  # start
        ax.scatter(
            seq.accel[e - 1, 0],
            seq.accel[e - 1, 1],
            seq.accel[e - 1, 2],
            color="green",
            marker="v",
            s=50,
        )  # end

    ax.legend()
    plt.tight_layout()

    if interactive:
        # For interactive, could use plotly, but for now save PNG
        pass

    fig.savefig(out_path)
    plt.close(fig)
    return fig
