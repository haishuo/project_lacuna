"""
lacuna.visualization.confusion

Confusion matrix visualization.
"""

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: 2D numpy array of shape [n_classes, n_classes].
        class_names: Optional list of class names for axis labels.

    Returns:
        matplotlib Figure with confusion matrix heatmap.

    Raises:
        ValueError: If cm is not a 2D array.
        ValueError: If cm is not square.
        ValueError: If class_names length does not match cm dimensions.
    """
    if not isinstance(cm, np.ndarray):
        raise ValueError(f"cm must be a numpy ndarray, got {type(cm).__name__}")
    if cm.ndim != 2:
        raise ValueError(f"cm must be 2D, got {cm.ndim}D")
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"cm must be square, got shape {cm.shape}"
        )

    n_classes = cm.shape[0]
    if class_names is not None and len(class_names) != n_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) must match "
            f"cm dimensions ({n_classes})"
        )

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], "d") if cm.dtype.kind == "i" else format(cm[i, j], ".2f"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig
