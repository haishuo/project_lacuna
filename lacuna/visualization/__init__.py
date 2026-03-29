"""
lacuna.visualization

Visualization utilities for training metrics, calibration, and confusion matrices.
"""

from lacuna.visualization.curves import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_calibration_curve,
)
from lacuna.visualization.confusion import plot_confusion_matrix

__all__ = [
    "plot_loss_curves",
    "plot_accuracy_curves",
    "plot_calibration_curve",
    "plot_confusion_matrix",
]
