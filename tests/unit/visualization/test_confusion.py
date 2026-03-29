"""Tests for lacuna.visualization.confusion."""

import pytest
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from lacuna.visualization.confusion import plot_confusion_matrix


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix."""

    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        cm = np.array([[10, 2, 1], [3, 15, 0], [0, 1, 12]])
        fig = plot_confusion_matrix(cm, class_names=["A", "B", "C"])
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_default_class_names(self):
        """Should work without class_names."""
        cm = np.array([[5, 1], [2, 8]])
        fig = plot_confusion_matrix(cm)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_single_class(self):
        """Should work with 1x1 matrix."""
        cm = np.array([[10]])
        fig = plot_confusion_matrix(cm, class_names=["Only"])
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_non_2d_raises(self):
        """Should raise ValueError for 1D array."""
        with pytest.raises(ValueError, match="must be 2D"):
            plot_confusion_matrix(np.array([1, 2, 3]))

    def test_non_square_raises(self):
        """Should raise ValueError for non-square matrix."""
        with pytest.raises(ValueError, match="must be square"):
            plot_confusion_matrix(np.array([[1, 2], [3, 4], [5, 6]]))

    def test_wrong_class_names_length(self):
        """Should raise ValueError for mismatched class_names."""
        cm = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="class_names length"):
            plot_confusion_matrix(cm, class_names=["A", "B", "C"])

    def test_not_ndarray_raises(self):
        """Should raise ValueError for non-ndarray."""
        with pytest.raises(ValueError, match="must be a numpy ndarray"):
            plot_confusion_matrix([[1, 2], [3, 4]])

    def test_float_matrix(self):
        """Should handle float matrices."""
        cm = np.array([[0.5, 0.3], [0.2, 0.8]])
        fig = plot_confusion_matrix(cm, class_names=["X", "Y"])
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_zero_matrix(self):
        """Should handle all-zero matrix."""
        cm = np.zeros((3, 3), dtype=int)
        fig = plot_confusion_matrix(cm)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)
