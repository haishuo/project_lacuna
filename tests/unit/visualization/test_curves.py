"""Tests for lacuna.visualization.curves."""

import json
import pytest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from lacuna.visualization.curves import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_calibration_curve,
)


def _write_jsonl(path: Path, entries: list) -> Path:
    """Helper to write JSONL test data."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


class TestPlotLossCurves:
    """Tests for plot_loss_curves."""

    def test_returns_figure(self, tmp_path):
        """Should return a matplotlib Figure."""
        path = _write_jsonl(tmp_path / "metrics.jsonl", [
            {"type": "epoch", "epoch": 0, "loss": 1.0},
            {"type": "epoch", "epoch": 1, "loss": 0.5},
            {"type": "validation", "epoch": 0, "val_loss": 0.9},
            {"type": "validation", "epoch": 1, "val_loss": 0.4},
        ])
        fig = plot_loss_curves(path)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_train_only(self, tmp_path):
        """Should work with only train loss data."""
        path = _write_jsonl(tmp_path / "metrics.jsonl", [
            {"type": "epoch", "epoch": 0, "loss": 1.0},
        ])
        fig = plot_loss_curves(path)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_val_only(self, tmp_path):
        """Should work with only validation loss data."""
        path = _write_jsonl(tmp_path / "metrics.jsonl", [
            {"type": "validation", "epoch": 0, "val_loss": 0.9},
        ])
        fig = plot_loss_curves(path)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            plot_loss_curves(Path("/nonexistent/path.jsonl"))

    def test_empty_file(self, tmp_path):
        """Should raise ValueError for empty file."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No entries"):
            plot_loss_curves(path)

    def test_no_loss_data(self, tmp_path):
        """Should raise ValueError if no loss entries exist."""
        path = _write_jsonl(tmp_path / "metrics.jsonl", [
            {"type": "step", "lr": 0.01},
        ])
        with pytest.raises(ValueError, match="No loss data"):
            plot_loss_curves(path)


class TestPlotAccuracyCurves:
    """Tests for plot_accuracy_curves."""

    def test_returns_figure(self, tmp_path):
        """Should return a matplotlib Figure."""
        path = _write_jsonl(tmp_path / "metrics.jsonl", [
            {"type": "epoch", "epoch": 0, "accuracy": 0.5},
            {"type": "validation", "epoch": 0, "val_acc": 0.6},
        ])
        fig = plot_accuracy_curves(path)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_file_not_found(self):
        """Should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            plot_accuracy_curves(Path("/nonexistent/path.jsonl"))

    def test_no_accuracy_data(self, tmp_path):
        """Should raise ValueError if no accuracy entries."""
        path = _write_jsonl(tmp_path / "metrics.jsonl", [
            {"type": "step", "loss": 0.5},
        ])
        with pytest.raises(ValueError, match="No accuracy data"):
            plot_accuracy_curves(path)


class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve."""

    def test_returns_figure(self):
        """Should return a matplotlib Figure."""
        ece_data = {
            "ece": 0.05,
            "bins": [
                {"range": "0.00-0.10", "count": 5, "mean_confidence": 0.05, "accuracy": 0.0, "gap": 0.05},
                {"range": "0.90-1.00", "count": 10, "mean_confidence": 0.95, "accuracy": 0.9, "gap": 0.05},
            ],
        }
        fig = plot_calibration_curve(ece_data)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_missing_bins_key(self):
        """Should raise ValueError if bins key missing."""
        with pytest.raises(ValueError, match="must contain 'bins'"):
            plot_calibration_curve({"ece": 0.0})

    def test_missing_ece_key(self):
        """Should raise ValueError if ece key missing."""
        with pytest.raises(ValueError, match="must contain 'ece'"):
            plot_calibration_curve({"bins": []})

    def test_empty_bins(self):
        """Should raise ValueError if bins list is empty."""
        with pytest.raises(ValueError, match="no bins"):
            plot_calibration_curve({"ece": 0.0, "bins": []})

    def test_not_dict(self):
        """Should raise ValueError if not a dict."""
        with pytest.raises(ValueError, match="must be a dict"):
            plot_calibration_curve("not a dict")

    def test_all_empty_bins(self):
        """Should handle bins where all have count=0."""
        ece_data = {
            "ece": 0.0,
            "bins": [
                {"range": "0.00-0.50", "count": 0, "mean_confidence": 0.0, "accuracy": 0.0, "gap": 0.0},
                {"range": "0.50-1.00", "count": 0, "mean_confidence": 0.0, "accuracy": 0.0, "gap": 0.0},
            ],
        }
        # This should not raise -- the bins exist but all are empty
        # The function requires bins to be non-empty list (they are), just count=0
        fig = plot_calibration_curve(ece_data)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)
