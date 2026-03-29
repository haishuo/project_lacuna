"""Tests for lacuna.metrics.calibration."""

import pytest
import torch

from lacuna.metrics.calibration import compute_ece


class TestComputeEce:
    """Tests for compute_ece."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        # All predictions are [1, 0, 0] and true class is 0
        p_class = torch.tensor([[1.0, 0.0, 0.0]] * 10)
        true_class = torch.tensor([0] * 10)
        result = compute_ece(p_class, true_class)
        assert result["ece"] == pytest.approx(0.0, abs=0.01)
        assert "bins" in result
        assert result["n_bins"] == 10

    def test_empty_input(self):
        """Empty tensors should return ece=0 and empty bins."""
        p_class = torch.zeros(0, 3)
        true_class = torch.zeros(0, dtype=torch.long)
        result = compute_ece(p_class, true_class)
        assert result["ece"] == 0.0
        assert result["bins"] == []

    def test_all_incorrect_high_confidence(self):
        """All wrong with high confidence should have high ECE."""
        p_class = torch.tensor([[0.95, 0.03, 0.02]] * 20)
        true_class = torch.tensor([1] * 20)
        result = compute_ece(p_class, true_class)
        # Confidence ~0.95, accuracy 0 -> gap ~0.95
        assert result["ece"] > 0.5

    def test_custom_n_bins(self):
        """Should respect custom n_bins."""
        p_class = torch.tensor([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1]])
        true_class = torch.tensor([0, 0])
        result = compute_ece(p_class, true_class, n_bins=5)
        assert result["n_bins"] == 5
        assert len(result["bins"]) == 5

    def test_bins_sum_to_total(self):
        """All samples should appear in exactly one bin."""
        p_class = torch.rand(50, 3)
        p_class = p_class / p_class.sum(dim=-1, keepdim=True)
        true_class = torch.randint(0, 3, (50,))
        result = compute_ece(p_class, true_class)
        total = sum(b["count"] for b in result["bins"])
        assert total == 50

    def test_invalid_p_class_dim(self):
        """Should raise ValueError for non-2D p_class."""
        with pytest.raises(ValueError, match="p_class must be 2D"):
            compute_ece(torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0]))

    def test_invalid_true_class_dim(self):
        """Should raise ValueError for non-1D true_class."""
        with pytest.raises(ValueError, match="true_class must be 1D"):
            compute_ece(torch.tensor([[0.5, 0.3, 0.2]]), torch.tensor([[0]]))

    def test_invalid_n_bins(self):
        """Should raise ValueError for n_bins < 1."""
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_ece(torch.tensor([[0.5, 0.3, 0.2]]), torch.tensor([0]), n_bins=0)

    def test_single_sample(self):
        """Single sample should work without errors."""
        p_class = torch.tensor([[0.7, 0.2, 0.1]])
        true_class = torch.tensor([0])
        result = compute_ece(p_class, true_class)
        assert isinstance(result["ece"], float)
