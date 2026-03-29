"""Tests for lacuna.metrics.classification."""

import pytest
import torch

from lacuna.metrics.classification import (
    compute_per_generator_accuracy,
    compute_selective_accuracy,
)


class TestComputePerGeneratorAccuracy:
    """Tests for compute_per_generator_accuracy."""

    def test_perfect_accuracy(self):
        """All correct predictions should give 1.0 accuracy."""
        preds = torch.tensor([0, 1, 2, 0, 1])
        true_class = torch.tensor([0, 1, 2, 0, 1])
        gen_ids = torch.tensor([0, 0, 1, 1, 1])
        result = compute_per_generator_accuracy(preds, true_class, gen_ids)
        assert result["0"]["accuracy"] == 1.0
        assert result["1"]["accuracy"] == 1.0

    def test_zero_accuracy(self):
        """All wrong predictions should give 0.0 accuracy."""
        preds = torch.tensor([1, 1, 1])
        true_class = torch.tensor([0, 0, 0])
        gen_ids = torch.tensor([0, 0, 0])
        result = compute_per_generator_accuracy(preds, true_class, gen_ids)
        assert result["0"]["accuracy"] == 0.0

    def test_class_name_lookup(self):
        """Should map class indices to MCAR/MAR/MNAR."""
        preds = torch.tensor([0, 1, 2])
        true_class = torch.tensor([0, 1, 2])
        gen_ids = torch.tensor([0, 1, 2])
        result = compute_per_generator_accuracy(preds, true_class, gen_ids)
        assert result["0"]["class"] == "MCAR"
        assert result["1"]["class"] == "MAR"
        assert result["2"]["class"] == "MNAR"

    def test_count_per_generator(self):
        """Counts should match number of samples per generator."""
        preds = torch.tensor([0, 0, 1, 1, 1])
        true_class = torch.tensor([0, 0, 1, 1, 1])
        gen_ids = torch.tensor([0, 0, 1, 1, 1])
        result = compute_per_generator_accuracy(preds, true_class, gen_ids)
        assert result["0"]["count"] == 2
        assert result["1"]["count"] == 3

    def test_invalid_preds_dim(self):
        """Should raise ValueError for non-1D preds."""
        with pytest.raises(ValueError, match="preds must be 1D"):
            compute_per_generator_accuracy(
                torch.tensor([[0]]), torch.tensor([0]), torch.tensor([0])
            )

    def test_mismatched_lengths(self):
        """Should raise ValueError for mismatched tensor lengths."""
        with pytest.raises(ValueError, match="Tensor lengths must match"):
            compute_per_generator_accuracy(
                torch.tensor([0, 1]), torch.tensor([0]), torch.tensor([0])
            )

    def test_single_sample(self):
        """Single sample should work."""
        preds = torch.tensor([0])
        true_class = torch.tensor([0])
        gen_ids = torch.tensor([5])
        result = compute_per_generator_accuracy(preds, true_class, gen_ids)
        assert "5" in result
        assert result["5"]["accuracy"] == 1.0


class TestComputeSelectiveAccuracy:
    """Tests for compute_selective_accuracy."""

    def test_all_high_confidence_correct(self):
        """High confidence correct predictions should have high accuracy."""
        p_class = torch.tensor([[0.95, 0.03, 0.02]] * 10)
        true_class = torch.tensor([0] * 10)
        result = compute_selective_accuracy(p_class, true_class)
        assert "thresholds" in result
        # At threshold 0.90, all 10 samples pass and all are correct
        row_90 = next(r for r in result["thresholds"] if r["threshold"] == 0.90)
        assert row_90["accuracy"] == 1.0
        assert row_90["coverage"] == 1.0

    def test_empty_input(self):
        """Empty input should return error dict."""
        p_class = torch.zeros(0, 3)
        true_class = torch.zeros(0, dtype=torch.long)
        result = compute_selective_accuracy(p_class, true_class)
        assert "error" in result

    def test_custom_thresholds(self):
        """Should respect custom thresholds."""
        p_class = torch.tensor([[0.8, 0.1, 0.1]])
        true_class = torch.tensor([0])
        result = compute_selective_accuracy(p_class, true_class, thresholds=[0.5, 0.9])
        assert len(result["thresholds"]) == 2

    def test_coverage_decreases_with_threshold(self):
        """Coverage should decrease as threshold increases."""
        torch.manual_seed(42)
        p_class = torch.rand(100, 3)
        p_class = p_class / p_class.sum(dim=-1, keepdim=True)
        true_class = torch.randint(0, 3, (100,))
        result = compute_selective_accuracy(p_class, true_class)
        coverages = [r["coverage"] for r in result["thresholds"]]
        # Coverage should be non-increasing
        for i in range(len(coverages) - 1):
            assert coverages[i] >= coverages[i + 1]

    def test_invalid_p_class_dim(self):
        """Should raise ValueError for non-2D p_class."""
        with pytest.raises(ValueError, match="p_class must be 2D"):
            compute_selective_accuracy(torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0]))

    def test_acc_thresholds_none_when_not_reached(self):
        """acc_90/95 should be None if never reached."""
        # Very low confidence, many wrong
        p_class = torch.tensor([[0.34, 0.33, 0.33]] * 10)
        true_class = torch.tensor([1] * 10)
        result = compute_selective_accuracy(p_class, true_class)
        # With uniform predictions, accuracy should be 0 everywhere
        assert result["acc_95_threshold"] is None
