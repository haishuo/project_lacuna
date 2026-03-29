"""Tests for lacuna.metrics.uncertainty."""

import pytest
import torch
import numpy as np

from lacuna.metrics.uncertainty import (
    compute_entropy_stats,
    compute_confidence_analysis,
    compute_probability_distributions,
)


class TestComputeEntropyStats:
    """Tests for compute_entropy_stats."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have entropy near log(3)."""
        p_class = torch.tensor([[1/3, 1/3, 1/3]] * 10)
        true_class = torch.tensor([0] * 10)
        result = compute_entropy_stats(p_class, true_class)
        max_ent = np.log(3.0)
        assert result["overall"]["mean"] == pytest.approx(max_ent, abs=0.01)
        assert result["overall"]["mean_normalized"] == pytest.approx(1.0, abs=0.01)

    def test_confident_prediction_low_entropy(self):
        """Confident predictions should have low entropy."""
        p_class = torch.tensor([[0.99, 0.005, 0.005]] * 10)
        true_class = torch.tensor([0] * 10)
        result = compute_entropy_stats(p_class, true_class)
        assert result["overall"]["mean"] < 0.1

    def test_per_class_breakdown(self):
        """Should have per-class stats for each class."""
        p_class = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        true_class = torch.tensor([0, 1, 2])
        result = compute_entropy_stats(p_class, true_class)
        assert "mcar" in result["per_class"]
        assert "mar" in result["per_class"]
        assert "mnar" in result["per_class"]
        for cls in ["mcar", "mar", "mnar"]:
            assert result["per_class"][cls]["count"] == 1

    def test_missing_class_returns_zeros(self):
        """Class with no samples should return zero stats."""
        p_class = torch.tensor([[0.9, 0.05, 0.05]] * 5)
        true_class = torch.tensor([0] * 5)
        result = compute_entropy_stats(p_class, true_class)
        assert result["per_class"]["mar"]["count"] == 0
        assert result["per_class"]["mar"]["mean"] == 0.0

    def test_invalid_p_class_dim(self):
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="p_class must be 2D"):
            compute_entropy_stats(torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0]))

    def test_single_sample(self):
        """Single sample should work without division errors."""
        p_class = torch.tensor([[0.7, 0.2, 0.1]])
        true_class = torch.tensor([0])
        result = compute_entropy_stats(p_class, true_class)
        assert isinstance(result["overall"]["mean"], float)


class TestComputeConfidenceAnalysis:
    """Tests for compute_confidence_analysis."""

    def test_high_confidence_correct(self):
        """High-confidence correct predictions should show high metrics."""
        p_class = torch.tensor([[0.95, 0.03, 0.02]] * 20)
        true_class = torch.tensor([0] * 20)
        result = compute_confidence_analysis(p_class, true_class)
        assert result["mean_confidence"] > 0.9
        assert result["high_confidence_accuracy"] == 1.0

    def test_empty_input(self):
        """Empty input should return error dict."""
        p_class = torch.zeros(0, 3)
        true_class = torch.zeros(0, dtype=torch.long)
        result = compute_confidence_analysis(p_class, true_class)
        assert "error" in result

    def test_confidence_buckets_exist(self):
        """Should have confidence buckets."""
        p_class = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])
        true_class = torch.tensor([0, 0])
        result = compute_confidence_analysis(p_class, true_class)
        assert "confidence_buckets" in result
        assert len(result["confidence_buckets"]) == 4

    def test_custom_threshold(self):
        """Should respect custom confidence threshold."""
        p_class = torch.tensor([[0.7, 0.2, 0.1]])
        true_class = torch.tensor([0])
        result = compute_confidence_analysis(p_class, true_class, confidence_threshold=0.8)
        assert result["confidence_threshold"] == 0.8

    def test_invalid_threshold_zero(self):
        """Should raise ValueError for threshold <= 0."""
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            compute_confidence_analysis(
                torch.tensor([[0.5, 0.3, 0.2]]),
                torch.tensor([0]),
                confidence_threshold=0.0,
            )

    def test_invalid_threshold_one(self):
        """Should raise ValueError for threshold >= 1."""
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            compute_confidence_analysis(
                torch.tensor([[0.5, 0.3, 0.2]]),
                torch.tensor([0]),
                confidence_threshold=1.0,
            )

    def test_invalid_p_class_dim(self):
        """Should raise ValueError for non-2D p_class."""
        with pytest.raises(ValueError, match="p_class must be 2D"):
            compute_confidence_analysis(torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0]))


class TestComputeProbabilityDistributions:
    """Tests for compute_probability_distributions."""

    def test_basic_distribution(self):
        """Should compute mean/std/median for each class."""
        p_class = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ])
        true_class = torch.tensor([0, 0, 1])
        result = compute_probability_distributions(p_class, true_class)
        assert "true_mcar" in result
        assert result["true_mcar"]["count"] == 2
        assert len(result["true_mcar"]["mean_p"]) == 3

    def test_missing_class_returns_zeros(self):
        """Class with no samples returns zero distributions."""
        p_class = torch.tensor([[0.9, 0.05, 0.05]])
        true_class = torch.tensor([0])
        result = compute_probability_distributions(p_class, true_class)
        assert result["true_mar"]["count"] == 0
        assert result["true_mar"]["mean_p"] == [0.0, 0.0, 0.0]

    def test_single_sample_per_class(self):
        """Single sample per class should have zero std."""
        p_class = torch.tensor([[0.9, 0.05, 0.05]])
        true_class = torch.tensor([0])
        result = compute_probability_distributions(p_class, true_class)
        assert result["true_mcar"]["std_p"] == [0.0, 0.0, 0.0]

    def test_invalid_p_class_dim(self):
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="p_class must be 2D"):
            compute_probability_distributions(
                torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0])
            )

    def test_all_classes_present(self):
        """All three class keys should be in result."""
        p_class = torch.tensor([[0.5, 0.3, 0.2]])
        true_class = torch.tensor([0])
        result = compute_probability_distributions(p_class, true_class)
        assert "true_mcar" in result
        assert "true_mar" in result
        assert "true_mnar" in result
