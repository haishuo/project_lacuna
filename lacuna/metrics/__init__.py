"""
lacuna.metrics

Evaluation metrics for classification, calibration, and uncertainty.

Re-exports all metric functions from submodules.
"""

from lacuna.metrics.calibration import compute_ece
from lacuna.metrics.classification import (
    compute_per_generator_accuracy,
    compute_selective_accuracy,
)
from lacuna.metrics.uncertainty import (
    compute_entropy_stats,
    compute_confidence_analysis,
    compute_probability_distributions,
)

__all__ = [
    "compute_ece",
    "compute_per_generator_accuracy",
    "compute_selective_accuracy",
    "compute_entropy_stats",
    "compute_confidence_analysis",
    "compute_probability_distributions",
]
