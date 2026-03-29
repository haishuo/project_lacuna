"""
lacuna.metrics.uncertainty

Entropy, confidence, and probability distribution metrics.
"""

from typing import List

import numpy as np
import torch


CLASS_NAMES = ("MCAR", "MAR", "MNAR")


def compute_entropy_stats(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
) -> dict:
    """Compute prediction entropy statistics per true class.

    Entropy of [1/3, 1/3, 1/3] = log(3) ~ 1.099 (max uncertainty).
    Entropy of [1, 0, 0] = 0 (perfect confidence).

    Args:
        p_class: [N, 3] probability predictions.
        true_class: [N] ground truth labels.

    Returns:
        Dict with overall and per-class entropy statistics.

    Raises:
        ValueError: If p_class is not 2D or true_class is not 1D.
    """
    if p_class.ndim != 2:
        raise ValueError(f"p_class must be 2D, got {p_class.ndim}D")
    if true_class.ndim != 1:
        raise ValueError(f"true_class must be 1D, got {true_class.ndim}D")

    # Clamp to avoid log(0)
    p_clamped = p_class.clamp(min=1e-8)
    entropy = -(p_clamped * p_clamped.log()).sum(dim=-1)  # [N]
    max_entropy = np.log(3.0)

    preds = p_class.argmax(dim=-1)
    correct = (preds == true_class)

    overall = {
        "mean": round(entropy.mean().item(), 4),
        "std": round(entropy.std().item(), 4) if len(entropy) > 1 else 0.0,
        "max_possible": round(max_entropy, 4),
        "mean_normalized": round(entropy.mean().item() / max_entropy, 4),
    }

    # Entropy for correct vs incorrect predictions
    correct_entropy = entropy[correct].mean().item() if correct.sum() > 0 else 0.0
    incorrect_entropy = entropy[~correct].mean().item() if (~correct).sum() > 0 else 0.0

    per_class = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = true_class == class_idx
        if mask.sum() == 0:
            per_class[class_name.lower()] = {"mean": 0.0, "std": 0.0, "count": 0}
            continue
        class_entropy = entropy[mask]
        per_class[class_name.lower()] = {
            "mean": round(class_entropy.mean().item(), 4),
            "std": round(class_entropy.std().item(), 4) if len(class_entropy) > 1 else 0.0,
            "count": int(mask.sum().item()),
        }

    return {
        "overall": overall,
        "correct_mean": round(correct_entropy, 4),
        "incorrect_mean": round(incorrect_entropy, 4),
        "per_class": per_class,
    }


def compute_confidence_analysis(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
    confidence_threshold: float = 0.5,
) -> dict:
    """Analyse prediction confidence vs correctness.

    Confidence = max probability assigned to any class (range [1/3, 1]).

    Args:
        p_class: [N, 3] probability predictions.
        true_class: [N] ground truth labels.
        confidence_threshold: boundary for "low confidence".

    Returns:
        Dict with confidence statistics and buckets.

    Raises:
        ValueError: If p_class is not 2D or true_class is not 1D.
        ValueError: If confidence_threshold is not in (0, 1).
    """
    if p_class.ndim != 2:
        raise ValueError(f"p_class must be 2D, got {p_class.ndim}D")
    if true_class.ndim != 1:
        raise ValueError(f"true_class must be 1D, got {true_class.ndim}D")
    if not (0.0 < confidence_threshold < 1.0):
        raise ValueError(
            f"confidence_threshold must be in (0, 1), got {confidence_threshold}"
        )

    confidence, _ = p_class.max(dim=-1)  # [N]
    preds = p_class.argmax(dim=-1)
    correct = (preds == true_class)

    n = len(confidence)
    if n == 0:
        return {"error": "no samples"}

    mean_conf = confidence.mean().item()
    mean_conf_correct = confidence[correct].mean().item() if correct.sum() > 0 else 0.0
    mean_conf_incorrect = confidence[~correct].mean().item() if (~correct).sum() > 0 else 0.0

    # High-confidence accuracy (confidence >= 0.7)
    high_conf_mask = confidence >= 0.7
    high_conf_acc = (
        correct[high_conf_mask].float().mean().item()
        if high_conf_mask.sum() > 0
        else 0.0
    )

    low_conf_pct = (confidence < confidence_threshold).float().mean().item()

    # Confidence buckets
    bucket_edges = [1.0 / 3.0, 0.50, 0.70, 0.90, 1.001]
    buckets = []
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        count = mask.sum().item()
        acc = correct[mask].float().mean().item() if count > 0 else 0.0
        buckets.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "count": count,
            "accuracy": round(acc, 4),
        })

    return {
        "mean_confidence": round(mean_conf, 4),
        "mean_confidence_correct": round(mean_conf_correct, 4),
        "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
        "high_confidence_accuracy": round(high_conf_acc, 4),
        "low_confidence_samples_pct": round(low_conf_pct, 4),
        "confidence_threshold": confidence_threshold,
        "confidence_buckets": buckets,
    }


def compute_probability_distributions(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
) -> dict:
    """Compute mean/std/median of p_class for each true class.

    This answers: "When the true label is MCAR, what does the model
    typically predict?" -- revealing whether the model is confident or
    confused for each class.

    Args:
        p_class: [N, 3] probability predictions.
        true_class: [N] ground truth labels.

    Returns:
        Dict with probability statistics per true class.

    Raises:
        ValueError: If p_class is not 2D or true_class is not 1D.
    """
    if p_class.ndim != 2:
        raise ValueError(f"p_class must be 2D, got {p_class.ndim}D")
    if true_class.ndim != 1:
        raise ValueError(f"true_class must be 1D, got {true_class.ndim}D")

    result = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = true_class == class_idx
        if mask.sum() == 0:
            result[f"true_{class_name.lower()}"] = {
                "mean_p": [0.0, 0.0, 0.0],
                "std_p": [0.0, 0.0, 0.0],
                "median_p": [0.0, 0.0, 0.0],
                "count": 0,
            }
            continue

        subset = p_class[mask]  # [K, 3]
        result[f"true_{class_name.lower()}"] = {
            "mean_p": subset.mean(dim=0).tolist(),
            "std_p": subset.std(dim=0).tolist() if subset.shape[0] > 1 else [0.0, 0.0, 0.0],
            "median_p": subset.median(dim=0).values.tolist(),
            "count": int(mask.sum().item()),
        }

    return result
