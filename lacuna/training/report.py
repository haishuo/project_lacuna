"""
lacuna.training.report

Evaluation report generation utilities.

Produces JSON-serializable evaluation reports from DetailedValResult,
including confusion matrix, confidence analysis, probability distributions,
entropy statistics, Expected Calibration Error (ECE), and per-generator
accuracy breakdown.

Raw prediction tensors (p_class, true_class, generator_ids) are saved
separately as .pt files for downstream analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from lacuna.training.trainer import DetailedValResult


CLASS_NAMES = ["MCAR", "MAR", "MNAR"]


def generate_eval_report(
    result: DetailedValResult,
    registry=None,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> dict:
    """Build a complete JSON-serializable evaluation report.

    Args:
        result: DetailedValResult from trainer.validate_detailed().
        registry: Optional GeneratorRegistry for generator name lookup.
        checkpoint_path: Path to the checkpoint being evaluated.
        config_path: Path to the config used.

    Returns:
        Dict ready for json.dump().
    """
    p_class = result.all_p_class       # [N, 3] CPU
    true_class = result.all_true_class  # [N] CPU
    gen_ids = result.all_generator_ids  # [N] CPU
    preds = p_class.argmax(dim=-1)     # [N]

    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "config": str(config_path) if config_path else None,
        "n_samples": result.n_samples,
        "summary": _summary_section(result.metrics),
        "confusion_matrix": _confusion_section(result.confusion_matrix),
        "confidence_analysis": compute_confidence_analysis(p_class, true_class),
        "probability_distributions": compute_probability_distributions(p_class, true_class),
        "entropy": compute_entropy_stats(p_class, true_class),
        "calibration": compute_ece(p_class, true_class),
        "per_generator_accuracy": compute_per_generator_accuracy(
            preds, true_class, gen_ids, registry
        ),
    }
    return report


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _summary_section(metrics: Dict[str, float]) -> dict:
    return {
        "accuracy": metrics.get("val_acc", 0.0),
        "mcar_acc": metrics.get("val_mcar_acc", 0.0),
        "mar_acc": metrics.get("val_mar_acc", 0.0),
        "mnar_acc": metrics.get("val_mnar_acc", 0.0),
        "loss": metrics.get("val_loss", 0.0),
    }


def _confusion_section(cm: np.ndarray) -> dict:
    """Build confusion matrix section with precision/recall/F1."""
    cm_list = cm.tolist()

    # Precision per class = TP / (TP + FP)
    precision = []
    recall = []
    f1 = []
    for i in range(3):
        tp = cm[i, i]
        col_sum = cm[:, i].sum()
        row_sum = cm[i, :].sum()
        p = tp / col_sum if col_sum > 0 else 0.0
        r = tp / row_sum if row_sum > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(round(p, 4))
        recall.append(round(r, 4))
        f1.append(round(f, 4))

    return {
        "matrix": cm_list,
        "labels": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

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
    """
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


# ---------------------------------------------------------------------------
# Probability distributions per true class
# ---------------------------------------------------------------------------

def compute_probability_distributions(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
) -> dict:
    """Compute mean/std/median of p_class for each true class.

    This answers: "When the true label is MCAR, what does the model
    typically predict?" — revealing whether the model is confident or
    confused for each class.
    """
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


# ---------------------------------------------------------------------------
# Entropy statistics
# ---------------------------------------------------------------------------

def compute_entropy_stats(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
) -> dict:
    """Compute prediction entropy statistics per true class.

    Entropy of [1/3, 1/3, 1/3] = log(3) ≈ 1.099 (max uncertainty).
    Entropy of [1, 0, 0] = 0 (perfect confidence).
    """
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


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error.

    ECE measures how well the predicted probabilities match observed
    accuracy. A perfectly calibrated model has ECE = 0.

    We use the standard binning approach: partition predictions by
    confidence into equal-width bins, then compute the weighted average
    of |accuracy - confidence| per bin.

    Args:
        p_class: [N, 3] probability predictions.
        true_class: [N] ground truth labels.
        n_bins: Number of calibration bins.

    Returns:
        Dict with ECE value and per-bin calibration data.
    """
    confidence, preds = p_class.max(dim=-1)  # [N], [N]
    correct = (preds == true_class).float()

    n = len(confidence)
    if n == 0:
        return {"ece": 0.0, "bins": []}

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []
    ece = 0.0

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]

        # Include right boundary for last bin
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        count = mask.sum().item()
        if count == 0:
            bins_data.append({
                "range": f"{lo:.2f}-{hi:.2f}",
                "count": 0,
                "mean_confidence": 0.0,
                "accuracy": 0.0,
                "gap": 0.0,
            })
            continue

        bin_conf = confidence[mask].mean().item()
        bin_acc = correct[mask].mean().item()
        gap = abs(bin_acc - bin_conf)
        ece += gap * count / n

        bins_data.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "count": count,
            "mean_confidence": round(bin_conf, 4),
            "accuracy": round(bin_acc, 4),
            "gap": round(gap, 4),
        })

    return {
        "ece": round(ece, 4),
        "n_bins": n_bins,
        "bins": bins_data,
    }


# ---------------------------------------------------------------------------
# Per-generator accuracy
# ---------------------------------------------------------------------------

def compute_per_generator_accuracy(
    preds: torch.Tensor,
    true_class: torch.Tensor,
    generator_ids: torch.Tensor,
    registry=None,
) -> dict:
    """Compute accuracy breakdown per generator.

    Args:
        preds: [N] predicted class indices.
        true_class: [N] ground truth labels.
        generator_ids: [N] generator IDs.
        registry: Optional GeneratorRegistry for name lookup.

    Returns:
        Dict mapping generator_id -> {name, class, accuracy, count}.
    """
    class_map = {0: "MCAR", 1: "MAR", 2: "MNAR"}
    result = {}

    unique_gens = generator_ids.unique().tolist()
    for gen_id in sorted(unique_gens):
        gen_id_int = int(gen_id)
        mask = generator_ids == gen_id
        count = mask.sum().item()
        correct = (preds[mask] == true_class[mask]).sum().item()
        acc = correct / count if count > 0 else 0.0

        # Determine class from ground truth (all samples for a generator
        # should share the same class, but take mode to be safe)
        gen_true_classes = true_class[mask]
        most_common_class = gen_true_classes.mode().values.item()

        # Look up name from registry if available
        name = f"generator_{gen_id_int}"
        if registry is not None:
            try:
                gen = registry[gen_id_int]
                name = gen.name
            except (KeyError, IndexError):
                pass

        result[str(gen_id_int)] = {
            "name": name,
            "class": class_map.get(most_common_class, "UNKNOWN"),
            "accuracy": round(acc, 4),
            "count": count,
        }

    return result


# ---------------------------------------------------------------------------
# Raw predictions I/O
# ---------------------------------------------------------------------------

def save_raw_predictions(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
    generator_ids: torch.Tensor,
    path: Path,
) -> None:
    """Save raw prediction tensors to a .pt file for downstream analysis.

    Args:
        p_class: [N, 3] probability predictions.
        true_class: [N] ground truth labels.
        generator_ids: [N] generator IDs.
        path: Output .pt file path.
    """
    torch.save({
        "p_class": p_class,
        "true_class": true_class,
        "generator_ids": generator_ids,
    }, path)


def load_raw_predictions(path: Path) -> dict:
    """Load raw prediction tensors from a .pt file.

    Returns:
        Dict with keys: p_class, true_class, generator_ids.
    """
    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

def print_eval_summary(report: dict) -> None:
    """Print concise evaluation summary to console."""
    s = report["summary"]
    cm = report["confusion_matrix"]

    print("\nEvaluation Report")
    print("=" * 50)
    print(f"Accuracy: {s['accuracy']*100:.1f}%  |  "
          f"MCAR: {s['mcar_acc']*100:.1f}%  "
          f"MAR: {s['mar_acc']*100:.1f}%  "
          f"MNAR: {s['mnar_acc']*100:.1f}%")

    # Confusion matrix
    m = cm["matrix"]
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"         {'MCAR':>5s}  {'MAR':>5s}  {'MNAR':>5s}")
    for i, label in enumerate(CLASS_NAMES):
        row = "  ".join(f"{m[i][j]:5d}" for j in range(3))
        print(f"  {label:4s} [ {row} ]")

    # Precision / Recall / F1
    print(f"\n  Prec:  {'  '.join(f'{v*100:5.1f}' for v in cm['precision'])}")
    print(f"  Rec:   {'  '.join(f'{v*100:5.1f}' for v in cm['recall'])}")
    print(f"  F1:    {'  '.join(f'{v*100:5.1f}' for v in cm['f1'])}")

    # Confidence
    conf = report["confidence_analysis"]
    print(f"\nConfidence: mean={conf['mean_confidence']:.3f}, "
          f"correct={conf['mean_confidence_correct']:.3f}, "
          f"incorrect={conf['mean_confidence_incorrect']:.3f}")
    print(f"Low-confidence (<{conf['confidence_threshold']*100:.0f}%): "
          f"{conf['low_confidence_samples_pct']*100:.1f}% of predictions")

    # Calibration
    cal = report["calibration"]
    print(f"ECE: {cal['ece']:.4f}")

    # Entropy
    ent = report["entropy"]
    print(f"Entropy: mean={ent['overall']['mean']:.3f} "
          f"(normalized={ent['overall']['mean_normalized']:.3f})")

    # Probability distributions
    pd = report["probability_distributions"]
    print(f"\nMean predicted probabilities by true class:")
    print(f"           {'MCAR':>7s} {'MAR':>7s} {'MNAR':>7s}")
    for class_name in CLASS_NAMES:
        key = f"true_{class_name.lower()}"
        if key in pd and pd[key]["count"] > 0:
            mp = pd[key]["mean_p"]
            print(f"  {class_name:4s}:  {mp[0]:7.3f} {mp[1]:7.3f} {mp[2]:7.3f}  (n={pd[key]['count']})")
