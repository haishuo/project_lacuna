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
from lacuna.metrics import (
    compute_ece,
    compute_per_generator_accuracy,
    compute_selective_accuracy,
    compute_entropy_stats,
    compute_confidence_analysis,
    compute_probability_distributions,
)


CLASS_NAMES = ("MCAR", "MAR", "MNAR")


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
        "timestamp": datetime.now().isoformat(),  # NON-DETERMINISTIC: system clock
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "config": str(config_path) if config_path else None,
        "n_samples": result.n_samples,
        "summary": _summary_section(result.metrics),
        "confusion_matrix": _confusion_section(result.confusion_matrix),
        "confidence_analysis": compute_confidence_analysis(p_class, true_class),
        "probability_distributions": compute_probability_distributions(p_class, true_class),
        "entropy": compute_entropy_stats(p_class, true_class),
        "calibration": compute_ece(p_class, true_class),
        "selective_accuracy": compute_selective_accuracy(p_class, true_class),
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

    # Selective accuracy (accuracy vs coverage)
    sel = report.get("selective_accuracy", {})
    sel_rows = sel.get("thresholds", [])
    if sel_rows:
        print(f"\nSelective Accuracy (accuracy @ confidence threshold):")
        print(f"  {'τ':>5s}  {'Acc':>6s}  {'Cov':>6s}  {'Count':>6s}")
        for r in sel_rows:
            print(f"  {r['threshold']:5.2f}  {r['accuracy']*100:5.1f}%  "
                  f"{r['coverage']*100:5.1f}%  {r['count']:6d}")
        if sel.get("acc_90_threshold") is not None:
            print(f"  → 90% accuracy reached at τ={sel['acc_90_threshold']:.2f} "
                  f"(coverage={sel['acc_90_coverage']*100:.1f}%)")
        if sel.get("acc_95_threshold") is not None:
            print(f"  → 95% accuracy reached at τ={sel['acc_95_threshold']:.2f} "
                  f"(coverage={sel['acc_95_coverage']*100:.1f}%)")

    # Entropy
    ent = report["entropy"]
    print(f"\nEntropy: mean={ent['overall']['mean']:.3f} "
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
