"""
lacuna.metrics.classification

Per-generator accuracy and selective accuracy metrics.
"""

from typing import List, Optional

import torch


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

    Raises:
        ValueError: If any input tensor is not 1D.
        ValueError: If tensor lengths do not match.
    """
    if preds.ndim != 1:
        raise ValueError(f"preds must be 1D, got {preds.ndim}D")
    if true_class.ndim != 1:
        raise ValueError(f"true_class must be 1D, got {true_class.ndim}D")
    if generator_ids.ndim != 1:
        raise ValueError(f"generator_ids must be 1D, got {generator_ids.ndim}D")
    if not (len(preds) == len(true_class) == len(generator_ids)):
        raise ValueError(
            f"Tensor lengths must match: preds={len(preds)}, "
            f"true_class={len(true_class)}, generator_ids={len(generator_ids)}"
        )

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
            except (KeyError, IndexError) as e:
                # Rule 1: Log rather than silently swallow. The fallback
                # name is intentional (generators may be pruned from registry)
                # but we document the miss rather than hiding it.
                import warnings
                warnings.warn(
                    f"Generator {gen_id_int} not found in registry: {e}. "
                    f"Using fallback name '{name}'."
                )

        result[str(gen_id_int)] = {
            "name": name,
            "class": class_map.get(most_common_class, "UNKNOWN"),
            "accuracy": round(acc, 4),
            "count": count,
        }

    return result


def compute_selective_accuracy(
    p_class: torch.Tensor,
    true_class: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> dict:
    """Compute accuracy-coverage trade-off at confidence thresholds.

    At threshold t, we only predict on samples where max(p_class) >= t.
    - **Accuracy @ t**: accuracy on those high-confidence samples.
    - **Coverage @ t**: fraction of all samples that meet the threshold.

    Args:
        p_class: [N, 3] probability predictions.
        true_class: [N] ground truth labels.
        thresholds: Confidence thresholds to evaluate.
            Defaults to [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95].

    Returns:
        Dict with per-threshold accuracy/coverage and overall summary.

    Raises:
        ValueError: If p_class is not 2D or true_class is not 1D.
    """
    if p_class.ndim != 2:
        raise ValueError(f"p_class must be 2D, got {p_class.ndim}D")
    if true_class.ndim != 1:
        raise ValueError(f"true_class must be 1D, got {true_class.ndim}D")

    if thresholds is None:
        thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    confidence, preds = p_class.max(dim=-1)  # [N], [N]
    correct = (preds == true_class)
    n = len(confidence)

    if n == 0:
        return {"error": "no samples"}

    rows = []
    for tau in thresholds:
        mask = confidence >= tau
        count = mask.sum().item()
        coverage = count / n
        acc = correct[mask].float().mean().item() if count > 0 else 0.0

        rows.append({
            "threshold": tau,
            "accuracy": round(acc, 4),
            "coverage": round(coverage, 4),
            "count": count,
        })

    # Find the threshold where accuracy first exceeds 90% and 95%
    acc_90_row = next((r for r in rows if r["accuracy"] >= 0.90), None)
    acc_95_row = next((r for r in rows if r["accuracy"] >= 0.95), None)

    return {
        "thresholds": rows,
        "acc_90_threshold": acc_90_row["threshold"] if acc_90_row else None,
        "acc_90_coverage": acc_90_row["coverage"] if acc_90_row else None,
        "acc_95_threshold": acc_95_row["threshold"] if acc_95_row else None,
        "acc_95_coverage": acc_95_row["coverage"] if acc_95_row else None,
    }
