"""
lacuna.metrics.calibration

Expected Calibration Error (ECE) computation.
"""

import torch


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

    Raises:
        ValueError: If p_class is not 2D or true_class is not 1D.
        ValueError: If n_bins < 1.
    """
    if p_class.ndim != 2:
        raise ValueError(f"p_class must be 2D, got {p_class.ndim}D")
    if true_class.ndim != 1:
        raise ValueError(f"true_class must be 1D, got {true_class.ndim}D")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

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
