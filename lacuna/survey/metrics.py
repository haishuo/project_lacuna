"""
lacuna.survey.metrics

Reporting metrics for the δ-prior (PROPOSAL-P2 §4; audit §6, §7). Calibration-FIRST: these are
diagnostics, NOT the training objective (the optimizer only sees RPS). Accuracy is explicitly
secondary to calibration (charter §4.2).

ONE job: derive the reported quantities from a predicted δ-bin distribution and the truth —
E[δ] error, top-1 and adjacent bin accuracy, P(δ=0), credible-interval coverage, and ECE
(reusing `lacuna.metrics.calibration.compute_ece`, which is generic over K).

Determinism: pure functions of their tensor inputs; no RNG, no global state.
"""

import torch

from lacuna.metrics.calibration import compute_ece

from .delta_bins import bin_edges

# Representative δ for the open tail bin = last finite edge + this offset (recorded; audit §6).
TAIL_OFFSET = 0.5


def bin_centers(tail_offset: float = TAIL_OFFSET) -> torch.Tensor:
    """Representative δ value per bin: 0 for the MAR bin, midpoints for finite bins, edge+offset
    for the open tail. Deterministic, derived from `delta_bins.bin_edges`."""
    meta = bin_edges()
    edges = meta["finite_upper_edges"]
    centers = [0.0]  # bin 0 (MAR)
    lower = 0.0
    for upper in edges:
        centers.append(0.5 * (lower + upper))
        lower = upper
    centers.append(lower + tail_offset)  # tail bin
    return torch.tensor(centers, dtype=torch.float32)


def expected_delta(probs: torch.Tensor, tail_offset: float = TAIL_OFFSET) -> torch.Tensor:
    """E[δ] per example = Σ p_k · center_k. `probs` is [B, K]."""
    if probs.dim() != 2:
        raise ValueError(f"probs must be [B, K], got {tuple(probs.shape)}")
    centers = bin_centers(tail_offset).to(probs.device)
    if probs.shape[1] != centers.shape[0]:
        raise ValueError(f"probs K={probs.shape[1]} != num_bins={centers.shape[0]}")
    return (probs * centers.unsqueeze(0)).sum(dim=-1)


def e_delta_error(
    probs: torch.Tensor, true_delta: torch.Tensor, tail_offset: float = TAIL_OFFSET
) -> dict:
    """MAE / RMSE of E[δ] vs the answer-sheet continuous δ."""
    if true_delta.dim() != 1 or true_delta.shape[0] != probs.shape[0]:
        raise ValueError(f"true_delta must be [B={probs.shape[0]}], got {tuple(true_delta.shape)}")
    pred = expected_delta(probs, tail_offset)
    err = pred - true_delta.to(pred.dtype)
    return {"mae": float(err.abs().mean().item()), "rmse": float((err ** 2).mean().sqrt().item())}


def bin_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 bin accuracy (argmax vs true bin)."""
    if probs.dim() != 2:
        raise ValueError(f"probs must be [B, K], got {tuple(probs.shape)}")
    pred = probs.argmax(dim=-1)
    return float((pred == labels.to(pred.device)).float().mean().item())


def adjacent_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of examples whose argmax bin is within ±1 of the true bin (order-aware)."""
    if probs.dim() != 2:
        raise ValueError(f"probs must be [B, K], got {tuple(probs.shape)}")
    pred = probs.argmax(dim=-1)
    return float(((pred - labels.to(pred.device)).abs() <= 1).float().mean().item())


def p_delta_zero(probs: torch.Tensor) -> float:
    """Mean predicted mass on bin 0 (the 'plausibly MAR?' quantity). Reported, not thresholded."""
    if probs.dim() != 2:
        raise ValueError(f"probs must be [B, K], got {tuple(probs.shape)}")
    return float(probs[:, 0].mean().item())


def interval_coverage(probs: torch.Tensor, labels: torch.Tensor, nominal: float) -> float:
    """Empirical coverage of the central credible interval at `nominal` mass.

    The interval per example is [lo, hi] in bin-index space taken from the predicted CDF:
    lo = smallest k with CDF_k >= (1-m)/2, hi = smallest k with CDF_k >= (1+m)/2. Coverage is
    the fraction of examples whose true bin lies in [lo, hi]. A well-calibrated prior has
    coverage ≈ nominal (audit §7).
    """
    if not (0.0 < nominal < 1.0):
        raise ValueError(f"nominal must be in (0, 1), got {nominal}")
    if probs.dim() != 2:
        raise ValueError(f"probs must be [B, K], got {tuple(probs.shape)}")
    cdf = torch.cumsum(probs, dim=-1)  # [B, K]
    lo_q = (1.0 - nominal) / 2.0
    hi_q = (1.0 + nominal) / 2.0
    # smallest index where cdf >= q  ==  count of bins with cdf < q
    lo = (cdf < lo_q).sum(dim=-1)  # [B]
    hi = (cdf < hi_q).sum(dim=-1)  # [B]
    k = probs.shape[1]
    lo = lo.clamp(max=k - 1)
    hi = hi.clamp(max=k - 1)
    lab = labels.to(lo.device)
    inside = (lab >= lo) & (lab <= hi)
    return float(inside.float().mean().item())


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> dict:
    """Expected calibration error (top-1 confidence vs accuracy), reusing the generic helper."""
    return compute_ece(probs, labels.to(probs.device), n_bins=n_bins)


def coverage_table(probs: torch.Tensor, labels: torch.Tensor, nominals=(0.5, 0.8, 0.9)) -> dict:
    """Coverage at several nominal levels — the headline calibration check for an ordered prior."""
    return {f"{int(round(m * 100))}": interval_coverage(probs, labels, m) for m in nominals}
