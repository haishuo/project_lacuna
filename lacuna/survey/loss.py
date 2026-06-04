"""
lacuna.survey.loss

The P2.2 training objective and calibration fit (PROPOSAL-P2 §4; audit §5, §7).

ONE job: the proper, order-aware scoring of an ordered δ-bin distribution against the true bin,
plus the post-hoc temperature fit. Primary loss = **Ranked Probability Score (RPS)** — the
discrete CRPS over the bin CDF — which is proper AND penalizes far-off-bin misses more than
near misses (the property a plain cross-entropy cannot see). RPS is the ONLY training objective;
the cross-entropy / log-score here is a SECONDARY diagnostic, never the main objective (P2.2
forbids 3-class CE and binary MAR/MNAR loss).

Definitions (per example; mean-reduced over the batch):
    p_k = softmax(logits)_k                       predicted bin probabilities
    P_k = sum_{j<=k} p_j                          predicted CDF
    Y_k = 1[k >= true_bin]                         truth CDF (one-hot truth's cumulative)
    RPS = sum_{k=0..K-1} (P_k - Y_k)^2            discrete RPS (a.k.a. CRPS over bins)
`normalize=True` divides by (K-1) so a single-step miss in a K-bin scale is comparable across K.

Contract (Rule 1): logits must be [B, K] with K>=2; labels [B] integer in [0, K); shape/dtype/range
violations raise loudly. Reductions: "mean" (default) or "none".
"""

import torch
import torch.nn.functional as F


def _validate(logits: torch.Tensor, labels: torch.Tensor) -> int:
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [B, K], got {tuple(logits.shape)}")
    b, k = logits.shape
    if k < 2:
        raise ValueError(f"need K >= 2 ordered bins, got K={k}")
    if labels.dim() != 1 or labels.shape[0] != b:
        raise ValueError(f"labels must be [B={b}], got {tuple(labels.shape)}")
    if not (labels.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long)):
        raise ValueError(f"labels must be integer dtype, got {labels.dtype}")
    if int(labels.min()) < 0 or int(labels.max()) >= k:
        raise ValueError(f"labels must be in [0, {k}); got min={int(labels.min())}, max={int(labels.max())}")
    return k


def _reduce(per_example: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return per_example.mean()
    if reduction == "none":
        return per_example
    raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")


def rps_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    normalize: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Ranked Probability Score over ordered δ-bins. Differentiable in `logits`.

    Args:
        logits: [B, K] raw bin logits.
        labels: [B] integer true-bin indices in [0, K).
        normalize: divide the per-example RPS by (K-1) to land in [0, 1].
        reduction: "mean" or "none".
    """
    k = _validate(logits, labels)
    p = torch.softmax(logits, dim=-1)
    pred_cdf = torch.cumsum(p, dim=-1)  # [B, K]
    # truth CDF: 1 where bin index >= true label
    idx = torch.arange(k, device=logits.device).unsqueeze(0)  # [1, K]
    truth_cdf = (idx >= labels.unsqueeze(1)).to(p.dtype)  # [B, K]
    per_example = ((pred_cdf - truth_cdf) ** 2).sum(dim=-1)  # [B]
    if normalize:
        per_example = per_example / (k - 1)
    return _reduce(per_example, reduction)


def log_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy / negative log-likelihood of the true bin — SECONDARY diagnostic only."""
    _validate(logits, labels)
    per_example = F.cross_entropy(logits, labels, reduction="none")
    return _reduce(per_example, reduction)


def uniform_rps(num_bins: int, normalize: bool = True) -> float:
    """RPS of the uniform prediction against ANY single true bin, AVERAGED over true bins.

    A reference baseline for the success criteria (audit §13). Deterministic, closed form via
    direct evaluation (no RNG).
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    logits = torch.zeros(num_bins, num_bins)  # uniform for every row
    labels = torch.arange(num_bins)
    return float(rps_loss(logits, labels, normalize=normalize, reduction="mean").item())


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lo: float = 0.05,
    hi: float = 10.0,
    iters: int = 50,
) -> float:
    """Post-hoc temperature minimizing val NLL (log-score) on frozen logits (audit §7).

    Golden-section search on a unimodal-in-practice NLL(T) over [lo, hi]; deterministic,
    no gradient, no global state. Returns the scalar T > 0. Implemented locally rather than
    reusing `training.calibration.find_optimal_temperature` because that path is coupled to
    `LacunaModel` gate logits, not a bare logits tensor.
    """
    _validate(logits, labels)
    if not (0.0 < lo < hi):
        raise ValueError(f"require 0 < lo < hi, got lo={lo}, hi={hi}")

    logits = logits.detach()

    def nll(t: float) -> float:
        return float(log_score(logits / t, labels, reduction="mean").item())

    invphi = (5 ** 0.5 - 1) / 2  # 1/phi
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = nll(d)
    return 0.5 * (a + b)
