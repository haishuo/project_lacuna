"""
lacuna.survey.distributional_stream

Differentiable distributional-consequence pooling for the δ-prior head
(PROPOSAL-distributional-consequence-stream-audit §1A-ii, §3; operationalizes Proposal B of
ARCHITECTURE-FITNESS-delta-estimation.md).

ONE job: turn the encoder's per-row representation of the SUPPLIED target column into a fixed-width
**order-statistic** vector — a learned ECDF — across rows, REPLACING the mean pool that averaged the
tail away (`conditioned_head._pool_target`). Each valid row's target rep is projected to `n_probes`
scalar "shape probes"; for each probe we take a fixed grid of `Q` quantiles across rows plus the max.
This exposes the within-column, across-row distributional SHAPE (truncated/depleted upper tail) the
δ footprint lives in, and is mechanism-GENERAL: no τ / step indicator / idiom label enters — it asks
only "what does the observed distribution of this column look like?" (NORTH-STAR §3, §6).

Differentiable (Proposal §5): masked-quantile pooling is sort + gather + linear interpolation, so
gradients flow from the δ loss through the quantiles into the encoder — the mechanism by which the
encoder is TAUGHT to surface the tail. Deterministic (Rule 6): sorting is deterministic; no RNG.
Fail loud (Rule 1) on bad shapes or an all-padding example.
"""

import torch
import torch.nn as nn

# Frozen quantile grid (order matters; recorded in the manifest). Upper-tail-weighted because the
# censoring consequence (truncation/top-coding) lives in the upper tail.
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
N_QUANTILES = len(QUANTILE_LEVELS)


def masked_quantile_pool(values: torch.Tensor, row_mask: torch.Tensor,
                         levels=QUANTILE_LEVELS) -> torch.Tensor:
    """Per-(example, probe) quantiles + max over the VALID rows.

    Args:
        values: [B, R, M] probe values (M probe channels per row).
        row_mask: [B, R] bool — True = valid row.
        levels: ascending quantile levels in [0, 1].

    Returns:
        [B, M, len(levels)+1] — the `levels` quantiles then the max, per probe.

    Raises:
        ValueError: bad shapes, or an example with zero valid rows.
    """
    if values.dim() != 3:
        raise ValueError(f"values must be [B, R, M], got {tuple(values.shape)}")
    b, r, m = values.shape
    if row_mask.shape != (b, r):
        raise ValueError(f"row_mask must be [{b}, {r}], got {tuple(row_mask.shape)}")
    mask = row_mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every example must have >= 1 valid row for distributional pooling")

    # Fold the probe channels into the batch: [B*M, R].
    v = values.permute(0, 2, 1).reshape(b * m, r)
    mexp = mask.unsqueeze(1).expand(b, m, r).reshape(b * m, r)
    # Push invalid rows to +inf so they sort to the end and are never gathered.
    masked = torch.where(mexp, v, torch.full_like(v, float("inf")))
    sorted_vals, _ = masked.sort(dim=1)  # ascending; valid first, +inf last
    n = mexp.sum(dim=1).clamp(min=1).to(values.dtype)  # [B*M] valid count

    stats = []
    for q in levels:
        pos = float(q) * (n - 1.0)  # interpolation position in [0, n-1]
        lo = pos.floor().long()
        hi = pos.ceil().long()
        frac = (pos - lo.to(values.dtype)).unsqueeze(1)
        lo_v = sorted_vals.gather(1, lo.unsqueeze(1))
        hi_v = sorted_vals.gather(1, hi.unsqueeze(1))
        stats.append(lo_v + frac * (hi_v - lo_v))  # [B*M, 1]
    max_idx = (n.long() - 1).unsqueeze(1)
    stats.append(sorted_vals.gather(1, max_idx))  # [B*M, 1] max = quantile at 1.0
    out = torch.cat(stats, dim=1)  # [B*M, Q+1]
    return out.reshape(b, m, len(levels) + 1)


class RepECDFPooling(nn.Module):
    """Learned ECDF pooling: project target reps to `n_probes`, then masked quantiles + max over rows.

    Replaces the across-row MEAN in the target-conditioned head with an order-statistic summary, so
    the head (and, via backprop, the encoder) sees the tail SHAPE rather than its average.
    """

    def __init__(self, hidden_dim: int, n_probes: int = 4, levels=QUANTILE_LEVELS):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if n_probes < 1:
            raise ValueError(f"n_probes must be >= 1, got {n_probes}")
        self.hidden_dim = int(hidden_dim)
        self.n_probes = int(n_probes)
        self.levels = tuple(float(x) for x in levels)
        self.proj = nn.Linear(hidden_dim, n_probes)
        self.out_dim = n_probes * (len(self.levels) + 1)

    def forward(self, target_reps: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        """(target per-row reps [B, R, H], row_mask [B, R]) -> [B, out_dim] order-statistic vector."""
        if target_reps.dim() != 3 or target_reps.shape[2] != self.hidden_dim:
            raise ValueError(
                f"target_reps must be [B, R, {self.hidden_dim}], got {tuple(target_reps.shape)}"
            )
        probes = self.proj(target_reps)  # [B, R, n_probes]
        stats = masked_quantile_pool(probes, row_mask, self.levels)  # [B, n_probes, Q+1]
        return stats.reshape(stats.shape[0], self.out_dim)

    def schema(self) -> dict:
        """Manifest-ready description of the frozen pooling scheme."""
        return {"kind": "rep_ecdf_pooling", "n_probes": self.n_probes,
                "quantile_levels": list(self.levels), "plus_max": True, "out_dim": self.out_dim}
