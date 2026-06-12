"""
lacuna/survey/conditional_phi.py

The T3 conditional-φ (PREREGISTRATION-network-load-bearing-review §4): an amortized,
single-forward-pass row-set encoder that ingests the target column AND its predictors and
emits a calibrated δ logit. This is the one genuinely new architectural element of the review
— the design-of-record φ is target-marginal-only; T3 asks whether a conditional φ can
reproduce the explicit multi-imputer pipeline's per-column posterior in one pass.

Contract (Rule 2 — validate inputs at the boundary):
  forward(P, pcol_mask, obs, tval, row_mask) consumes column-major standardized tensors:
    P         [B, R, Dp]  predictor values, standardized, padded over rows and columns
    pcol_mask [B, Dp]     True = valid (non-pad) predictor column
    obs       [B, R]      1.0 = target observed in that row, 0.0 = missing
    tval      [B, R]      standardized target value where observed, 0.0 where missing/pad
    row_mask  [B, R]      True = valid (non-pad) row
  returns logits [B] (pre-temperature). Calibration temperature is applied by the caller.

Permutation-invariance over predictor COLUMNS (shared per-cell encoder + masked mean) lets one
weight set serve datasets of different width — necessary for cross-domain weight sharing. Row
pooling is done separately over {all, observed, missing} row groups so the observed-vs-missing
contrast (the MAR/MNAR footprint) is exposed to the head; empty groups pool to zeros.

Deterministic given parameters and inputs (Rule 6). No hidden state (Rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

QUANTILE_LEVELS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ConditionalPhiConfig:
    """Fixed T3 config (one configuration, locked by the pre-registration)."""
    pred_hidden: int = 32
    hidden: int = 48
    head_hidden: int = 128
    quantile_levels: tuple = QUANTILE_LEVELS


def _masked_group_stats(tokens: torch.Tensor, group_mask: torch.Tensor,
                        levels: tuple) -> torch.Tensor:
    """Mean + interpolated quantiles of `tokens` over the valid rows of `group_mask`.

    Args:
        tokens:     [B, R, H] per-row embeddings.
        group_mask: [B, R] bool — rows belonging to this group.
        levels:     ascending quantile levels in [0, 1].

    Returns:
        [B, H * (1 + len(levels))] — mean then each quantile; zeros for empty groups.
    """
    b, r, h = tokens.shape
    cnt = group_mask.sum(dim=1)                       # [B] valid rows in group
    empty = cnt == 0
    safe = group_mask.clone()
    safe[empty, 0] = True                             # avoid 0/0; zeroed out below
    denom = safe.sum(dim=1, keepdim=True).clamp(min=1).to(tokens.dtype)
    mean = (tokens * safe.unsqueeze(-1)).sum(dim=1) / denom

    big = torch.where(safe.unsqueeze(-1), tokens,
                      torch.full_like(tokens, float("inf")))
    sorted_v, _ = big.sort(dim=1)                     # ascending; +inf last
    n = safe.sum(dim=1).to(tokens.dtype)              # [B]
    qstats = []
    for q in levels:
        pos = float(q) * (n - 1.0).clamp(min=0.0)     # [B]
        lo = pos.floor().long()
        hi = pos.ceil().long()
        frac = (pos - lo.to(tokens.dtype)).view(b, 1)
        lo_v = sorted_v.gather(1, lo.view(b, 1, 1).expand(b, 1, h)).squeeze(1)
        hi_v = sorted_v.gather(1, hi.view(b, 1, 1).expand(b, 1, h)).squeeze(1)
        qstats.append(lo_v + (hi_v - lo_v) * frac)
    stats = torch.cat([mean, *qstats], dim=-1)
    stats[empty] = 0.0
    return stats


class ConditionalPhi(nn.Module):
    """Conditional row-set encoder φ(predictors, target-view) → δ logit."""

    def __init__(self, cfg: ConditionalPhiConfig | None = None):
        super().__init__()
        self.cfg = cfg or ConditionalPhiConfig()
        h, ph = self.cfg.hidden, self.cfg.pred_hidden
        self.pred_cell = nn.Sequential(nn.Linear(1, ph), nn.ReLU(), nn.Linear(ph, h))
        self.row = nn.Sequential(nn.Linear(h + 2, h), nn.ReLU(), nn.Linear(h, h))
        n_stats = h * (1 + len(self.cfg.quantile_levels))
        self.head = nn.Sequential(
            nn.Linear(3 * n_stats, self.cfg.head_hidden), nn.ReLU(),
            nn.Linear(self.cfg.head_hidden, 1))

    def forward(self, P: torch.Tensor, pcol_mask: torch.Tensor, obs: torch.Tensor,
                tval: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        if P.dim() != 3:
            raise ValueError(f"P must be [B, R, Dp], got {tuple(P.shape)}")
        b, r, dp = P.shape
        if pcol_mask.shape != (b, dp):
            raise ValueError(f"pcol_mask must be [{b}, {dp}], got {tuple(pcol_mask.shape)}")
        for nm, t in (("obs", obs), ("tval", tval), ("row_mask", row_mask)):
            if t.shape != (b, r):
                raise ValueError(f"{nm} must be [{b}, {r}], got {tuple(t.shape)}")
        if not bool(row_mask.bool().any(dim=1).all()):
            raise ValueError("every example must have >= 1 valid row")
        if not bool(pcol_mask.bool().any(dim=1).all()):
            raise ValueError("every example must have >= 1 valid predictor column")

        # Per-cell predictor embedding, masked mean over valid predictor columns → [B, R, H].
        cells = self.pred_cell(P.unsqueeze(-1))                 # [B, R, Dp, H]
        cmask = pcol_mask.bool().view(b, 1, dp, 1)
        s = (cells * cmask).sum(dim=2) / cmask.sum(dim=2).clamp(min=1).to(cells.dtype)

        row_in = torch.cat([s, obs.unsqueeze(-1), tval.unsqueeze(-1)], dim=-1)
        tokens = self.row(row_in)                               # [B, R, H]

        rmask = row_mask.bool()
        obs_rows = rmask & (obs > 0.5)
        mis_rows = rmask & (obs <= 0.5)
        feats = torch.cat([
            _masked_group_stats(tokens, rmask, self.cfg.quantile_levels),
            _masked_group_stats(tokens, obs_rows, self.cfg.quantile_levels),
            _masked_group_stats(tokens, mis_rows, self.cfg.quantile_levels)], dim=-1)
        return self.head(feats).squeeze(-1)                     # [B]
