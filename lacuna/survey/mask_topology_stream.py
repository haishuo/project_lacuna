"""
lacuna/survey/mask_topology_stream.py

The T2 Stage-2 mask-topology stream (PREREGISTRATION-network-load-bearing-review §3; MASTER §5 —
"φ per-column + mask-topology encoder + fusion"). Given a whole table's values + observed-mask and a
TARGET column, it learns a per-column embedding of the CROSS-COLUMN structure that distinguishes the
mechanism families — predictor-driven / skip (target missingness tracks another column's VALUES or
MASK) from self-driven (no cross-column dependence) from null.

It earns the structure from raw values+mask (NOT hand-fed correlation statistics — that is the feature
arm; the network must compute its own): per predictor cell it sees [value, predictor-observed,
target-observed]; it pools each predictor column's cells SEPARATELY over the target-observed and
target-missing row groups, so a predictor whose value/mask distribution shifts with the target's
missingness (the MAR/skip footprint) is exposed; then it pools permutation-invariantly over predictor
columns (variable width). Determinism (Rule 6); no hidden state (Rule 5).
"""

from __future__ import annotations

import torch
from torch import nn


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `tokens` [B, N, D, H] over rows N where `mask` [B, N, D] is True; 0 for empty."""
    m = mask.unsqueeze(-1).to(tokens.dtype)                 # [B, N, D, 1]
    s = (tokens * m).sum(dim=1)                             # [B, D, H]
    c = m.sum(dim=1).clamp(min=1.0)
    out = s / c
    empty = (mask.sum(dim=1) == 0).unsqueeze(-1)            # [B, D, 1]
    return out.masked_fill(empty, 0.0)


class MaskTopologyStream(nn.Module):
    """Whole-table (values, mask, target column) → per-target cross-column embedding."""

    def __init__(self, cell_hidden: int = 24, out_dim: int = 48, dropout: float = 0.1):
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(3, cell_hidden), nn.ReLU(),
                                  nn.Linear(cell_hidden, cell_hidden))
        # per-predictor embedding = [all, target-observed, target-missing] group means → mix
        self.col = nn.Sequential(nn.Linear(3 * cell_hidden, out_dim), nn.ReLU())
        self.proj = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, Xv: torch.Tensor, R: torch.Tensor, row_mask: torch.Tensor,
                pcol_mask: torch.Tensor, tij: torch.Tensor) -> torch.Tensor:
        """Args:
            Xv:        [B, N, D] standardized values, 0 where missing/pad.
            R:         [B, N, D] 1.0 observed, 0.0 missing/pad.
            row_mask:  [B, N] bool — valid rows.
            pcol_mask: [B, D] bool — valid PREDICTOR columns (target column excluded).
            tij:       [B, N] 1.0 = target observed in that row, 0.0 = target missing.
        Returns:
            [B, out_dim] per-target cross-column embedding.
        """
        b, n, d = Xv.shape
        if R.shape != (b, n, d):
            raise ValueError(f"R must match Xv {tuple(Xv.shape)}, got {tuple(R.shape)}")
        if pcol_mask.shape != (b, d) or row_mask.shape != (b, n) or tij.shape != (b, n):
            raise ValueError("mask/tij shapes inconsistent with Xv")
        if not bool(pcol_mask.bool().any(dim=1).all()):
            raise ValueError("every example needs >= 1 valid predictor column")

        tij_nd = tij.unsqueeze(-1).expand(b, n, d)
        cells = self.cell(torch.stack([Xv, R, tij_nd], dim=-1))      # [B, N, D, H]

        rmask = row_mask.unsqueeze(-1).expand(b, n, d)               # [B, N, D]
        obs = rmask & (tij_nd > 0.5)
        mis = rmask & (tij_nd <= 0.5)
        col_feat = torch.cat([_masked_mean(cells, rmask),
                              _masked_mean(cells, obs),
                              _masked_mean(cells, mis)], dim=-1)      # [B, D, 3H]
        col_emb = self.col(col_feat)                                 # [B, D, out_dim]

        pm = pcol_mask.unsqueeze(-1).to(col_emb.dtype)               # [B, D, 1]
        mean = (col_emb * pm).sum(dim=1) / pm.sum(dim=1).clamp(min=1.0)
        neg_inf = torch.finfo(col_emb.dtype).min
        mx = col_emb.masked_fill(~pcol_mask.unsqueeze(-1), neg_inf).max(dim=1).values
        return self.proj(torch.cat([mean, mx], dim=-1))             # [B, out_dim]
