"""
lacuna.survey.column_phi

The per-column distribution encoder φ — the Level-1 δ spine (Stage-1 spec §1.1; MASTER §5).

ONE job: map the observed values of one column (a variable-length, row-permutation-invariant set)
to a fixed-width distribution embedding that PRESERVES order statistics — the substrate Stage 0
proved recovers the signal the BERT backbone loses. Per-value MLP φ → masked-quantile pooling
(reuses the tested `distributional_stream.masked_quantile_pool` on RAW values; 12 quantiles + max)
→ small MLP → column embedding e ∈ ℝ^{e_col}. No BERT, no row-wise attention, no mechanism-specific
detector.

Determinism (Rule 6): weights (re)initialized from an injected RNGState via the shared
`delta_head.init_parameters_`. Fail loud (Rule 1) on bad dims / wrong value-shape.
"""

import numpy as np
import torch
import torch.nn as nn

from lacuna.core.rng import RNGState

from .distributional_stream import masked_quantile_pool

# 12-quantile grid (PI decision 2026-06-06: "12 quantiles + max"); deterministic, recorded in manifest.
QUANTILE_LEVELS = tuple(np.round(np.linspace(0.02, 0.98, 12), 4).tolist())
N_QUANTILES = len(QUANTILE_LEVELS)


class ColumnPhi(nn.Module):
    """Observed-value set → distribution embedding via per-value MLP + quantile pooling + MLP."""

    def __init__(self, m: int = 16, e_col: int = 32, hidden: int = 48, dropout: float = 0.1,
                 levels=QUANTILE_LEVELS):
        super().__init__()
        if m < 1:
            raise ValueError(f"m (per-value width) must be >= 1, got {m}")
        if e_col < 1:
            raise ValueError(f"e_col must be >= 1, got {e_col}")
        self.m = int(m)
        self.e_col = int(e_col)
        self.levels = tuple(float(x) for x in levels)
        self.h = nn.Sequential(nn.Linear(1, m), nn.GELU(), nn.Linear(m, m), nn.GELU())
        self.rho = nn.Sequential(
            nn.Linear(m * (len(self.levels) + 1), hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, e_col),
        )

    def forward(self, values: torch.Tensor, value_mask: torch.Tensor) -> torch.Tensor:
        """(values [B, R, 1], value_mask [B, R]) -> column embedding [B, e_col]."""
        if values.dim() != 3 or values.shape[2] != 1:
            raise ValueError(f"values must be [B, R, 1], got {tuple(values.shape)}")
        probes = self.h(values)                                  # [B, R, m]
        stats = masked_quantile_pool(probes, value_mask, self.levels)  # [B, m, Q+1]
        return self.rho(stats.reshape(stats.shape[0], -1))       # [B, e_col]

    def schema(self) -> dict:
        """Manifest-ready description of the frozen φ scheme."""
        return {"kind": "column_phi", "m": self.m, "e_col": self.e_col,
                "quantile_levels": list(self.levels), "plus_max": True}


def create_column_phi(*, m: int = 16, e_col: int = 32, hidden: int = 48, dropout: float = 0.1,
                      rng: RNGState = None) -> ColumnPhi:
    """Build a ColumnPhi; if `rng` is given, deterministically initialize its weights."""
    from .delta_head import init_parameters_
    model = ColumnPhi(m=m, e_col=e_col, hidden=hidden, dropout=dropout)
    if rng is not None:
        init_parameters_(model, rng)
    return model
