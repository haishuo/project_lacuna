"""
lacuna/survey/t2_model.py

The T2 network of record (PREREGISTRATION-network-load-bearing-review §3; MASTER §5): the design-of-
record Level-1 φ-spine COMPLETED with its Stage-2 mask-topology stream and the joint family + δ heads.
ONE forward pass per (table, target column) → mechanism-family logits (4) + δ-bin logit (1).

  e_target = ColumnPhi(target observed values)            # the per-column distribution footprint
  e_topo   = MaskTopologyStream(table values+mask, target) # the cross-column / gate / block structure
  trunk    = MLP([e_target ; e_topo])
  family   = Linear(trunk, 4)   δ = Linear(trunk, 1)

This is exactly the φ-per-column + mask-topology-encoder + fusion + family/δ-heads architecture the
MASTER always specified; Level1Model deliberately omitted the fusion slot, which T2 fills. One fixed
config (no tuning). Determinism via the shared injected-RNG init; no hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lacuna.core.rng import RNGState
from .column_phi import create_column_phi
from .delta_head import init_parameters_
from .mask_topology_stream import MaskTopologyStream
from .skip_logic_generator import FAMILIES


@dataclass(frozen=True)
class T2ModelConfig:
    m: int = 16
    e_col: int = 32
    phi_hidden: int = 48
    topo_cell_hidden: int = 24
    topo_out: int = 48
    trunk_hidden: int = 96
    dropout: float = 0.1


class T2Model(nn.Module):
    """φ-spine + mask-topology stream + fusion + joint family/δ heads."""

    def __init__(self, cfg: T2ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or T2ModelConfig()
        self.phi = create_column_phi(m=self.cfg.m, e_col=self.cfg.e_col,
                                     hidden=self.cfg.phi_hidden, dropout=self.cfg.dropout, rng=None)
        self.topo = MaskTopologyStream(cell_hidden=self.cfg.topo_cell_hidden,
                                       out_dim=self.cfg.topo_out, dropout=self.cfg.dropout)
        self.trunk = nn.Sequential(
            nn.Linear(self.cfg.e_col + self.cfg.topo_out, self.cfg.trunk_hidden), nn.ReLU(),
            nn.Dropout(self.cfg.dropout))
        self.family_head = nn.Linear(self.cfg.trunk_hidden, len(FAMILIES))
        self.delta_head = nn.Linear(self.cfg.trunk_hidden, 1)

    def forward(self, Xv: torch.Tensor, R: torch.Tensor, row_mask: torch.Tensor,
                pcol_mask: torch.Tensor, tij: torch.Tensor, tval: torch.Tensor):
        """Args:
            Xv, R, row_mask, pcol_mask, tij: as in MaskTopologyStream.forward.
            tval: [B, N] standardized target value where observed, 0 where missing/pad.
        Returns:
            (family_logits [B, 4], delta_logit [B]).
        """
        e_topo = self.topo(Xv, R, row_mask, pcol_mask, tij)
        e_target = self.phi(tval.unsqueeze(-1), (tij > 0.5) & row_mask.bool())
        trunk = self.trunk(torch.cat([e_target, e_topo], dim=-1))
        return self.family_head(trunk), self.delta_head(trunk).squeeze(-1)


def create_t2_model(cfg: T2ModelConfig | None = None, rng: RNGState | None = None) -> T2Model:
    """Build a T2Model; if `rng` is given, deterministically initialize every parameter."""
    model = T2Model(cfg)
    if rng is not None:
        init_parameters_(model, rng)
    return model
