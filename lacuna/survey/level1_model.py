"""
lacuna.survey.level1_model

The Level-1 δ-prior model: column-primary φ spine + ordered δ-bin head (Stage-1 spec §1.4; MASTER §5).

ONE job: turn a `column_batching.ColumnBatch` into raw logits over the ordered δ-bins, using ONLY the
per-column distribution encoder φ (`column_phi`) and a `DeltaBinHead`. The BERT `LacunaEncoder` is
NOT the spine and is never constructed; no v1.0 reconstruction/MoE/3-class/decision head exists on
this object (asserted in tests). This is the architecture-validation core: reproduce the Stage-0
column-primary signal inside the governed pipeline (Stage-1 M1).

A `fusion` slot for the Stage-2 mask-topology stream is intentionally NOT added here (Stage 1 is the
φ spine alone). RPS imposes the bin ordering; temperature is a post-hoc buffer (not trained).
Determinism (Rule 6) via the shared injected-RNG init.
"""

import torch
import torch.nn as nn

from lacuna.core.rng import RNGState

from .column_phi import create_column_phi
from .delta_bins import NUM_BINS
from .delta_head import DeltaBinHead, init_parameters_


class Level1Model(nn.Module):
    """ColumnPhi (spine) + DeltaBinHead. Returns δ-bin logits for a ColumnBatch."""

    def __init__(self, *, m: int = 16, e_col: int = 32, phi_hidden: int = 48,
                 head_hidden_dim: int = None, dropout: float = 0.1, num_bins: int = NUM_BINS):
        super().__init__()
        self.phi = create_column_phi(m=m, e_col=e_col, hidden=phi_hidden, dropout=dropout, rng=None)
        self.head = DeltaBinHead(evidence_dim=e_col, num_bins=num_bins,
                                 hidden_dim=head_hidden_dim, dropout=dropout)
        self.num_bins = int(num_bins)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, batch) -> torch.Tensor:
        """ColumnBatch -> [B, num_bins] raw logits (temperature NOT applied)."""
        device = next(self.parameters()).device
        e = self.phi(batch.target_values.to(device), batch.value_mask.to(device))
        return self.head(e)

    def predict_proba(self, batch) -> torch.Tensor:
        """Calibrated probabilities: softmax(logits / temperature)."""
        logits = self.forward(batch)
        return torch.softmax(logits / self.temperature.clamp(min=1e-6), dim=-1)

    def set_temperature(self, value: float) -> None:
        if not (value > 0.0):
            raise ValueError(f"temperature must be > 0, got {value}")
        self.temperature = torch.tensor(float(value), dtype=torch.float32)


def create_level1_model(*, m: int = 16, e_col: int = 32, phi_hidden: int = 48,
                        head_hidden_dim: int = None, dropout: float = 0.1,
                        num_bins: int = NUM_BINS, rng: RNGState = None) -> Level1Model:
    """Build a Level1Model; if `rng` is given, deterministically initialize every parameter."""
    model = Level1Model(m=m, e_col=e_col, phi_hidden=phi_hidden, head_hidden_dim=head_hidden_dim,
                        dropout=dropout, num_bins=num_bins)
    if rng is not None:
        init_parameters_(model, rng)
    return model
