"""
lacuna.survey.feature_only_head

Features-only δ-prior model (PROPOSAL-P2.2c coarse audit — the MAIN coarse-curriculum model).

ONE job: a learned, calibrated δ-bin head on the FIXED consequence-feature vector ALONE — no
encoder. This isolates the scientific question of the coarse rung: *can a learned model recover the
coarse, transferable LOD signal that the LR baseline already demonstrates out-of-family?* It is the
honest minimal learned δ-prior on the observable features, directly comparable to the LR baseline.

`forward(batch, target_idx, consequence)` ignores `batch`/`target_idx` (uniform interface with the
encoder models) and uses ONLY `consequence`. No v1.0 heads, no encoder. Deterministic init via the
shared `init_parameters_` (injected RNG).
"""

import torch
import torch.nn as nn

from lacuna.core.rng import RNGState

from .delta_bins import NUM_BINS
from .delta_head import DeltaBinHead, init_parameters_


class FeatureOnlyDeltaModel(nn.Module):
    """LayerNorm(features) -> DeltaBinHead. A learned δ-prior on the fixed consequence vector only."""

    def __init__(self, n_features: int, num_bins: int = NUM_BINS, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        self.n_features = int(n_features)
        self.num_bins = int(num_bins)
        self.norm = nn.LayerNorm(self.n_features)
        self.head = DeltaBinHead(evidence_dim=self.n_features, num_bins=num_bins,
                                 hidden_dim=hidden_dim, dropout=dropout)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, batch=None, target_idx=None, consequence=None) -> torch.Tensor:
        if consequence is None:
            raise ValueError("FeatureOnlyDeltaModel requires `consequence` features")
        device = next(self.parameters()).device
        c = consequence.to(device)
        if c.dim() != 2 or c.shape[1] != self.n_features:
            raise ValueError(f"consequence must be [B, {self.n_features}], got {tuple(c.shape)}")
        return self.head(self.norm(c))

    def predict_proba(self, batch=None, target_idx=None, consequence=None) -> torch.Tensor:
        logits = self.forward(batch, target_idx, consequence)
        return torch.softmax(logits / self.temperature.clamp(min=1e-6), dim=-1)

    def set_temperature(self, value: float) -> None:
        if not (value > 0.0):
            raise ValueError(f"temperature must be > 0, got {value}")
        self.temperature = torch.tensor(float(value), dtype=torch.float32)


def create_feature_only_model(
    *, n_features: int, num_bins: int = NUM_BINS, hidden_dim: int = None,
    dropout: float = 0.1, rng: RNGState = None,
) -> FeatureOnlyDeltaModel:
    """Build a FeatureOnlyDeltaModel; if `rng` is given, deterministically init its weights."""
    model = FeatureOnlyDeltaModel(n_features=n_features, num_bins=num_bins,
                                  hidden_dim=hidden_dim, dropout=dropout)
    if rng is not None:
        init_parameters_(model, rng)
    return model
