"""
lacuna.survey.conditioned_head

Target-conditioned δ-prior model (P2.2b rung 3; audit §4, approved design B).

ONE job: like `DeltaPriorModel`, but the δ-bin head also sees a representation of the SUPPLIED
candidate target column — so the model is told which column's δ to report instead of inferring it
(NORTH-STAR human-parity, §4.8). Head-side conditioning ONLY: tokenization is untouched (4
channels, no target-marker channel); we reuse the encoder's exposed
`token_representations [B, rows, cols, hidden]`, gather the target column's per-row reps, masked-mean
pool over valid rows, and concatenate with the global `evidence` before the head.

The old `LacunaModel` is still NOT instantiated — only `LacunaEncoder` + `DeltaBinHead` exist on
this object, so reconstruction/MoE/3-class/binary/decision never enter the graph.

Determinism (Rule 6): weights init via the shared `init_parameters_` (injected RNG). Fail loud
(Rule 1) on a missing/out-of-range target index.
"""

import torch
import torch.nn as nn

from lacuna.core.rng import RNGState
from lacuna.models.encoder import EncoderConfig, LacunaEncoder

from .delta_bins import NUM_BINS
from .delta_head import DeltaBinHead, init_parameters_
from .distributional_stream import RepECDFPooling

CONDITIONING_METHOD = "head_side_target_token_pool"  # recorded in the manifest


class TargetConditionedDeltaModel(nn.Module):
    """LacunaEncoder + a δ-bin head conditioned on [global evidence ; pooled target-column reps]."""

    def __init__(
        self,
        encoder_config: EncoderConfig,
        head_hidden_dim: int = None,
        dropout: float = 0.1,
        num_bins: int = NUM_BINS,
        n_consequence_features: int = 0,
        rep_ecdf_pooling: bool = False,
        n_shape_probes: int = 4,
    ):
        super().__init__()
        self.encoder = LacunaEncoder(encoder_config)
        self.n_consequence_features = int(n_consequence_features)
        # Across-row target summary: learned order-statistic ECDF pooling (replaces the mean) when
        # enabled, else the legacy masked-mean. The mean averaged the tail away (Proposal B); the
        # ECDF exposes the within-column distributional shape the δ footprint lives in.
        self.rep_ecdf_pooling = bool(rep_ecdf_pooling)
        if self.rep_ecdf_pooling:
            self.rep_pool = RepECDFPooling(encoder_config.hidden_dim, n_probes=n_shape_probes)
            target_summary_dim = self.rep_pool.out_dim
        else:
            target_summary_dim = encoder_config.hidden_dim
        self.cond_dim = encoder_config.evidence_dim + target_summary_dim + self.n_consequence_features
        if self.n_consequence_features > 0:
            self.consequence_norm = nn.LayerNorm(self.n_consequence_features)
        self.head = DeltaBinHead(
            evidence_dim=self.cond_dim, num_bins=num_bins,
            hidden_dim=head_hidden_dim, dropout=dropout,
        )
        self.num_bins = num_bins
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def _gather_target(self, token_repr, target_idx) -> torch.Tensor:
        """Gather the supplied target column's per-row token reps -> [B, R, H] (fail loud on bad idx)."""
        b, r, c, h = token_repr.shape
        if target_idx is None:
            raise ValueError("TargetConditionedDeltaModel requires target_idx (supplied target column)")
        target_idx = target_idx.to(token_repr.device).long()
        if int(target_idx.min()) < 0 or int(target_idx.max()) >= c:
            raise ValueError(f"target_idx out of range [0, {c}); got [{int(target_idx.min())}, {int(target_idx.max())}]")
        gather_idx = target_idx.view(b, 1, 1, 1).expand(b, r, 1, h)
        return token_repr.gather(2, gather_idx).squeeze(2)  # [B, R, H]

    def _pool_target(self, token_repr, row_mask, target_idx) -> torch.Tensor:
        """Across-row target summary: ECDF order-statistics (if enabled) else masked mean -> [B, D]."""
        tgt = self._gather_target(token_repr, target_idx)  # [B, R, H]
        rm = row_mask.to(token_repr.device)
        if self.rep_ecdf_pooling:
            return self.rep_pool(tgt, rm)  # [B, out_dim]
        rmf = rm.unsqueeze(-1).float()  # [B, R, 1]
        return (tgt * rmf).sum(dim=1) / rmf.sum(dim=1).clamp(min=1.0)  # [B, H]

    def forward(self, batch, target_idx, consequence=None) -> torch.Tensor:
        """(TokenBatch, [B] target index, optional [B, F] consequence) -> [B, num_bins] logits."""
        device = next(self.parameters()).device
        batch = batch.to(device)
        enc = self.encoder(
            batch.tokens, batch.row_mask, batch.col_mask, return_intermediates=True
        )
        evidence = enc["evidence"]  # [B, E]
        pooled_target = self._pool_target(enc["token_representations"], batch.row_mask, target_idx)
        parts = [evidence, pooled_target]
        if self.n_consequence_features > 0:
            if consequence is None:
                raise ValueError("model has consequence features enabled but `consequence` is None")
            c = consequence.to(device)
            if c.dim() != 2 or c.shape[1] != self.n_consequence_features:
                raise ValueError(
                    f"consequence must be [B, {self.n_consequence_features}], got {tuple(c.shape)}"
                )
            parts.append(self.consequence_norm(c))
        combined = torch.cat(parts, dim=-1)  # [B, E + H (+ F)]
        return self.head(combined)

    def predict_proba(self, batch, target_idx, consequence=None) -> torch.Tensor:
        logits = self.forward(batch, target_idx, consequence)
        t = self.temperature.clamp(min=1e-6)
        return torch.softmax(logits / t, dim=-1)

    def set_temperature(self, value: float) -> None:
        if not (value > 0.0):
            raise ValueError(f"temperature must be > 0, got {value}")
        self.temperature = torch.tensor(float(value), dtype=torch.float32)


def create_target_conditioned_model(
    *,
    hidden_dim: int = 128,
    evidence_dim: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    max_cols: int = 32,
    dropout: float = 0.1,
    head_hidden_dim: int = None,
    num_bins: int = NUM_BINS,
    n_consequence_features: int = 0,
    rep_ecdf_pooling: bool = False,
    n_shape_probes: int = 4,
    rng: RNGState = None,
) -> TargetConditionedDeltaModel:
    """Build a TargetConditionedDeltaModel; if `rng` is given, deterministically init its weights."""
    cfg = EncoderConfig(
        hidden_dim=hidden_dim, evidence_dim=evidence_dim, n_layers=n_layers,
        n_heads=n_heads, max_cols=max_cols, dropout=dropout,
    )
    model = TargetConditionedDeltaModel(
        encoder_config=cfg, head_hidden_dim=head_hidden_dim, dropout=dropout, num_bins=num_bins,
        n_consequence_features=n_consequence_features,
        rep_ecdf_pooling=rep_ecdf_pooling, n_shape_probes=n_shape_probes,
    )
    if rng is not None:
        init_parameters_(model, rng)
    return model
