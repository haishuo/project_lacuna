"""
lacuna.survey.delta_head

The P2.2 δ-prior model: the validated LacunaEncoder backbone + a NEW ordered-δ-bin head.

ONE job: turn a tokenized batch of (real-X + δ-self-censored) datasets into raw logits over
the 7 ordered δ-bins (PROPOSAL-P2 §2; P2.2 audit §1–§3). The model is cut at the clean seam
`evidence = encoder(...)` (assembly.py:363) and attaches ONLY a small MLP head. It deliberately
does NOT instantiate `LacunaModel`, so the reconstruction heads, the MoE gate, the 3-class
posterior, the feasibility binary head, and `BayesOptimalDecision` are NEVER constructed and
cannot enter the graph, the loss, the gradients, or the parameter count (P2.2 constraint #1).

Ordering over the bins is imposed by the RPS LOSS (see `loss.py`), not by a cumulative-link
architecture — the head is a plain K-logit MLP (audit §2). The calibration temperature is a
non-trained buffer (T=1.0 during training; set post-hoc on val — see `loss.fit_temperature`).

Determinism (Rule 6): `init_parameters_` re-initializes EVERY parameter from an injected
RNGState's torch generator (in-place `uniform_`/`normal_` accept `generator=`), so a given seed
reproduces identical weights WITHOUT ever touching the global torch seed.

Contract (Rule 1): construction fails loud on bad dims; `num_bins < 2` is rejected.
"""

import math

import torch
import torch.nn as nn

from lacuna.core.rng import RNGState
from lacuna.models.encoder import EncoderConfig, LacunaEncoder

from .delta_bins import NUM_BINS


class DeltaBinHead(nn.Module):
    """MLP head mapping an evidence vector to ordered δ-bin logits.

    Mirrors the `lacuna.models.heads.GeneratorHead` idiom: a single hidden layer
    (GELU + dropout) then a linear projection to `num_bins` logits. No softmax inside —
    the loss consumes logits; `DeltaPriorModel.predict_proba` applies temperature + softmax.
    """

    def __init__(
        self,
        evidence_dim: int,
        num_bins: int = NUM_BINS,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if evidence_dim < 1:
            raise ValueError(f"evidence_dim must be >= 1, got {evidence_dim}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2 for an ordered head, got {num_bins}")
        self.evidence_dim = evidence_dim
        self.num_bins = num_bins
        hidden = hidden_dim if hidden_dim is not None else evidence_dim
        self.net = nn.Sequential(
            nn.Linear(evidence_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_bins),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """[B, evidence_dim] -> [B, num_bins] raw logits."""
        if evidence.dim() != 2 or evidence.shape[1] != self.evidence_dim:
            raise ValueError(
                f"evidence must be [B, {self.evidence_dim}], got {tuple(evidence.shape)}"
            )
        return self.net(evidence)


class DeltaPriorModel(nn.Module):
    """LacunaEncoder (backbone) + DeltaBinHead (new). Returns δ-bin logits for a TokenBatch.

    The encoder is called WITHOUT intermediates (the δ head needs only `evidence`). The
    temperature buffer is 1.0 during training and overwritten post-hoc for calibrated
    `predict_proba`. No v1.0 downstream module exists on this object (assert in tests).
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        head_hidden_dim: int = None,
        dropout: float = 0.1,
        num_bins: int = NUM_BINS,
    ):
        super().__init__()
        self.encoder = LacunaEncoder(encoder_config)
        self.head = DeltaBinHead(
            evidence_dim=encoder_config.evidence_dim,
            num_bins=num_bins,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        self.num_bins = num_bins
        # Calibration temperature (post-hoc); NOT a trained parameter — a buffer.
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, batch, target_idx=None) -> torch.Tensor:
        """TokenBatch -> [B, num_bins] raw logits (temperature NOT applied).

        `target_idx` is accepted for a uniform interface with the target-conditioned model and is
        IGNORED here — the global-evidence head conditions on the whole dataset, not a column.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        evidence = self.encoder(
            batch.tokens, batch.row_mask, batch.col_mask, return_intermediates=False
        )
        return self.head(evidence)

    def predict_proba(self, batch, target_idx=None) -> torch.Tensor:
        """Calibrated probabilities: softmax(logits / temperature)."""
        logits = self.forward(batch, target_idx)
        t = self.temperature.clamp(min=1e-6)
        return torch.softmax(logits / t, dim=-1)

    def set_temperature(self, value: float) -> None:
        """Set the post-hoc calibration temperature (must be > 0)."""
        if not (value > 0.0):
            raise ValueError(f"temperature must be > 0, got {value}")
        self.temperature = torch.tensor(float(value), dtype=torch.float32)


def init_parameters_(model: nn.Module, rng: RNGState) -> None:
    """Deterministically (re)initialize every parameter from the injected RNG.

    Xavier-uniform for Linear weights, N(0,1) for Embedding weights, ones/zeros for
    LayerNorm, zeros for biases. Uses the RNGState's torch generator so the result is
    reproducible for a given seed and NEVER mutates the global torch RNG (Rule 6).
    """
    gen = rng.torch_generator
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_out, fan_in = m.weight.shape
                a = math.sqrt(6.0 / (fan_in + fan_out))
                m.weight.uniform_(-a, a, generator=gen)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.normal_(0.0, 1.0, generator=gen)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.zero_()


def count_parameters(model: nn.Module) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in model.parameters())


def assert_fresh_and_trainable(model: nn.Module) -> int:
    """Fail loud unless EVERY parameter is trainable; return the total count (audit §3)."""
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_param != n_train:
        raise ValueError(f"not all layers trainable: {n_train}/{n_param}")
    return n_param


def create_delta_prior_model(
    *,
    hidden_dim: int = 128,
    evidence_dim: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    max_cols: int = 32,
    dropout: float = 0.1,
    head_hidden_dim: int = None,
    num_bins: int = NUM_BINS,
    rng: RNGState = None,
) -> DeltaPriorModel:
    """Build a DeltaPriorModel; if `rng` is given, deterministically initialize its weights.

    `max_cols` must be >= the widest dataset's column count (survey_bfi d=28 -> 32 base config).
    """
    cfg = EncoderConfig(
        hidden_dim=hidden_dim,
        evidence_dim=evidence_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_cols=max_cols,
        dropout=dropout,
    )
    model = DeltaPriorModel(
        encoder_config=cfg,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        num_bins=num_bins,
    )
    if rng is not None:
        init_parameters_(model, rng)
    return model
