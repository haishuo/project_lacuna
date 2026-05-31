"""
lacuna.models.composition_head

Dirichlet composition head — the Stage-C readout for the ADR-0007 estimand.

Lacuna's estimand is a *calibrated distribution over a dataset's missingness composition*
`(f_MCAR, f_MAR, f_MNAR)` on the 2-simplex, plus an explicit "can't-tell" mass (ADR-0007). This head
sits on the frozen dataset-level `evidence` vector the encoder already produces and emits a **Dirichlet**
over the composition simplex: `Dir(alpha)`. The Dirichlet IS the posterior — the can't-tell mass and the
example queries ("60% sure ≥70% MAR") are then *queries on it*, not separate outputs (ADR §estimand).

Why a Dirichlet (evidential head). It is the natural distribution over the simplex, and its
concentration carries the model's confidence directly: `alpha = softplus(logits) + 1` (so `alpha_k ≥ 1`,
`alpha0 = Σ alpha_k ≥ K`). The expected composition is `alpha / alpha0`; the **total uncertainty / can't-
tell mass is `K / alpha0`** (Sensoy et al. 2018 "vacuity") — 1 when the head has no evidence (uniform over
the simplex = "can't tell"), →0 as evidence concentrates. This is exactly the explicit can't-tell mass the
ADR asks for, expressed as a query on the Dirichlet rather than a bolted-on scalar.

The head is purely additive (multi-task): it does not touch the existing dataset-level class head, so v1.0
capability is preserved. A deep ensemble is formed by summing the per-model evidence (`alpha - 1`) across
independently-trained heads (`ensemble_alpha`) — the standard evidential pooling that adds *model*
uncertainty on top of each head's *distributional* uncertainty.

Determinism (Coding Bible Rule 6): the module is deterministic; the only stochastic helper
(`prob_region`, a Monte-Carlo simplex-region query) draws through an injected RNGState. Fails loud
(Rule 1) on malformed concentrations.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from lacuna.core.rng import RNGState

_N_CLASSES = 3  # MCAR / MAR / MNAR — the composition simplex


class CompositionHead(nn.Module):
    """Map the dataset-level evidence vector to a Dirichlet over the composition simplex.

    Args:
        evidence_dim: input dimension (the encoder's evidence vector width).
        hidden_dim: optional hidden layer; None = direct linear projection.
        n_classes: composition dimension (3 = MCAR/MAR/MNAR).
        dropout: dropout rate (only used when hidden_dim is set).
        n_extra_features: width of optional explicit per-dataset features concatenated to the
            evidence (e.g. the observable missingness footprint). 0 (default) = evidence only, so
            the original behaviour is unchanged. When > 0, the extra features are BatchNorm-
            standardised (they are heterogeneous in scale) before concatenation. This is the
            composition-level analogue of the column arc's deployable-features path: the encoder's
            pooled evidence misses cross-column structure that the footprint makes explicit.
    """

    def __init__(self, evidence_dim: int, hidden_dim: Optional[int] = 64,
                 n_classes: int = _N_CLASSES, dropout: float = 0.1, n_extra_features: int = 0,
                 n_hidden_layers: int = 1, use_evidence: bool = True):
        super().__init__()
        self.evidence_dim = evidence_dim
        self.n_classes = n_classes
        self.n_extra_features = n_extra_features
        self.use_evidence = use_evidence
        if not use_evidence and n_extra_features <= 0:
            raise ValueError("use_evidence=False requires n_extra_features > 0 (nothing to read otherwise)")
        self.extra_norm = nn.BatchNorm1d(n_extra_features) if n_extra_features > 0 else None
        in_dim = (evidence_dim if use_evidence else 0) + n_extra_features
        if hidden_dim is None or n_hidden_layers < 1:
            self.net = nn.Linear(in_dim, n_classes)
        else:
            # `n_hidden_layers` hidden blocks; default 1 reproduces the original
            # Linear→GELU→Dropout→Linear (net.0/net.3) so saved heads load unchanged.
            layers, d = [], in_dim
            for _ in range(n_hidden_layers):
                layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
                d = hidden_dim
            layers += [nn.Linear(d, n_classes)]
            self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, evidence: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evidence [B, evidence_dim] (+ optional extra [B, n_extra_features]) -> alpha [B, n_classes].

        Returns Dirichlet concentrations alpha_k >= 1 (softplus + 1).
        """
        if self.n_extra_features > 0:
            if extra is None or extra.shape[-1] != self.n_extra_features:
                raise ValueError(f"head expects extra features of width {self.n_extra_features}, "
                                 f"got {None if extra is None else tuple(extra.shape)}")
            extra = self.extra_norm(extra)
            x = torch.cat([evidence, extra], dim=-1) if self.use_evidence else extra
        else:
            x = evidence
        return torch.nn.functional.softplus(self.net(x)) + 1.0


def _check_alpha(alpha: torch.Tensor) -> None:
    if alpha.dim() != 2 or alpha.shape[-1] < 2:
        raise ValueError(f"alpha must be [B, K>=2], got shape {tuple(alpha.shape)}")
    if not torch.isfinite(alpha).all() or bool((alpha <= 0).any()):
        raise ValueError("alpha must be finite and strictly positive")


def composition_mean(alpha: torch.Tensor) -> torch.Tensor:
    """Expected composition E[p] = alpha / alpha0 [B, K] (the point estimate; rows sum to 1)."""
    _check_alpha(alpha)
    return alpha / alpha.sum(dim=-1, keepdim=True)


def cant_tell_mass(alpha: torch.Tensor) -> torch.Tensor:
    """Explicit can't-tell mass = total uncertainty K / alpha0 [B], in (0, 1].

    1.0 when alpha0 = K (no evidence → uniform over the simplex → genuinely can't tell); →0 as the
    head accumulates evidence and the Dirichlet concentrates. This is the ADR-0007 can't-tell mass
    expressed as a query on the Dirichlet (Sensoy 2018 vacuity).
    """
    _check_alpha(alpha)
    return alpha.shape[-1] / alpha.sum(dim=-1)


def ensemble_alpha(alphas: torch.Tensor) -> torch.Tensor:
    """Pool a deep ensemble by summing evidence: alpha_ens = 1 + Σ_m (alpha_m - 1).

    Args:
        alphas: [M, B, K] concentrations from M independently-trained heads.
    Returns:
        [B, K] pooled Dirichlet concentration (adds model uncertainty to each head's vacuity).
    """
    if alphas.dim() != 3:
        raise ValueError(f"alphas must be [M, B, K], got shape {tuple(alphas.shape)}")
    return 1.0 + (alphas - 1.0).sum(dim=0)


def prob_region(alpha: torch.Tensor, predicate: Callable[[np.ndarray], np.ndarray],
                rng: RNGState, *, n_samples: int = 4000) -> torch.Tensor:
    """Monte-Carlo posterior mass that the composition lands in a simplex region [B].

    This evaluates the ADR-0007 example queries ("posterior puts 0.60 of its mass on {f_MAR ≥ 0.7}")
    on the Dirichlet. `predicate` maps a sample array [n, K] (columns = MCAR/MAR/MNAR fractions) to a
    boolean array [n]; the returned per-item value is the fraction of samples satisfying it.

    Deterministic given `rng` (sampled via the injected numpy generator, not the global RNG).
    """
    _check_alpha(alpha)
    a = alpha.detach().cpu().numpy()
    np_rng = rng.numpy_rng
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        samples = np_rng.dirichlet(a[i], size=n_samples)  # [n, K], each row on the simplex
        hits = np.asarray(predicate(samples), dtype=bool)
        if hits.shape != (n_samples,):
            raise ValueError(f"predicate must return a boolean array of shape ({n_samples},), "
                             f"got {hits.shape}")
        out[i] = float(hits.mean())
    return torch.from_numpy(out)
