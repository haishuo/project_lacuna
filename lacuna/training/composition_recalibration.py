"""
lacuna.training.composition_recalibration

Feature-conditional recalibration of the composition posterior (ADR-0007, Stage D-v2).

Stage D's global temperature made the posterior's region statements reliable ON AVERAGE, but it could
not fix the per-instance uncertainty RANKING: the can't-tell mass still anti-correlated with error
(corr ≈ −0.15) — the model was not more uncertain on the datasets it actually got more wrong. A single
scalar cannot, by construction. This module fits a temperature that DEPENDS on the observable footprint,
so the posterior can be spread (uncertain) on datasets whose footprint signals unreliability and sharp
where the footprint signals a clean read.

Why this can't overfit the composition. The recalibrator only scales the Dirichlet concentration —
`alpha_cal = 1 + (alpha-1)/tau(footprint)` — and a temperature **preserves the mean's argmax / ordering**
(it scales evidence uniformly). So it can only change per-dataset CONFIDENCE, never which composition is
predicted. It is a 1-D-per-instance knob: well-regularised by construction.

It is fit by minimising the held-out Dirichlet NLL (a proper score; `composition_loss.dirichlet_nll`):
that objective penalises a confident-AND-wrong posterior sharply, so minimising it drives `tau` UP
(uncertain) exactly on the datasets where the mean is unreliable — IF that unreliability is predictable
from the footprint. (The Bayes-risk expected CE does NOT work here — it over-rewards sharpening-when-
right and collapses to a near-global temperature; this was checked.) Whether the unreliability is
predictable at all is an empirical question the Stage-D-v2 feasibility probe answers first; if it is
not, a global temperature is the best attainable and this module is a documented null.

Determinism (Coding Bible Rule 6): init + training flow through the injected RNGState. Fails loud
(Rule 1) on contract violations.
"""

import torch
import torch.nn as nn

from lacuna.core.rng import RNGState
from lacuna.training.composition_loss import dirichlet_nll


class FeatureTemperature(nn.Module):
    """Map an observable footprint to a per-dataset Dirichlet temperature in [tau_min, tau_max].

    Args:
        n_features: footprint width (e.g. 20).
        hidden: hidden width (0 = linear map).
        tau_min, tau_max: temperature bounds (tau<1 sharpens, tau>1 spreads toward the prior).
    """

    def __init__(self, n_features: int, hidden: int = 16, tau_min: float = 0.5, tau_max: float = 25.0):
        super().__init__()
        if not 0 < tau_min < tau_max:
            raise ValueError(f"require 0 < tau_min < tau_max, got ({tau_min}, {tau_max})")
        self.tau_min, self.tau_max = tau_min, tau_max
        self.norm = nn.BatchNorm1d(n_features)
        if hidden <= 0:
            self.net = nn.Linear(n_features, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Random (not zero) init: a zero-init starts the net as a CONSTANT function and the
                # optimiser settles in the global-temperature basin without ever engaging the
                # features. Xavier lets feature-dependence be explored from step one.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, footprint: torch.Tensor) -> torch.Tensor:
        """Footprint [N, n_features] -> temperature [N] in (tau_min, tau_max)."""
        if footprint.dim() != 2:
            raise ValueError(f"footprint must be [N, n_features], got {tuple(footprint.shape)}")
        logit = self.net(self.norm(footprint)).squeeze(-1)
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(logit)

    def recalibrate(self, alpha: torch.Tensor, footprint: torch.Tensor) -> torch.Tensor:
        """Per-instance temperature scaling: alpha_cal = 1 + (alpha-1)/tau(footprint) [N, K]."""
        if alpha.dim() != 2:
            raise ValueError(f"alpha must be [N, K], got {tuple(alpha.shape)}")
        tau = self.forward(footprint).unsqueeze(-1).clamp(min=1e-6)
        return 1.0 + (alpha - 1.0) / tau


def fit_feature_temperature(
    footprints: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
    rng: RNGState,
    *,
    hidden: int = 16,
    tau_min: float = 0.5,
    tau_max: float = 25.0,
    epochs: int = 600,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
) -> FeatureTemperature:
    """Fit a FeatureTemperature on a calibration split by minimising the Dirichlet NLL (proper score).

    Args:
        footprints: [N, F] observable features the temperature is conditioned on.
        alpha: [N, K] the head's (uncalibrated) Dirichlet concentrations.
        target: [N, K] realised compositions (rows sum to 1).
        rng: explicit RNG (seeds the torch init/training deterministically).
        hidden/tau_min/tau_max: passed to FeatureTemperature.
        epochs/lr/weight_decay: optimiser settings.

    Returns:
        The trained FeatureTemperature (eval mode).

    Raises:
        ValueError: on shape mismatch.
    """
    if not (footprints.dim() == 2 and alpha.dim() == 2 and target.shape == alpha.shape):
        raise ValueError("footprints[N,F], alpha[N,K], target[N,K] required with matching N")
    if footprints.shape[0] != alpha.shape[0]:
        raise ValueError("footprints and alpha must share N")
    torch.manual_seed(rng.seed)
    model = FeatureTemperature(footprints.shape[1], hidden=hidden, tau_min=tau_min, tau_max=tau_max)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        alpha_cal = model.recalibrate(alpha, footprints)
        loss = dirichlet_nll(alpha_cal, target).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model
