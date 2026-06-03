"""
lacuna.feasibility.xmodel

X-model interface for the feasibility oracle.

The analytic Bayes oracle must integrate the censored target value against
p(z_t | z_p) and must sample observed data under each hypothesis. This module
supplies that conditional and the predictor marginal. The X-model is the ONLY
assumption entering the real-X oracle; it is therefore always recorded in the run
manifest (`descriptor`).

Two regimes (charter-honest, §4.9 / PROPOSAL §5–6):
  - EXACT synthetic X: `ConditionalGaussian.synthetic(rho)` — standard bivariate
    normal with correlation rho. No fit, no assumption. This is the clean
    mathematical ceiling for the mechanism family.
  - FITTED real X: `ConditionalGaussian.fit(z_p, z_t)` — a transparent
    conditional-Gaussian fit on the z-scored survey columns. The assumption is the
    Gaussianity of (z_p, z_t); it is reported, never hidden.

Pluggable by design: any conditional model (copula, nonparametric, learned) can be
added behind `XModel` later. ConditionalGaussian ships now (PROPOSAL §5 decision).
"""

from abc import ABC, abstractmethod
import math
from typing import Dict, Tuple

import torch

from lacuna.core.rng import RNGState

_LOG_2PI = math.log(2.0 * math.pi)


class XModel(ABC):
    """Conditional model of the target given the predictor, both in z-scored space."""

    @abstractmethod
    def conditional_mean_var(self, z_p: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Return (mean[z_t | z_p] as a tensor like z_p, Var[z_t | z_p] as a scalar)."""

    @abstractmethod
    def conditional_logpdf(self, z_t: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """log p(z_t | z_p), elementwise."""

    @abstractmethod
    def sample_predictor(self, n: int, rng: RNGState) -> torch.Tensor:
        """Draw z_p from the predictor marginal."""

    @abstractmethod
    def sample_conditional(self, z_p: torch.Tensor, rng: RNGState) -> torch.Tensor:
        """Draw z_t ~ p(z_t | z_p)."""

    @property
    @abstractmethod
    def descriptor(self) -> Dict:
        """Manifest-ready description of this X-model and its assumptions."""


class ConditionalGaussian(XModel):
    """Bivariate-Gaussian X-model: z_t | z_p is Gaussian with constant conditional variance.

    Parameterized by the marginal means/variances and the cross-covariance. The
    conditional is  z_t | z_p ~ N(mu_t + slope*(z_p - mu_p),  var_t - cov^2/var_p).
    """

    def __init__(
        self,
        mu_p: float,
        mu_t: float,
        var_p: float,
        var_t: float,
        cov_pt: float,
        *,
        fitted: bool,
    ):
        if var_p <= 0.0 or var_t <= 0.0:
            raise ValueError(f"variances must be positive (var_p={var_p}, var_t={var_t})")
        cond_var = var_t - (cov_pt * cov_pt) / var_p
        if cond_var <= 0.0:
            raise ValueError(
                f"degenerate conditional variance {cond_var:.4g} "
                f"(|cov| too large vs var_p*var_t); cannot form conditional Gaussian"
            )
        self.mu_p = float(mu_p)
        self.mu_t = float(mu_t)
        self.var_p = float(var_p)
        self.var_t = float(var_t)
        self.cov_pt = float(cov_pt)
        self.fitted = bool(fitted)
        self._slope = float(cov_pt / var_p)
        self._cond_var = float(cond_var)

    @classmethod
    def synthetic(cls, rho: float) -> "ConditionalGaussian":
        """Exact standard bivariate normal with correlation rho (no fit, no assumption)."""
        if not (-1.0 < rho < 1.0):
            raise ValueError(f"rho must be in (-1, 1), got {rho}")
        return cls(0.0, 0.0, 1.0, 1.0, rho, fitted=False)

    @classmethod
    def fit(cls, z_p: torch.Tensor, z_t: torch.Tensor) -> "ConditionalGaussian":
        """Fit a conditional Gaussian to z-scored real columns (the stated assumption)."""
        if z_p.shape != z_t.shape or z_p.dim() != 1:
            raise ValueError("z_p and z_t must be 1D tensors of equal length")
        if z_p.numel() < 3:
            raise ValueError("need >= 3 rows to fit a conditional Gaussian")
        mu_p = float(z_p.mean().item())
        mu_t = float(z_t.mean().item())
        var_p = float(z_p.var(unbiased=False).item())
        var_t = float(z_t.var(unbiased=False).item())
        cov_pt = float(((z_p - mu_p) * (z_t - mu_t)).mean().item())
        return cls(mu_p, mu_t, var_p, var_t, cov_pt, fitted=True)

    def conditional_mean_var(self, z_p: torch.Tensor) -> Tuple[torch.Tensor, float]:
        mean = self.mu_t + self._slope * (z_p - self.mu_p)
        return mean, self._cond_var

    def conditional_logpdf(self, z_t: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        mean, v = self.conditional_mean_var(z_p)
        return -0.5 * (((z_t - mean) ** 2) / v + math.log(v) + _LOG_2PI)

    def sample_predictor(self, n: int, rng: RNGState) -> torch.Tensor:
        return rng.randn(n) * math.sqrt(self.var_p) + self.mu_p

    def sample_conditional(self, z_p: torch.Tensor, rng: RNGState) -> torch.Tensor:
        mean, v = self.conditional_mean_var(z_p)
        return mean + rng.randn(z_p.numel()) * math.sqrt(v)

    @property
    def descriptor(self) -> Dict:
        return {
            "type": "ConditionalGaussian",
            "fitted": self.fitted,
            "exact_synthetic": (not self.fitted),
            "mu_p": self.mu_p,
            "mu_t": self.mu_t,
            "var_p": self.var_p,
            "var_t": self.var_t,
            "cov_pt": self.cov_pt,
            "conditional_var": self._cond_var,
        }
