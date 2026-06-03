"""
lacuna.feasibility.oracle

Analytic Bayes-optimal oracle for the self-censoring feasibility probe.

This is the ONLY object the charter (North Star §4.9) permits to be called a
"ceiling", and only under the stated X-model. It is NOT a learned estimator and NOT
a baseline: it is the Bayes-optimal discriminator on OBSERVED data, computed from the
KNOWN generative model. By Neyman–Pearson no estimator on the observed data can beat
it, so a negative here (Bayes error ≈ 0.5 at matched rate across the realistic δ
range) is an unconfounded, assumption-stated kill.

Two hypotheses, each INDEPENDENTLY rate-matched via beta0 (so rate is not a tell):
  H0 : delta = 0   (MAR — missingness depends only on the observed predictor)
  H1 : delta = δ*  (MNAR — missingness depends on the column's own value)

Per-row observed-data log-likelihood-ratio (predictor marginal and, for observed
rows, p(z_t|z_p) cancel between the hypotheses):
  target observed (z_t seen):  log(1-σ(η1)) - log(1-σ(η0))   = softplus(η0) - softplus(η1)
  target missing:              log m1(z_p) - log m0(z_p),
        m_h(z_p) = E_{z_t|z_p}[ σ(β0_h + β1 z_p + β2_h z_t) ]   (Gauss–Hermite quadrature)

The optimal test is the (Bayes-optimal) likelihood-ratio test built from the KNOWN
generative model. Its n-sample Bayes error is a THEORETICAL quantity; we report a
Monte-Carlo ESTIMATE of it (with a standard error and 95% CI), obtained by drawing
observed data from the X-model under each hypothesis and evaluating the known LLR. This
is NOT a *learned* estimate — no instrument is fitted, the LLR is the exact known one —
but it is still a numerical MC estimate, not a closed form. The Gauss–Hermite term and
the LLR/error arithmetic are done in float64 for numerical headroom (the censoring
logits and KL/Chernoff terms are sensitive); the upstream data-generation sampling is
float32 and is irrelevant to this precision since the likelihood is re-evaluated exactly.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lacuna.core.rng import RNGState
from .delta_generator import SelfCensorParams, _bisect_increasing
from .xmodel import XModel

_LOG_EPS = 1e-12
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def gauss_hermite(n_nodes: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gauss–Hermite nodes/weights (float32 tensors) for ∫ e^{-x^2} f(x) dx ≈ Σ w f(x)."""
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    return (
        torch.tensor(nodes, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
    )


def missing_prob(
    z_p: torch.Tensor,
    params: SelfCensorParams,
    xmodel: XModel,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """m(z_p) = E_{z_t|z_p}[ σ(β0 + β1 z_p + β2 z_t) ], via Gauss–Hermite quadrature.

    Exact (to quadrature order) for a Gaussian conditional; with β2==0 it reduces to
    σ(β0 + β1 z_p) regardless of the X-model (no integral needed) — a tight unit test.
    Computed in float64 (inputs are upcast) for numerical headroom.
    """
    z_p = z_p.to(torch.float64)
    nodes = nodes.to(torch.float64)
    weights = weights.to(torch.float64)
    mean, var = xmodel.conditional_mean_var(z_p)  # mean: [n], var: scalar
    scale = math.sqrt(2.0 * var)
    z_t_grid = mean.unsqueeze(1) + scale * nodes.unsqueeze(0)  # [n, K]
    eta = params.beta0 + params.beta1 * z_p.unsqueeze(1) + params.beta2 * z_t_grid  # [n, K]
    sig = torch.sigmoid(eta)
    return _INV_SQRT_PI * (sig * weights.unsqueeze(0)).sum(dim=1)  # [n]


def llr_rows(
    z_p: torch.Tensor,
    z_t: torch.Tensor,
    observed: torch.Tensor,
    params_h0: SelfCensorParams,
    params_h1: SelfCensorParams,
    xmodel: XModel,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-row observed-data log-likelihood-ratio (H1 over H0). `observed` is the bool mask.

    β1 MAY differ between the hypotheses: the X-model terms p(z_p) and p(z_t|z_p) are identical
    under H0 and H1 (only the missingness mechanism differs), so they cancel in the ratio for any
    β1. Equal-β1 is the point-null special case; the β1′-profiled MAR null uses differing β1.
    """
    z_p = z_p.to(torch.float64)
    z_t = z_t.to(torch.float64)
    eta0 = params_h0.beta0 + params_h0.beta1 * z_p + params_h0.beta2 * z_t
    eta1 = params_h1.beta0 + params_h1.beta1 * z_p + params_h1.beta2 * z_t
    # log(1 - σ(η)) = -softplus(η)
    obs_llr = F.softplus(eta0) - F.softplus(eta1)
    m0 = missing_prob(z_p, params_h0, xmodel, nodes, weights).clamp_min(_LOG_EPS)
    m1 = missing_prob(z_p, params_h1, xmodel, nodes, weights).clamp_min(_LOG_EPS)
    mis_llr = torch.log(m1) - torch.log(m0)
    return torch.where(observed, obs_llr, mis_llr)


def population_missing_rate(
    xmodel: XModel,
    params: SelfCensorParams,
    z_p_sample: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Population marginal missing rate E_{z_p}[ m(z_p) ] over a fixed predictor sample."""
    return float(missing_prob(z_p_sample, params, xmodel, nodes, weights).mean().item())


def solve_beta0_population(
    xmodel: XModel,
    beta1: float,
    beta2: float,
    target_rate: float,
    rng: RNGState,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    n_sample: int = 20000,
    bracket: Tuple[float, float] = (-50.0, 50.0),
) -> SelfCensorParams:
    """Solve beta0 so the POPULATION marginal rate (under the X-model) equals target_rate.

    Used for the oracle, where 'matched rate' means matched in expectation under the
    X-model (not on a particular finite sample). Deterministic given rng seed.
    """
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    z_p_sample = xmodel.sample_predictor(n_sample, rng)

    def f(b0: float) -> float:
        p = SelfCensorParams(beta0=b0, beta1=beta1, beta2=beta2)
        return population_missing_rate(xmodel, p, z_p_sample, nodes, weights) - target_rate

    beta0 = _bisect_increasing(f, bracket[0], bracket[1])
    return SelfCensorParams(beta0=beta0, beta1=beta1, beta2=beta2)


def _simulate_llr(
    true_params: SelfCensorParams,
    params_h0: SelfCensorParams,
    params_h1: SelfCensorParams,
    xmodel: XModel,
    n_rows: int,
    rng: RNGState,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Draw `n_rows` observed-data rows under `true_params` and return their per-row LLR."""
    z_p = xmodel.sample_predictor(n_rows, rng)
    z_t = xmodel.sample_conditional(z_p, rng)
    eta_true = true_params.beta0 + true_params.beta1 * z_p + true_params.beta2 * z_t
    p_missing = torch.sigmoid(eta_true)
    observed = rng.rand(n_rows) >= p_missing
    return llr_rows(z_p, z_t, observed, params_h0, params_h1, xmodel, nodes, weights)


def per_row_kl(
    params_h0: SelfCensorParams,
    params_h1: SelfCensorParams,
    xmodel: XModel,
    rng: RNGState,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    n_sample: int = 50000,
) -> Tuple[float, float]:
    """Per-row KL(H1‖H0) = E_{H1}[LLR] and KL(H0‖H1) = E_{H0}[-LLR]. Non-negative in expectation."""
    llr_h1 = _simulate_llr(params_h1, params_h0, params_h1, xmodel, n_sample, rng.spawn(), nodes, weights)
    llr_h0 = _simulate_llr(params_h0, params_h0, params_h1, xmodel, n_sample, rng.spawn(), nodes, weights)
    return float(llr_h1.mean().item()), float((-llr_h0).mean().item())


def bayes_error_nsample(
    params_h0: SelfCensorParams,
    params_h1: SelfCensorParams,
    xmodel: XModel,
    n: int,
    rng: RNGState,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    n_mc: int = 4000,
    max_rows_per_chunk: int = 250_000,
) -> Dict[str, float]:
    """Monte-Carlo estimate of the n-sample Bayes error of the optimal LLR test (equal priors).

    Decide H1 iff Σ_i LLR_i > 0. The error under each truth is a THEORETICAL quantity; here
    it is ESTIMATED by Monte-Carlo over `n_mc` datasets of `n` rows, evaluating the KNOWN LLR
    (no learned instrument). The returned dict reports the point estimate plus a standard error
    and 95% CI so the MC noise is explicit. Ties (Σ=0) contribute 0.5.

    Memory is bounded: replicates are processed in chunks of <= `max_rows_per_chunk` rows, so
    the estimate is valid for arbitrarily large n_mc*n without allocating it all at once.
    """
    reps_per_chunk = max(1, max_rows_per_chunk // n)
    e0_parts, e1_parts = [], []
    done = 0
    while done < n_mc:
        r = min(reps_per_chunk, n_mc - done)
        llr0 = _simulate_llr(params_h0, params_h0, params_h1, xmodel, r * n, rng.spawn(), nodes, weights)
        llr1 = _simulate_llr(params_h1, params_h0, params_h1, xmodel, r * n, rng.spawn(), nodes, weights)
        s0 = llr0.reshape(r, n).sum(dim=1)
        s1 = llr1.reshape(r, n).sum(dim=1)
        e0_parts.append((s0 > 0).double() + 0.5 * (s0 == 0).double())
        e1_parts.append((s1 < 0).double() + 0.5 * (s1 == 0).double())
        done += r
    e0 = torch.cat(e0_parts)
    e1 = torch.cat(e1_parts)
    n_mc = int(e0.numel())
    err_h0 = float(e0.mean().item())
    err_h1 = float(e1.mean().item())
    bayes_error = 0.5 * (err_h0 + err_h1)
    # SE of the mean of (err_h0+err_h1)/2 over independent H0/H1 MC draws
    var0 = float(e0.var(unbiased=True).item()) / n_mc
    var1 = float(e1.var(unbiased=True).item()) / n_mc
    se = 0.5 * math.sqrt(var0 + var1)
    return {
        "bayes_error": bayes_error,
        "bayes_error_se": se,
        "ci_low": bayes_error - 1.96 * se,
        "ci_high": bayes_error + 1.96 * se,
        "err_h0": err_h0,
        "err_h1": err_h1,
        "n_mc": n_mc,
        "n": n,
    }


@dataclass
class OracleCell:
    """One cell of the oracle surface: config + computed ceiling quantities."""

    delta: float
    beta1: float
    target_rate: float
    n: int
    rho: Optional[float]  # synthetic-X only; None for fitted real-X
    beta0_h0: float
    beta0_h1: float
    bayes_error: float  # Monte-Carlo estimate of the theoretical Bayes error
    bayes_error_se: float  # MC standard error of the estimate
    ci_low: float  # 95% CI lower
    ci_high: float  # 95% CI upper
    n_mc: int  # MC replicates used
    err_h0: float
    err_h1: float
    kl_10: float
    kl_01: float
    xmodel: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "delta": self.delta,
            "beta1": self.beta1,
            "target_rate": self.target_rate,
            "n": self.n,
            "rho": self.rho,
            "beta0_h0": self.beta0_h0,
            "beta0_h1": self.beta0_h1,
            "bayes_error": self.bayes_error,
            "bayes_error_se": self.bayes_error_se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_mc": self.n_mc,
            "err_h0": self.err_h0,
            "err_h1": self.err_h1,
            "kl_10": self.kl_10,
            "kl_01": self.kl_01,
            "xmodel": self.xmodel,
        }
