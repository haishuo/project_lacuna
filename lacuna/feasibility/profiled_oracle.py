"""
lacuna.feasibility.profiled_oracle

Composite / β₁′-profiled MAR-null oracle — the DEPLOYMENT-RELEVANT ceiling for the
β₁-flexible MAR family (minimum-viable composite null, PROPOSAL P1 next stage).

Point-null (oracle.py) compares two FIXED hypotheses (same β1). This module compares:
  H1 = MNAR(β₂=δ, β₁ original, β₀ rate-matched)
  H0 = best-fitting MAR over the family { MAR(β₂=0, β₁′ free, β₀′ rate-matched) }
β₁′ is chosen to MAXIMIZE the observed-data likelihood of the MAR family under H1 data —
equivalently the KL-projection of H1's observed-data law onto the MAR-β₁′ family, i.e. the MAR
that minimizes distinguishability. Then the n-sample Bayes error between H1 and that best MAR is
reported (with SE/CI), alongside the point-null for direct comparison.

Vocabulary: the point-null oracle is a ceiling for the fixed-null question ONLY; THIS profiled
oracle is the deployment-relevant ceiling for the β₁-flexible MAR family. Neither is a learned
estimator. NO model training here.

Minimum-viable caveat (documented, not hidden): a SINGLE β₁′ is selected per cell from one finite
fit-sample (not re-fit per Monte-Carlo replicate, which would be the full profile/GLRT). β₁′ is
unbiased (≈ the true MLE), but a finite-sample β₁′≠truth makes the best-fit MAR marginally distinct
from H1 even at δ=0, so the profiled Bayes error at δ=0 sits at ≈0.5 within ~0.03 rather than exactly
0.5. Read the profiled surface against its OWN δ=0 baseline, not against 0.5 exactly. A per-replicate
GLRT (later stage) would remove this floor.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lacuna.core.rng import RNGState
from .delta_generator import SelfCensorParams, _bisect_increasing
from .oracle import bayes_error_nsample, gauss_hermite, solve_beta0_population
from .xmodel import ConditionalGaussian, XModel


def _golden_max(f, a: float, b: float, iters: int = 28) -> float:
    """Golden-section maximization of a unimodal f on [a, b]."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
    return 0.5 * (a + b)


def mar_params(
    xmodel: XModel,
    beta1p: float,
    target_rate: float,
    rng: RNGState,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    n_sample: int = 20000,
) -> SelfCensorParams:
    """A rate-matched MAR mechanism (β₂=0) with predictor slope β₁′."""
    return solve_beta0_population(xmodel, beta1p, 0.0, target_rate, rng, nodes, weights, n_sample=n_sample)


def profile_beta1(
    h1_params: SelfCensorParams,
    xmodel: XModel,
    target_rate: float,
    rng: RNGState,
    beta1_grid: List[float],
    n_fit_sample: int = 30000,
    n_pop_sample: int = 20000,
) -> Tuple[float, SelfCensorParams, float]:
    """Select β₁′ maximizing the MAR observed-data log-likelihood under H1 data.

    Returns (β₁′*, best-fit MAR params, fit log-lik). Deterministic given rng (a single H1 fit
    sample and a single predictor sample are drawn and reused across all β₁′ candidates).
    """
    z_p_pop = xmodel.sample_predictor(n_pop_sample, rng.spawn()).double()

    def solve_b0(b1p: float) -> float:
        return _bisect_increasing(
            lambda b0: float(torch.sigmoid(b0 + b1p * z_p_pop).mean().item()) - target_rate,
            -50.0, 50.0,
        )

    # Fixed H1 observed-data sample (only z_p + observed matter for the β₂=0 MAR log-lik).
    z_p = xmodel.sample_predictor(n_fit_sample, rng.spawn()).double()
    z_t = xmodel.sample_conditional(z_p, rng.spawn()).double()
    eta_h1 = h1_params.beta0 + h1_params.beta1 * z_p + h1_params.beta2 * z_t
    observed = (rng.rand(n_fit_sample).double() >= torch.sigmoid(eta_h1)).bool()

    def loglik(b1p: float) -> float:
        eta = solve_b0(b1p) + b1p * z_p
        ll = torch.where(observed, -F.softplus(eta), -F.softplus(-eta)).mean()
        return float(ll.item())

    grid_vals = [(b1p, loglik(b1p)) for b1p in beta1_grid]
    best_i = int(np.argmax([v for _, v in grid_vals]))
    a = beta1_grid[max(0, best_i - 1)]
    b = beta1_grid[min(len(beta1_grid) - 1, best_i + 1)]
    b1_star = _golden_max(loglik, a, b) if b > a else beta1_grid[best_i]
    # keep the better of the refined point and the best grid point
    if loglik(b1_star) < grid_vals[best_i][1]:
        b1_star = beta1_grid[best_i]
    b0_star = solve_b0(b1_star)
    return b1_star, SelfCensorParams(beta0=b0_star, beta1=b1_star, beta2=0.0), loglik(b1_star)


def bayes_error_vs_mar(
    h1_params: SelfCensorParams,
    beta1p: float,
    xmodel: XModel,
    n: int,
    rng: RNGState,
    nodes: torch.Tensor,
    weights: torch.Tensor,
    target_rate: float,
    n_mc: int = 1000,
) -> Dict[str, float]:
    """Bayes error between H1 and the rate-matched MAR with slope β₁′ (β₂=0).

    At β₁′ == h1_params.beta1 this reproduces the point-null result (same H0).
    """
    h0 = mar_params(xmodel, beta1p, target_rate, rng.spawn(), nodes, weights)
    return bayes_error_nsample(h0, h1_params, xmodel, n, rng.spawn(), nodes, weights, n_mc=n_mc)


def _beta1_grid(beta1: float) -> List[float]:
    hi = max(5.0, 2.0 * beta1 + 1.0)
    return sorted(set([round(v, 4) for v in np.linspace(-1.0, hi, 25)] + [float(beta1)]))


def compute_profiled_cell(
    delta: float,
    beta1: float,
    target_rate: float,
    n: int,
    rng: RNGState,
    *,
    rho: Optional[float] = None,
    xmodel: Optional[XModel] = None,
    n_quad: int = 32,
    n_mc: int = 1000,
    n_pop_sample: int = 20000,
    n_fit_sample: int = 30000,
) -> Dict:
    """Compute one cell: point-null AND β₁′-profiled Bayes error, plus the selected β₁′."""
    if xmodel is None:
        if rho is None:
            raise ValueError("provide either an xmodel or rho")
        xmodel = ConditionalGaussian.synthetic(rho)
    nodes, weights = gauss_hermite(n_quad)

    h1 = solve_beta0_population(xmodel, beta1, delta, target_rate, rng.spawn(), nodes, weights, n_sample=n_pop_sample)
    h0_point = solve_beta0_population(xmodel, beta1, 0.0, target_rate, rng.spawn(), nodes, weights, n_sample=n_pop_sample)
    point = bayes_error_nsample(h0_point, h1, xmodel, n, rng.spawn(), nodes, weights, n_mc=n_mc)

    b1_star, h0_prof, fit_ll = profile_beta1(
        h1, xmodel, target_rate, rng.spawn(), _beta1_grid(beta1),
        n_fit_sample=n_fit_sample, n_pop_sample=n_pop_sample,
    )
    prof = bayes_error_nsample(h0_prof, h1, xmodel, n, rng.spawn(), nodes, weights, n_mc=n_mc)

    return {
        "delta": delta,
        "beta1": beta1,
        "beta1_prime": b1_star,
        "target_rate": target_rate,
        "rho": rho,
        "n": n,
        "point_bayes_error": point["bayes_error"],
        "point_se": point["bayes_error_se"],
        "profiled_bayes_error": prof["bayes_error"],
        "profiled_se": prof["bayes_error_se"],
        "profiled_ci_low": prof["ci_low"],
        "profiled_ci_high": prof["ci_high"],
        "beta0_h1": h1.beta0,
        "beta0_h0_point": h0_point.beta0,
        "beta0_h0_profiled": h0_prof.beta0,
        "n_mc": prof["n_mc"],
        "xmodel": xmodel.descriptor,
    }
