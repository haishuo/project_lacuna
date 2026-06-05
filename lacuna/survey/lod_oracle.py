"""
lacuna.survey.lod_oracle

Bayes-optimal oracle for the LOD / top-coding idiom (PROPOSAL-P2.2c §3) — the ORACLE-FIRST gate.

Mirrors `lacuna.feasibility.oracle` but for the STEP mechanism
`P(missing)=sigmoid(β₀ + β₁ z_p + δ·1[z_t>τ])`. By Neyman–Pearson the n-sample Bayes error of the
optimal LLR test built from the KNOWN generative model is the ceiling no estimator on observed data
can beat — so this tells us, BEFORE training any model, whether the matched-rate LOD footprint
carries δ information (Bayes error << 0.5) or is flat (≈ 0.5). Run this before blaming the model
(NORTH-STAR §5). H0 is MAR (δ=0, depends only on z_p); we report both the point-null (same β₁) and
the β₁′-PROFILED MAR null (the strongest re-fitting MAR competitor — a signal that survives it is
real, not a rate/slope artifact).

Likelihood arithmetic in float64 (the censoring logits and KL terms are sensitive). Deterministic
given the injected RNGState. Reuses `gauss_hermite`, `_bisect_increasing`, and the X-model interface.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import _bisect_increasing
from lacuna.feasibility.oracle import gauss_hermite
from lacuna.feasibility.xmodel import XModel

from .lod_generator import LODParams

_LOG_EPS = 1e-12
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def _step_eta(z_p, z_t, p: LODParams):
    return p.beta0 + p.beta1 * z_p + p.delta * (z_t > p.tau).to(z_t.dtype)


def lod_missing_prob(z_p, params: LODParams, xmodel: XModel, nodes, weights) -> torch.Tensor:
    """m(z_p) = E_{z_t|z_p}[ σ(β₀ + β₁ z_p + δ·1[z_t>τ]) ] by Gauss–Hermite. δ=0 ⇒ σ(β₀+β₁ z_p)."""
    z_p = z_p.to(torch.float64)
    nodes = nodes.to(torch.float64)
    weights = weights.to(torch.float64)
    mean, var = xmodel.conditional_mean_var(z_p)
    scale = math.sqrt(2.0 * var)
    z_t_grid = mean.unsqueeze(1) + scale * nodes.unsqueeze(0)  # [n, K]
    step = (z_t_grid > params.tau).to(torch.float64)
    eta = params.beta0 + params.beta1 * z_p.unsqueeze(1) + params.delta * step
    return _INV_SQRT_PI * (torch.sigmoid(eta) * weights.unsqueeze(0)).sum(dim=1)


def lod_llr_rows(z_p, z_t, observed, h0: LODParams, h1: LODParams, xmodel, nodes, weights):
    """Per-row observed-data LLR (H1 over H0). X-model terms cancel; only the mechanism differs."""
    z_p = z_p.to(torch.float64)
    z_t = z_t.to(torch.float64)
    obs_llr = F.softplus(_step_eta(z_p, z_t, h0)) - F.softplus(_step_eta(z_p, z_t, h1))
    m0 = lod_missing_prob(z_p, h0, xmodel, nodes, weights).clamp_min(_LOG_EPS)
    m1 = lod_missing_prob(z_p, h1, xmodel, nodes, weights).clamp_min(_LOG_EPS)
    return torch.where(observed, obs_llr, torch.log(m1) - torch.log(m0))


def lod_solve_beta0(xmodel, beta1, delta, tau, target_rate, rng, nodes, weights, n_sample=20000):
    """Solve β₀ so the POPULATION marginal rate under the X-model equals target_rate."""
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    z_p = xmodel.sample_predictor(n_sample, rng).to(torch.float64)

    def f(b0):
        p = LODParams(b0, beta1, delta, tau)
        return float(lod_missing_prob(z_p, p, xmodel, nodes, weights).mean().item()) - target_rate

    return LODParams(_bisect_increasing(f, -50.0, 50.0), beta1, delta, tau)


def tau_from_xmodel(xmodel, tau_quantile, rng, n_sample=80000) -> float:
    """Population z_t quantile τ under the X-model (matches the generator's empirical-quantile τ)."""
    if not (0.0 < tau_quantile < 1.0):
        raise ValueError(f"tau_quantile must be in (0, 1), got {tau_quantile}")
    z_t = xmodel.sample_conditional(xmodel.sample_predictor(n_sample, rng.spawn()), rng.spawn())
    return float(torch.quantile(z_t.to(torch.float64), tau_quantile).item())


def _simulate_llr(true_p, h0, h1, xmodel, n_rows, rng, nodes, weights):
    z_p = xmodel.sample_predictor(n_rows, rng)
    z_t = xmodel.sample_conditional(z_p, rng)
    p_missing = torch.sigmoid(_step_eta(z_p.to(torch.float64), z_t.to(torch.float64), true_p))
    observed = rng.rand(n_rows).to(torch.float64) >= p_missing
    return lod_llr_rows(z_p, z_t, observed, h0, h1, xmodel, nodes, weights)


def lod_bayes_error(h0, h1, xmodel, n, rng, nodes, weights, n_mc=2000, max_rows_per_chunk=250_000):
    """MC estimate of the n-sample Bayes error of the optimal LLR test (equal priors). Ties ⇒ 0.5."""
    reps_per_chunk = max(1, max_rows_per_chunk // n)
    e0_parts, e1_parts, done = [], [], 0
    while done < n_mc:
        r = min(reps_per_chunk, n_mc - done)
        llr0 = _simulate_llr(h0, h0, h1, xmodel, r * n, rng.spawn(), nodes, weights)
        llr1 = _simulate_llr(h1, h0, h1, xmodel, r * n, rng.spawn(), nodes, weights)
        s0 = llr0.reshape(r, n).sum(dim=1)
        s1 = llr1.reshape(r, n).sum(dim=1)
        e0_parts.append((s0 > 0).double() + 0.5 * (s0 == 0).double())
        e1_parts.append((s1 < 0).double() + 0.5 * (s1 == 0).double())
        done += r
    e0 = torch.cat(e0_parts)
    e1 = torch.cat(e1_parts)
    m = int(e0.numel())
    err_h0, err_h1 = float(e0.mean().item()), float(e1.mean().item())
    be = 0.5 * (err_h0 + err_h1)
    se = 0.5 * math.sqrt(float(e0.var(unbiased=True).item()) / m + float(e1.var(unbiased=True).item()) / m)
    return {"bayes_error": be, "bayes_error_se": se, "err_h0": err_h0, "err_h1": err_h1, "n_mc": m}


def profile_mar_null(h1: LODParams, xmodel, target_rate, rng, beta1_grid, nodes, weights, n_fit=30000):
    """Best-fit MAR null (δ=0): the β₁′ maximizing the MAR observed-data log-lik under H1 (LOD) data.

    The strongest re-fitting MAR competitor. Returns the MAR LODParams (δ=0). MAR cannot depend on
    z_t, so it can never reproduce the LOD target-value truncation — but we profile it to be honest.
    """
    z_p_pop = xmodel.sample_predictor(n_fit, rng.spawn()).to(torch.float64)

    def solve_b0(b1p):
        return _bisect_increasing(
            lambda b0: float(torch.sigmoid(b0 + b1p * z_p_pop).mean().item()) - target_rate, -50.0, 50.0
        )

    z_p = xmodel.sample_predictor(n_fit, rng.spawn())
    z_t = xmodel.sample_conditional(z_p, rng.spawn())
    observed = (rng.rand(n_fit).to(torch.float64) >= torch.sigmoid(
        _step_eta(z_p.to(torch.float64), z_t.to(torch.float64), h1))).bool()
    z_pd = z_p.to(torch.float64)

    def loglik(b1p):
        eta = solve_b0(b1p) + b1p * z_pd
        return float(torch.where(observed, -F.softplus(eta), -F.softplus(-eta)).mean().item())

    vals = [(b1p, loglik(b1p)) for b1p in beta1_grid]
    best = max(vals, key=lambda t: t[1])[0]
    return LODParams(solve_b0(best), best, 0.0, 0.0)


def lod_oracle_cell(
    xmodel, *, delta, beta1, tau_quantile, target_rate, n, rng,
    n_mc=2000, n_nodes=128, beta1_grid: List[float] = None,
) -> Dict:
    """Point-null AND profiled-MAR-null Bayes error for one LOD config. The pre-train gate cell."""
    nodes, weights = gauss_hermite(n_nodes)
    tau = tau_from_xmodel(xmodel, tau_quantile, rng.spawn())
    h1 = lod_solve_beta0(xmodel, beta1, delta, tau, target_rate, rng.spawn(), nodes, weights)
    h0_point = lod_solve_beta0(xmodel, beta1, 0.0, tau, target_rate, rng.spawn(), nodes, weights)
    point = lod_bayes_error(h0_point, h1, xmodel, n, rng.spawn(), nodes, weights, n_mc=n_mc)
    grid = beta1_grid if beta1_grid is not None else [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    h0_prof = profile_mar_null(h1, xmodel, target_rate, rng.spawn(), grid, nodes, weights)
    prof = lod_bayes_error(h0_prof, h1, xmodel, n, rng.spawn(), nodes, weights, n_mc=n_mc)
    return {
        "delta": delta, "beta1": beta1, "tau": tau, "tau_quantile": tau_quantile,
        "target_rate": target_rate, "n": n, "beta1p_star": h0_prof.beta1,
        "point_bayes_error": point["bayes_error"], "point_se": point["bayes_error_se"],
        "profiled_bayes_error": prof["bayes_error"], "profiled_se": prof["bayes_error_se"],
    }
