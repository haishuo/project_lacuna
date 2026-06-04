"""
lacuna.feasibility.profiled_oracle_mv

P1R-A: predictor-choice profiled MAR null, on a multi-column Gaussian X.

H1 (MNAR, fixed = the P1 cell): P(miss_t) = σ(β₀ + β₁·z_p + β₂·z_t).
H0 (MAR-on-j):                  P(miss_t) = σ(β₀′ + β₁′·z_j),  j ∈ candidate observed predictors.
  - restricted null: j fixed to the original predictor z_p (= the P1 β₁′-profiled null).
  - richer null:     j chosen from {z_p, z_a, ...} to best fit H1 (predictor choice).

β₁′ profiling (WITHIN each candidate j): β₁′*(j) = argmax_{β₁′} E_{H1}[log L_{H0(j,β₁′)}(mask)] (the
rate-matched MAR that best fits the missingness indicator given z_j), β₀′ rate-matched.

Candidate SELECTION (ACROSS predictors): the richer (composite) null's ceiling is the LEAST-FAVORABLE
member = j* = argmax_j Bayes_error(H1, H0(j)) — the candidate HARDEST to distinguish — NOT argmax of
the mask-likelihood. These disagree under a shadow column (z_a≈z_t at high ρ_a): the mask-likelihood
prefers z_a (missingness correlates with z_a) but the *observed-data* discriminator exploits z_a as a
shadow to detect the truncation, making MAR-on-z_a MORE distinguishable. Selecting by Bayes error is
the correct least-favorable-null criterion and guarantees E_richer ≥ E_restricted (z_p is always a
candidate). A residual violation beyond MC tolerance is a bug / MC noise, never "easier."

The oracle LLR generalizes the P1 form: the missing-target term integrates z_t over p(z_t | ALL
observed predictors) (the discriminator's shadow information); the observed/MAR terms use the chosen
single predictor. Bayes error is the MC estimate of the optimal LLR test's theoretical error (float64,
with SE/CI) — the only object called a ceiling. NO model training.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from lacuna.core.rng import RNGState
from .delta_generator import SelfCensorParams, _bisect_increasing
from .oracle import gauss_hermite
from .profiled_oracle import _golden_max
from .xmodel_mv import A_IDX, P_IDX, T_IDX, MultivariateGaussianX, build_p1ra_corr

_LOG_EPS = 1e-12
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_OBS = (P_IDX, A_IDX)  # observed predictors the discriminator conditions the missing target on


def mar_marginal_rate(beta0p: float, beta1p: float, nodes: torch.Tensor, weights: torch.Tensor) -> float:
    """E_{z~N(0,1)}[ σ(β0′ + β1′ z) ] via Gauss–Hermite (z_j has standard-normal margin)."""
    z = math.sqrt(2.0) * nodes
    return float((torch.sigmoid(beta0p + beta1p * z) * weights).sum().item() * _INV_SQRT_PI)


def solve_beta0_mar(beta1p: float, target_rate: float, nodes: torch.Tensor, weights: torch.Tensor) -> float:
    """β0′ such that the MAR predictor's marginal missing rate equals target_rate."""
    if not (0.0 < target_rate < 1.0):
        raise ValueError(f"target_rate must be in (0,1); got {target_rate}")
    return _bisect_increasing(lambda b0: mar_marginal_rate(b0, beta1p, nodes, weights) - target_rate, -50.0, 50.0)


def h1_missing_prob(zp, za, h1: SelfCensorParams, xmodel: MultivariateGaussianX, nodes, weights):
    """m1(z_p,z_a) = E_{z_t | z_p,z_a}[ σ(β0 + β1 z_p + β2 z_t) ] via GH over the conditional."""
    z_obs = torch.stack([zp, za], dim=1)
    mean, var = xmodel.conditional_mean_var(T_IDX, _OBS, z_obs)
    zt_grid = mean.unsqueeze(1) + math.sqrt(2.0 * var) * nodes.unsqueeze(0)
    eta = h1.beta0 + h1.beta1 * zp.unsqueeze(1) + h1.beta2 * zt_grid
    return _INV_SQRT_PI * (torch.sigmoid(eta) * weights.unsqueeze(0)).sum(dim=1)


def _predictor_col(zp, za, j: int):
    return zp if j == P_IDX else za


def llr_mv(zp, za, zt, observed, h1, j, beta0p, beta1p, xmodel, nodes, weights):
    """Per-row observed-data LLR (H1 over H0(j)). Float64."""
    eta1 = h1.beta0 + h1.beta1 * zp + h1.beta2 * zt
    eta0 = beta0p + beta1p * _predictor_col(zp, za, j)
    obs_llr = F.softplus(eta0) - F.softplus(eta1)  # log(1-σ1) - log(1-σ0)
    m1 = h1_missing_prob(zp, za, h1, xmodel, nodes, weights).clamp_min(_LOG_EPS)
    m0 = torch.sigmoid(eta0).clamp_min(_LOG_EPS)   # H0 missing prob (no z_t dependence)
    mis_llr = torch.log(m1) - torch.log(m0)
    return torch.where(observed, obs_llr, mis_llr)


def _simulate_llr_mv(true_is_h1, h1, j, b0p, b1p, xmodel, n_rows, rng, nodes, weights):
    X = xmodel.sample(n_rows, rng)
    zp, za, zt = X[:, P_IDX], X[:, A_IDX], X[:, T_IDX]
    if true_is_h1:
        eta_true = h1.beta0 + h1.beta1 * zp + h1.beta2 * zt
    else:
        eta_true = b0p + b1p * _predictor_col(zp, za, j)
    observed = rng.rand(n_rows, dtype=torch.float64) >= torch.sigmoid(eta_true)
    return llr_mv(zp, za, zt, observed, h1, j, b0p, b1p, xmodel, nodes, weights)


def bayes_error_mv(h1, j, b0p, b1p, xmodel, n, rng, nodes, weights, n_mc=2000, max_rows_per_chunk=250_000):
    """MC estimate of the n-sample Bayes error of the optimal LLR test between H1 and H0(j)."""
    reps_per_chunk = max(1, max_rows_per_chunk // n)
    e0, e1 = [], []
    done = 0
    while done < n_mc:
        r = min(reps_per_chunk, n_mc - done)
        s0 = _simulate_llr_mv(False, h1, j, b0p, b1p, xmodel, r * n, rng.spawn(), nodes, weights).reshape(r, n).sum(1)
        s1 = _simulate_llr_mv(True, h1, j, b0p, b1p, xmodel, r * n, rng.spawn(), nodes, weights).reshape(r, n).sum(1)
        e0.append((s0 > 0).double() + 0.5 * (s0 == 0).double())
        e1.append((s1 < 0).double() + 0.5 * (s1 == 0).double())
        done += r
    e0, e1 = torch.cat(e0), torch.cat(e1)
    err0, err1 = float(e0.mean()), float(e1.mean())
    be = 0.5 * (err0 + err1)
    se = 0.5 * math.sqrt(float(e0.var(unbiased=True)) / n_mc + float(e1.var(unbiased=True)) / n_mc)
    return {"bayes_error": be, "bayes_error_se": se, "ci_low": be - 1.96 * se, "ci_high": be + 1.96 * se,
            "err_h0": err0, "err_h1": err1, "n_mc": n_mc}


def _kl_h1(h1, j, b0p, b1p, xmodel, rng, nodes, weights, n_sample=40000):
    """KL(H1 ‖ H0(j)) = E_{H1}[LLR]."""
    return float(_simulate_llr_mv(True, h1, j, b0p, b1p, xmodel, n_sample, rng.spawn(), nodes, weights).mean())


def profile_candidate(h1, j, xmodel, target_rate, rng, nodes, weights, beta1_grid, n_fit=30000):
    """β₁′ profiling for ONE candidate predictor j (argmax MAR observed log-lik under H1 data)."""
    X = xmodel.sample(n_fit, rng.spawn())
    zp, za, zt = X[:, P_IDX], X[:, A_IDX], X[:, T_IDX]
    observed = rng.rand(n_fit, dtype=torch.float64) >= torch.sigmoid(h1.beta0 + h1.beta1 * zp + h1.beta2 * zt)
    z_j = _predictor_col(zp, za, j)

    def loglik(b1p):
        eta0 = solve_beta0_mar(b1p, target_rate, nodes, weights) + b1p * z_j
        return float(torch.where(observed, -F.softplus(eta0), -F.softplus(-eta0)).mean())

    vals = [(b, loglik(b)) for b in beta1_grid]
    bi = max(range(len(vals)), key=lambda i: vals[i][1])
    a, b = beta1_grid[max(0, bi - 1)], beta1_grid[min(len(beta1_grid) - 1, bi + 1)]
    b1_star = _golden_max(loglik, a, b) if b > a else beta1_grid[bi]
    if loglik(b1_star) < vals[bi][1]:
        b1_star = beta1_grid[bi]
    b0_star = solve_beta0_mar(b1_star, target_rate, nodes, weights)
    return {"j": j, "beta1p": b1_star, "beta0p": b0_star, "fit_loglik": loglik(b1_star)}


def _beta1_grid():
    import numpy as np
    return sorted(set(round(float(v), 4) for v in np.linspace(-1.0, 5.0, 25)))


def compute_p1ra_cell(
    rho_orig: float, rho_a: float, h1: SelfCensorParams, n: int, target_rate: float, rng: RNGState,
    *, candidates: Tuple[int, ...] = (P_IDX, A_IDX), n_quad: int = 32, n_mc: int = 2000,
    n_fit: int = 30000, kl_sample: int = 40000,
) -> Dict:
    """Restricted (z_p only) vs richer (predictor-choice) profiled MAR null on the SAME 3-col data."""
    xmodel = MultivariateGaussianX(build_p1ra_corr(rho_orig, rho_a), names=["z_p", "z_a", "z_t"])
    nodes, weights = gauss_hermite(n_quad)
    grid = _beta1_grid()

    # Within each candidate predictor, profile β₁′ by the mask-likelihood proxy (fine — the shadow
    # effect is the same across β₁′ for a fixed predictor). ACROSS candidates, the richer (composite)
    # null's ceiling is the LEAST-FAVORABLE member = the candidate with the MAXIMUM Bayes error (the
    # hardest to distinguish), NOT the max mask-likelihood: those disagree under a shadow column, and
    # max-mask-likelihood can pick a MORE distinguishable predictor (a monotonicity violation). Selecting
    # by Bayes error guarantees E_richer ≥ E_restricted by construction (z_p is always a candidate).
    profiles = {j: profile_candidate(h1, j, xmodel, target_rate, rng.spawn(), nodes, weights, grid, n_fit)
                for j in candidates}
    be = {j: bayes_error_mv(h1, j, p["beta0p"], p["beta1p"], xmodel, n, rng.spawn(), nodes, weights, n_mc)
          for j, p in profiles.items()}

    restricted = profiles[P_IDX]
    be_r = be[P_IDX]
    kl_r = _kl_h1(h1, P_IDX, restricted["beta0p"], restricted["beta1p"], xmodel, rng.spawn(), nodes, weights, kl_sample)
    richer_j = max(be, key=lambda j: be[j]["bayes_error"])  # least-favorable null member
    richer = profiles[richer_j]
    be_R = be[richer_j]
    kl_R = kl_r if richer_j == P_IDX else _kl_h1(h1, richer_j, richer["beta0p"], richer["beta1p"], xmodel, rng.spawn(), nodes, weights, kl_sample)
    mask_fit_j = max(profiles, key=lambda j: profiles[j]["fit_loglik"])  # diagnostic only

    delta = be_R["bayes_error"] - be_r["bayes_error"]
    delta_se = math.sqrt(be_R["bayes_error_se"] ** 2 + be_r["bayes_error_se"] ** 2)
    return {
        "rho_orig": rho_orig, "rho_a": rho_a, "delta": h1.beta2, "n": n, "target_rate": target_rate,
        "selected_predictor": "z_a" if richer["j"] == A_IDX else "z_p",
        "selected_predictor_idx": richer["j"],
        "mask_fit_selected": "z_a" if mask_fit_j == A_IDX else "z_p",  # diagnostic: where the mask-likelihood proxy disagrees
        "beta1p_restricted": restricted["beta1p"], "beta0p_restricted": restricted["beta0p"],
        "beta1p_richer": richer["beta1p"], "beta0p_richer": richer["beta0p"],
        "E_restricted": be_r["bayes_error"], "E_restricted_se": be_r["bayes_error_se"],
        "E_richer": be_R["bayes_error"], "E_richer_se": be_R["bayes_error_se"],
        "E_richer_minus_restricted": delta, "delta_se": delta_se,
        "delta_ci": [delta - 1.96 * delta_se, delta + 1.96 * delta_se],
        "monotonicity_violation": bool(delta < -2 * delta_se),
        "kl_restricted": kl_r, "kl_richer": kl_R,
        "xmodel": xmodel.descriptor,
    }
