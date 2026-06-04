"""
lacuna.feasibility.profiled_oracle_mv_multi

P1R-C: multi-predictor MAR null — the first richer null that can GENUINELY absorb the P1 signal.

H1 (MNAR, fixed = P1 cell): σ(β₀ + β₁·z_p + β₂·z_t).
H0 (multi-predictor MAR):   σ(β₀′ + β₁′·z_p + β₂′·z_a), β₀′ rate-matched, (β₁′,β₂′) profiled.

Analytical structure: z_t = a·z_p + b·z_a + ε (ε ⟂ z_p,z_a). The multi-predictor MAR matches the z_p
and z_a parts exactly, leaving the unabsorbable residual β₂·ε. So the surviving MNAR signal is governed
by Var(z_t | z_p, z_a); as ρ_a→1 it vanishes and MNAR collapses to MAR.

Least-favorable null (the P1R-A lesson): the richer null's ceiling is the (β₁′,β₂′) MAXIMIZING Bayes
error (minimizing distinguishability), NOT max mask-likelihood. We search a 2-D grid (centered on the
analytical KL-projection) that INCLUDES the β₂′=0 slice (the restricted single-predictor null), and take
E_multi = max Bayes error over the full grid, E_restricted = max over the β₂′=0 slice ⇒ E_multi ≥
E_restricted by construction. If the argmax hits an upper grid edge the grid is expanded before
interpretation. Finite-grid E_multi is a conservative lower bound on the true least-favorable.

CPU oracle; float64; NO model training.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lacuna.core.rng import RNGState
from .delta_generator import SelfCensorParams, _bisect_increasing
from .oracle import gauss_hermite
from .profiled_oracle_mv import h1_missing_prob
from .xmodel_mv import A_IDX, P_IDX, T_IDX, MultivariateGaussianX, build_p1ra_corr

_LOG_EPS = 1e-12
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def solve_beta0_multi(b1p, b2p, rho_pa, target_rate, nodes, weights):
    """β₀′ matching the marginal rate for the linear index β₁′z_p+β₂′z_a ~ N(0, s²)."""
    s2 = b1p * b1p + b2p * b2p + 2.0 * b1p * b2p * rho_pa
    s = math.sqrt(max(s2, 0.0))
    xi = s * math.sqrt(2.0) * nodes

    def rate(b0):
        return float((torch.sigmoid(b0 + xi) * weights).sum().item()) * _INV_SQRT_PI

    return _bisect_increasing(lambda b0: rate(b0) - target_rate, -50.0, 50.0)


def llr_multi(zp, za, zt, observed, h1, b0p, b1p, b2p, xmodel, nodes, weights):
    eta1 = h1.beta0 + h1.beta1 * zp + h1.beta2 * zt
    eta0 = b0p + b1p * zp + b2p * za
    obs_llr = F.softplus(eta0) - F.softplus(eta1)
    m1 = h1_missing_prob(zp, za, h1, xmodel, nodes, weights).clamp_min(_LOG_EPS)
    m0 = torch.sigmoid(eta0).clamp_min(_LOG_EPS)
    return torch.where(observed, obs_llr, torch.log(m1) - torch.log(m0))


def _simulate(true_is_h1, h1, b0p, b1p, b2p, xmodel, n_rows, rng, nodes, weights):
    X = xmodel.sample(n_rows, rng)
    zp, za, zt = X[:, P_IDX], X[:, A_IDX], X[:, T_IDX]
    eta = (h1.beta0 + h1.beta1 * zp + h1.beta2 * zt) if true_is_h1 else (b0p + b1p * zp + b2p * za)
    observed = rng.rand(n_rows, dtype=torch.float64) >= torch.sigmoid(eta)
    return llr_multi(zp, za, zt, observed, h1, b0p, b1p, b2p, xmodel, nodes, weights)


def bayes_error_multi(h1, b0p, b1p, b2p, xmodel, n, rng, nodes, weights, n_mc=400, max_rows_per_chunk=250_000):
    reps = max(1, max_rows_per_chunk // n)
    e0, e1 = [], []
    done = 0
    while done < n_mc:
        r = min(reps, n_mc - done)
        s0 = _simulate(False, h1, b0p, b1p, b2p, xmodel, r * n, rng.spawn(), nodes, weights).reshape(r, n).sum(1)
        s1 = _simulate(True, h1, b0p, b1p, b2p, xmodel, r * n, rng.spawn(), nodes, weights).reshape(r, n).sum(1)
        e0.append((s0 > 0).double() + 0.5 * (s0 == 0).double())
        e1.append((s1 < 0).double() + 0.5 * (s1 == 0).double())
        done += r
    e0, e1 = torch.cat(e0), torch.cat(e1)
    err0, err1 = float(e0.mean()), float(e1.mean())
    be = 0.5 * (err0 + err1)
    se = 0.5 * math.sqrt(float(e0.var(unbiased=True)) / n_mc + float(e1.var(unbiased=True)) / n_mc)
    return {"bayes_error": be, "bayes_error_se": se, "ci_low": be - 1.96 * se, "ci_high": be + 1.96 * se}


def _grids(center_b1, center_b2, n_grid, b1_span, b2_hi):
    """(β₁′,β₂′) grids centered on the analytical projection; always include the center and β₂′=0."""
    b1 = sorted(set(round(float(x), 4) for x in np.linspace(max(0.0, center_b1 - b1_span), center_b1 + b1_span, n_grid))
                | {round(float(center_b1), 4)})
    b2 = sorted(set(round(float(x), 4) for x in np.linspace(0.0, b2_hi, n_grid))
                | {0.0, round(float(max(0.0, center_b2)), 4)})
    return b1, b2


def _search_grid(h1, xmodel, rho_pa, rate, n, rng, nodes, weights, b1_grid, b2_grid, n_mc):
    """Evaluate Bayes error on the given (β₁′,β₂′) grid. Returns list of (b1,b2,b0,be)."""
    out = []
    for b1 in b1_grid:
        for b2 in b2_grid:
            b0 = solve_beta0_multi(b1, b2, rho_pa, rate, nodes, weights)
            be = bayes_error_multi(h1, b0, b1, b2, xmodel, n, rng.spawn(), nodes, weights, n_mc)["bayes_error"]
            out.append((float(b1), float(b2), float(b0), float(be)))
    return out


def compute_p1rc_cell(
    rho_orig, rho_a, h1: SelfCensorParams, n, target_rate, rng,
    *, n_quad=32, n_grid=6, n_mc_search=400, n_mc_final=2500, max_expansions=2,
) -> Dict:
    """Restricted single-predictor (β₂′=0) vs multi-predictor MAR null on the same 3-col data."""
    xmodel = MultivariateGaussianX(build_p1ra_corr(rho_orig, rho_a), names=["z_p", "z_a", "z_t"])
    nodes, weights = gauss_hermite(n_quad)
    rho_pa = rho_orig * rho_a
    breg, cond_var = xmodel._conditioner(T_IDX, (P_IDX, A_IDX))  # [a, b], Var(z_t|z_p,z_a)
    a, b = float(breg[0]), float(breg[1])
    center_b1, center_b2 = h1.beta1 + h1.beta2 * a, h1.beta2 * b

    b1_span = max(1.5, abs(center_b1))
    b2_hi = max(2.0 * center_b2, 0.5) + 0.5
    edge_status = "ok"
    for _ in range(max_expansions + 1):
        b1_grid, b2_grid = _grids(center_b1, center_b2, n_grid, b1_span, b2_hi)
        grid = _search_grid(h1, xmodel, rho_pa, target_rate, n, rng.spawn(), nodes, weights, b1_grid, b2_grid, n_mc_search)
        multi = max(grid, key=lambda r: r[3])
        on_b1_edge = abs(multi[0] - b1_grid[-1]) < 1e-9 or (multi[0] > 0 and abs(multi[0] - b1_grid[0]) < 1e-9)
        on_b2_edge = abs(multi[1] - b2_grid[-1]) < 1e-9 and multi[1] > 0
        if not (on_b1_edge or on_b2_edge):
            break
        edge_status = "expanded"
        if on_b1_edge:
            b1_span *= 1.8
        if on_b2_edge:
            b2_hi *= 1.8
    else:
        edge_status = "edge-after-max-expansion"

    restricted = max([r for r in grid if r[1] == 0.0], key=lambda r: r[3])  # β₂′=0 slice
    be_restr = bayes_error_multi(h1, restricted[2], restricted[0], restricted[1], xmodel, n, rng.spawn(), nodes, weights, n_mc_final)
    if multi[1] == 0.0:  # multi argmax is in the restricted slice ⇒ identical ⇒ reuse (Δ=0 exact)
        be_multi = be_restr
        multi = restricted
    else:
        be_multi = bayes_error_multi(h1, multi[2], multi[0], multi[1], xmodel, n, rng.spawn(), nodes, weights, n_mc_final)

    delta = be_multi["bayes_error"] - be_restr["bayes_error"]
    delta_se = math.sqrt(be_multi["bayes_error_se"] ** 2 + be_restr["bayes_error_se"] ** 2)
    return {
        "rho_orig": rho_orig, "rho_a": rho_a, "rho_pa": rho_pa, "delta": h1.beta2, "n": n,
        "target_rate": target_rate, "var_zt_given_obs": cond_var,
        "E_restricted": be_restr["bayes_error"], "E_restricted_se": be_restr["bayes_error_se"],
        "E_multi": be_multi["bayes_error"], "E_multi_se": be_multi["bayes_error_se"],
        "E_multi_ci": [be_multi["ci_low"], be_multi["ci_high"]],
        "E_multi_minus_restricted": delta, "delta_se": delta_se,
        "delta_ci": [delta - 1.96 * delta_se, delta + 1.96 * delta_se],
        "monotonicity_violation": bool(delta < -2 * delta_se),
        "beta1p_restricted": restricted[0], "beta0p_restricted": restricted[2],
        "beta1p_multi": multi[0], "beta2p_multi": multi[1], "beta0p_multi": multi[2],
        "projection_center": [center_b1, center_b2], "edge_status": edge_status,
        "xmodel": xmodel.descriptor,
    }
