"""Tests for lacuna.feasibility.profiled_oracle_mv_multi (P1R-C multi-predictor MAR)."""

import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.oracle import gauss_hermite
from lacuna.feasibility.profiled_oracle_mv_multi import (
    bayes_error_multi,
    compute_p1rc_cell,
    solve_beta0_multi,
)
from lacuna.feasibility.xmodel_mv import MultivariateGaussianX, build_p1ra_corr


def _h1(beta0=-1.5, delta=1.0):
    return SelfCensorParams(beta0=beta0, beta1=1.0, beta2=delta)


def test_rate_match_multi_predictor():
    nodes, weights = gauss_hermite(48)
    for b1, b2 in [(1.0, 0.0), (1.5, 0.5), (0.2, 2.0), (2.0, 2.0)]:
        b0 = solve_beta0_multi(b1, b2, rho_pa=0.3, target_rate=0.3, nodes=nodes, weights=weights)
        # verify realized marginal rate by direct MC
        xm = MultivariateGaussianX(build_p1ra_corr(0.5, 0.6))  # rho_pa here = 0.3
        X = xm.sample(200000, RNGState(seed=0))
        zp, za = X[:, 0], X[:, 1]
        rate = torch.sigmoid(b0 + b1 * zp + b2 * za).mean().item()
        assert abs(rate - 0.3) < 0.01


def test_beta2_zero_reduces_to_single_predictor():
    """β₂′=0 multi-predictor LLR equals the single-predictor (z_p) restricted Bayes error path."""
    from lacuna.feasibility.profiled_oracle_mv import bayes_error_mv
    xm = MultivariateGaussianX(build_p1ra_corr(0.9, 0.6))
    nodes, weights = gauss_hermite(32)
    h1 = _h1(beta0=-3.3, delta=1.0)
    b0, b1 = -3.0, 1.4
    a = bayes_error_multi(h1, b0, b1, 0.0, xm, 512, RNGState(seed=1), nodes, weights, n_mc=1500)["bayes_error"]
    b = bayes_error_mv(h1, 0, b0, b1, xm, 512, RNGState(seed=1), nodes, weights, n_mc=1500)["bayes_error"]
    assert abs(a - b) < 0.03  # same hypothesis (MAR on z_p), MC tolerance


def test_superset_monotonicity_no_violation():
    cell = compute_p1rc_cell(0.9, 0.9, _h1(delta=1.0), n=512, target_rate=0.3, rng=RNGState(seed=2),
                             n_grid=5, n_mc_search=300, n_mc_final=1500)
    assert cell["E_multi_minus_restricted"] >= -2 * cell["delta_se"]
    assert cell["monotonicity_violation"] is False


def test_known_absorption_high_rho_a_multi_beats_single():
    """At high ρ_a the multi-predictor MAR absorbs MORE than single-predictor choice (Δ>0, β₂′>0)."""
    cell = compute_p1rc_cell(0.9, 0.95, _h1(delta=1.0), n=512, target_rate=0.3, rng=RNGState(seed=3),
                             n_grid=6, n_mc_search=400, n_mc_final=2000)
    assert cell["beta2p_multi"] > 0.0
    assert cell["E_multi_minus_restricted"] > 0.02   # genuine absorption (P1R-A gave Δ=0 here)


def test_delta_zero_floor():
    nodes, weights = gauss_hermite(32)
    b0 = solve_beta0_multi(1.0, 0.0, 0.0, 0.3, nodes, weights)  # MAR-on-z_p at rate 0.3
    cell = compute_p1rc_cell(0.9, 0.6, SelfCensorParams(beta0=b0, beta1=1.0, beta2=0.0), n=256,
                             target_rate=0.3, rng=RNGState(seed=4), n_grid=5, n_mc_search=300, n_mc_final=1200)
    assert abs(cell["E_restricted"] - 0.5) < 0.07
    assert abs(cell["E_multi"] - 0.5) < 0.07


def test_records_coefficients_and_variance():
    cell = compute_p1rc_cell(0.3, 0.6, _h1(delta=1.0), n=256, target_rate=0.3, rng=RNGState(seed=5),
                             n_grid=5, n_mc_search=300, n_mc_final=1200)
    for k in ["beta1p_multi", "beta2p_multi", "beta0p_multi", "var_zt_given_obs", "edge_status",
              "E_restricted", "E_multi", "E_multi_minus_restricted", "rho_pa"]:
        assert k in cell
    # Var(z_t|z_p,z_a) ∈ (0,1) and below Var(z_t|z_p)=1-ρ_orig²
    assert 0.0 < cell["var_zt_given_obs"] < (1 - 0.3 ** 2) + 1e-9
