"""Tests for lacuna.feasibility.profiled_oracle_mv (P1R-A predictor-choice profiling)."""

import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.oracle import gauss_hermite
from lacuna.feasibility.profiled_oracle_mv import (
    bayes_error_mv,
    compute_p1ra_cell,
    mar_marginal_rate,
    solve_beta0_mar,
)
from lacuna.feasibility.xmodel_mv import A_IDX, P_IDX


def _h1(beta0=-1.5, delta=1.0):
    return SelfCensorParams(beta0=beta0, beta1=1.0, beta2=delta)


def test_mar_rate_match_per_candidate():
    nodes, weights = gauss_hermite(48)
    for beta1p in [0.5, 1.5, 2.5]:
        b0 = solve_beta0_mar(beta1p, 0.3, nodes, weights)
        assert abs(mar_marginal_rate(b0, beta1p, nodes, weights) - 0.3) < 1e-4


def test_reduction_when_only_zp_candidate_equals_restricted():
    """candidates={z_p} ⇒ richer == restricted exactly (selected predictor is z_p)."""
    cell = compute_p1ra_cell(0.9, 0.95, _h1(delta=1.0), n=256, target_rate=0.3, rng=RNGState(seed=1),
                             candidates=(P_IDX,), n_quad=24, n_mc=400, n_fit=8000, kl_sample=8000)
    assert cell["selected_predictor_idx"] == P_IDX
    assert abs(cell["E_richer_minus_restricted"]) < 1e-9   # identical computation path


def test_useless_added_proxy_no_absorption():
    """ρ_a=0 ⇒ z_a is pure noise ⇒ richer picks z_p, Δabsorb ≈ 0."""
    cell = compute_p1ra_cell(0.9, 0.0, _h1(delta=1.0), n=256, target_rate=0.3, rng=RNGState(seed=2),
                             n_quad=24, n_mc=800, n_fit=12000, kl_sample=12000)
    assert cell["selected_predictor_idx"] == P_IDX
    assert cell["E_richer_minus_restricted"] >= -0.03      # ~0 within MC tol
    assert cell["E_richer_minus_restricted"] < 0.05


def test_p1_style_regime_selects_zp_no_absorption():
    """In P1-style H1 (β1=1 ≥ δ), z_p is the best single predictor even at high ρ_a ⇒ Δ=0 exactly."""
    cell = compute_p1ra_cell(0.3, 0.9, _h1(delta=1.0), n=512, target_rate=0.3, rng=RNGState(seed=3),
                             n_quad=24, n_mc=1000, n_fit=15000, kl_sample=12000)
    assert cell["selected_predictor_idx"] == P_IDX
    assert cell["E_richer_minus_restricted"] == 0.0   # richer picked z_p ⇒ identical to restricted


def test_weak_beta1_owncensoring_selects_za_and_absorbs():
    """Own-value-dominated H1 (β1=0.2 < δ=2) with ρ_a>ρ_orig ⇒ richer picks z_a and absorbs."""
    h1 = SelfCensorParams(beta0=-1.5, beta1=0.2, beta2=2.0)
    cell = compute_p1ra_cell(0.3, 0.9, h1, n=512, target_rate=0.3, rng=RNGState(seed=30),
                             n_quad=24, n_mc=1000, n_fit=15000, kl_sample=12000)
    assert cell["selected_predictor_idx"] == A_IDX
    assert cell["E_richer_minus_restricted"] > 0.0     # z_a fits better ⇒ harder (higher Bayes error)


def test_monotonicity_no_violation_flag():
    cell = compute_p1ra_cell(0.9, 0.6, _h1(delta=1.0), n=512, target_rate=0.3, rng=RNGState(seed=4),
                             n_quad=24, n_mc=1000, n_fit=12000, kl_sample=12000)
    # richer must not be MORE distinguishable than restricted beyond MC tolerance
    assert cell["E_richer_minus_restricted"] >= -2 * cell["delta_se"]
    assert cell["monotonicity_violation"] is False


def test_delta_zero_is_chance():
    # H1 must be rate-matched: at δ=0 it IS a MAR-on-z_p, so the restricted null recovers it ⇒ ≈0.5.
    nodes, weights = gauss_hermite(32)
    b0 = solve_beta0_mar(1.0, 0.3, nodes, weights)
    cell = compute_p1ra_cell(0.9, 0.6, SelfCensorParams(beta0=b0, beta1=1.0, beta2=0.0), n=256,
                             target_rate=0.3, rng=RNGState(seed=5), n_quad=24, n_mc=800,
                             n_fit=10000, kl_sample=10000)
    # tolerance reflects the documented single-fit-β₁′ floor (~0.03-0.05 below 0.5) + MC noise
    assert abs(cell["E_restricted"] - 0.5) < 0.07
    assert abs(cell["E_richer"] - 0.5) < 0.07


def test_selected_predictor_field_always_recorded():
    cell = compute_p1ra_cell(0.6, 0.3, _h1(delta=1.0), n=256, target_rate=0.3, rng=RNGState(seed=6),
                             n_quad=24, n_mc=400, n_fit=8000, kl_sample=8000)
    assert cell["selected_predictor"] in ("z_p", "z_a")
    assert "beta1p_richer" in cell and "beta0p_richer" in cell
