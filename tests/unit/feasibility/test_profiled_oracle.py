"""Tests for lacuna.feasibility.profiled_oracle (β₁′-profiled MAR-null ceiling)."""

import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.oracle import bayes_error_nsample, gauss_hermite, solve_beta0_population
from lacuna.feasibility.profiled_oracle import (
    bayes_error_vs_mar,
    compute_profiled_cell,
    mar_params,
    profile_beta1,
)
from lacuna.feasibility.xmodel import ConditionalGaussian


def test_profiled_reproduces_point_null_when_beta1prime_equals_beta1():
    """bayes_error_vs_mar at β₁′ == original β₁ matches the point-null (same H0), up to MC tol."""
    xm = ConditionalGaussian.synthetic(0.6)
    nodes, weights = gauss_hermite(32)
    rate, beta1, delta, n = 0.3, 1.0, 1.0, 256

    h1 = solve_beta0_population(xm, beta1, delta, rate, RNGState(seed=10).spawn(), nodes, weights, n_sample=8000)
    h0_point = solve_beta0_population(xm, beta1, 0.0, rate, RNGState(seed=11).spawn(), nodes, weights, n_sample=8000)
    point = bayes_error_nsample(h0_point, h1, xm, n, RNGState(seed=1), nodes, weights, n_mc=1500)["bayes_error"]

    via_mar = bayes_error_vs_mar(h1, beta1, xm, n, RNGState(seed=1), nodes, weights, rate, n_mc=1500)["bayes_error"]
    assert abs(point - via_mar) < 0.03  # same H0 family member; only MC/solve noise differs


def test_profiling_never_makes_null_easier_to_distinguish():
    """Best-fitting MAR is >= as hard to distinguish as the fixed MAR: profiled >= point - tol."""
    cell = compute_profiled_cell(
        delta=0.5, beta1=1.0, target_rate=0.3, n=512, rng=RNGState(seed=3),
        rho=0.9, n_quad=32, n_mc=1200, n_pop_sample=8000, n_fit_sample=15000,
    )
    assert cell["profiled_bayes_error"] >= cell["point_bayes_error"] - 0.03


def test_delta_zero_profiled_is_chance_and_picks_original_beta1():
    """δ=0 ⟹ H1 is itself a MAR; best-fitting MAR ≈ itself ⟹ Bayes error ≈0.5, β₁′≈β₁.

    Tolerance reflects the documented minimum-viable floor: a single finite-fit-sample β₁′
    (unbiased) leaves <~0.03 apparent distinguishability at δ=0.
    """
    cell = compute_profiled_cell(
        delta=0.0, beta1=1.0, target_rate=0.3, n=256, rng=RNGState(seed=7),
        rho=0.6, n_quad=32, n_mc=1000, n_pop_sample=8000, n_fit_sample=30000,
    )
    assert abs(cell["profiled_bayes_error"] - 0.5) < 0.04
    assert abs(cell["beta1_prime"] - 1.0) < 0.1  # β₁′ unbiased toward the truth


def test_selected_beta1_prime_is_recorded():
    cell = compute_profiled_cell(
        delta=1.0, beta1=1.0, target_rate=0.3, n=256, rng=RNGState(seed=5),
        rho=0.9, n_quad=24, n_mc=600, n_pop_sample=6000, n_fit_sample=12000,
    )
    assert "beta1_prime" in cell
    assert isinstance(cell["beta1_prime"], float)
    # under high ρ the best-fitting MAR absorbs apparent slope: β₁′ should move ABOVE original β₁
    assert cell["beta1_prime"] > 1.0


def test_profile_returns_mar_with_zero_beta2():
    xm = ConditionalGaussian.synthetic(0.5)
    h1 = SelfCensorParams(beta0=-0.5, beta1=1.0, beta2=1.0)
    grid = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    b1, h0, ll = profile_beta1(h1, xm, 0.3, RNGState(seed=2), grid, n_fit_sample=8000, n_pop_sample=6000)
    assert h0.beta2 == 0.0
    assert isinstance(b1, float)
