"""Tests for lacuna.feasibility.oracle (the analytic Bayes ceiling)."""

import math

import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.oracle import (
    bayes_error_nsample,
    gauss_hermite,
    llr_rows,
    missing_prob,
    per_row_kl,
    solve_beta0_population,
)
from lacuna.feasibility.xmodel import ConditionalGaussian


def _nodes(k=48):
    return gauss_hermite(k)


def test_missing_prob_reduces_to_sigmoid_when_delta_zero():
    """β2==0 ⟹ m(z_p) = σ(β0 + β1 z_p), no integral dependence on the X-model."""
    xm = ConditionalGaussian.synthetic(0.6)
    nodes, weights = _nodes()
    z_p = torch.linspace(-2, 2, 11)
    params = SelfCensorParams(beta0=-0.4, beta1=1.3, beta2=0.0)
    got = missing_prob(z_p, params, xm, nodes, weights)
    expected = torch.sigmoid(-0.4 + 1.3 * z_p)
    assert torch.allclose(got, expected, atol=1e-5)


def test_missing_prob_matches_brute_force_integration():
    xm = ConditionalGaussian.synthetic(0.5)
    nodes, weights = _nodes(64)
    z_p = torch.tensor([-1.0, 0.5])
    params = SelfCensorParams(beta0=-0.3, beta1=0.8, beta2=1.5)
    got = missing_prob(z_p, params, xm, nodes, weights)

    grid = torch.linspace(-15, 15, 6000)
    out = []
    for zp in z_p:
        mean = 0.5 * zp
        var = 1 - 0.25
        dens = torch.exp(-0.5 * (grid - mean) ** 2 / var) / math.sqrt(2 * math.pi * var)
        integ = torch.trapz(dens * torch.sigmoid(-0.3 + 0.8 * zp + 1.5 * grid), grid)
        out.append(integ)
    expected = torch.stack(out)
    assert torch.allclose(got, expected, atol=1e-3)


def test_llr_zero_when_hypotheses_identical():
    xm = ConditionalGaussian.synthetic(0.4)
    nodes, weights = _nodes()
    p = SelfCensorParams(beta0=-0.5, beta1=1.0, beta2=0.0)
    z_p = torch.randn(50)
    z_t = torch.randn(50)
    observed = torch.rand(50) > 0.3
    llr = llr_rows(z_p, z_t, observed, p, p, xm, nodes, weights)
    assert torch.allclose(llr, torch.zeros_like(llr), atol=1e-6)


def test_bayes_error_half_at_delta_zero():
    """δ*=0 ⟹ hypotheses identical ⟹ Bayes error = 0.5 exactly."""
    xm = ConditionalGaussian.synthetic(0.0)
    nodes, weights = _nodes(32)
    p = SelfCensorParams(beta0=-0.8, beta1=1.0, beta2=0.0)
    be = bayes_error_nsample(p, p, xm, n=128, rng=RNGState(seed=0), nodes=nodes, weights=weights, n_mc=500)
    assert abs(be["bayes_error"] - 0.5) < 1e-6


def test_bayes_error_decreases_with_delta():
    """Signal exists at ρ=0: strong δ at matched rate is distinguishable below chance."""
    xm = ConditionalGaussian.synthetic(0.0)
    nodes, weights = _nodes(32)
    rng = RNGState(seed=5)
    rate = 0.3
    p_h0 = solve_beta0_population(xm, 0.0, 0.0, rate, rng.spawn(), nodes, weights, n_sample=8000)

    def be_for(delta, n=256):
        p_h1 = solve_beta0_population(xm, 0.0, delta, rate, rng.spawn(), nodes, weights, n_sample=8000)
        return bayes_error_nsample(p_h0, p_h1, xm, n=n, rng=rng.spawn(), nodes=nodes, weights=weights, n_mc=800)["bayes_error"]

    be_small = be_for(0.5)
    be_large = be_for(2.0)
    assert be_large < be_small             # more departure -> more distinguishable
    assert be_large < 0.45                 # strong δ leaves real observed-data signal


def test_per_row_kl_nonnegative_and_zero_at_delta_zero():
    xm = ConditionalGaussian.synthetic(0.0)
    nodes, weights = _nodes(32)
    p = SelfCensorParams(beta0=-0.8, beta1=1.0, beta2=0.0)
    kl10, kl01 = per_row_kl(p, p, xm, RNGState(seed=1), nodes, weights, n_sample=20000)
    assert abs(kl10) < 1e-6 and abs(kl01) < 1e-6

    p1 = SelfCensorParams(beta0=-0.8, beta1=1.0, beta2=1.5)
    kl10b, kl01b = per_row_kl(p, p1, xm, RNGState(seed=1), nodes, weights, n_sample=20000)
    assert kl10b > 0 and kl01b > 0
