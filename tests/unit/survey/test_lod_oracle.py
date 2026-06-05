"""Tests for lacuna.survey.lod_oracle (LOD Bayes-optimal pre-check gate)."""

import math

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.oracle import gauss_hermite
from lacuna.feasibility.xmodel import ConditionalGaussian
from lacuna.survey.lod_generator import LODParams
from lacuna.survey.lod_oracle import (
    lod_missing_prob,
    lod_oracle_cell,
    lod_solve_beta0,
)


def _cell(delta, seed=1, n_mc=300, n=600):
    xm = ConditionalGaussian.synthetic(0.3)
    return lod_oracle_cell(xm, delta=delta, beta1=1.0, tau_quantile=0.70, target_rate=0.3,
                           n=n, rng=RNGState(seed=seed), n_mc=n_mc, n_nodes=96)


# ---------- missing_prob reductions ----------

def test_missing_prob_delta_zero_reduces_to_sigmoid():
    xm = ConditionalGaussian.synthetic(0.5)
    nodes, weights = gauss_hermite(64)
    z_p = torch.linspace(-2, 2, 50)
    p = LODParams(beta0=-1.0, beta1=1.0, delta=0.0, tau=0.0)
    got = lod_missing_prob(z_p, p, xm, nodes, weights)
    want = torch.sigmoid((-1.0 + 1.0 * z_p).double())
    assert torch.allclose(got, want, atol=1e-9)


def test_solve_beta0_hits_rate():
    xm = ConditionalGaussian.synthetic(0.3)
    nodes, weights = gauss_hermite(96)
    p = lod_solve_beta0(xm, beta1=1.0, delta=2.0, tau=0.5, target_rate=0.3,
                        rng=RNGState(seed=1), nodes=nodes, weights=weights)
    z_p = xm.sample_predictor(40000, RNGState(seed=2))
    rate = float(lod_missing_prob(z_p, p, xm, nodes, weights).mean().item())
    assert abs(rate - 0.3) < 1e-2


# ---------- the gate: δ=0 chance, large δ detectable ----------

def test_delta_zero_near_chance():
    c = _cell(0.0)
    assert abs(c["point_bayes_error"] - 0.5) < 0.07
    assert abs(c["profiled_bayes_error"] - 0.5) < 0.07


def test_large_delta_detectable_on_synthetic():
    c = _cell(3.0)
    # LOD is strongly detectable on clean Gaussian X (NORTH-STAR §2)
    assert c["point_bayes_error"] < 0.3
    assert c["profiled_bayes_error"] < 0.35  # survives the re-fitting MAR null


def test_bayes_error_decreases_with_delta():
    be = [_cell(d)["profiled_bayes_error"] for d in (0.0, 1.0, 3.0)]
    assert be[0] > be[1] > be[2]  # more value-localization -> more separable


def test_determinism():
    a = _cell(2.0, seed=7)
    b = _cell(2.0, seed=7)
    assert a["point_bayes_error"] == b["point_bayes_error"]
    assert a["profiled_bayes_error"] == b["profiled_bayes_error"]


# ---------- failures ----------

def test_rejects_bad_tau_quantile():
    xm = ConditionalGaussian.synthetic(0.3)
    with pytest.raises(ValueError):
        lod_oracle_cell(xm, delta=1.0, beta1=1.0, tau_quantile=1.0, target_rate=0.3,
                        n=500, rng=RNGState(seed=1), n_mc=100)


def test_rejects_bad_target_rate():
    xm = ConditionalGaussian.synthetic(0.3)
    with pytest.raises(ValueError):
        lod_oracle_cell(xm, delta=1.0, beta1=1.0, tau_quantile=0.7, target_rate=1.0,
                        n=500, rng=RNGState(seed=1), n_mc=100)
