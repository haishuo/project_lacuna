"""Tests for lacuna.feasibility.xmodel."""

import math

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.xmodel import ConditionalGaussian


def test_synthetic_conditional_moments():
    rho = 0.6
    xm = ConditionalGaussian.synthetic(rho)
    z_p = torch.tensor([-1.0, 0.0, 2.0])
    mean, var = xm.conditional_mean_var(z_p)
    assert torch.allclose(mean, rho * z_p, atol=1e-6)
    assert abs(var - (1.0 - rho * rho)) < 1e-6


def test_conditional_logpdf_integrates_to_one():
    xm = ConditionalGaussian.synthetic(0.5)
    z_p = torch.tensor([0.7])
    grid = torch.linspace(-12, 12, 4000)
    logp = xm.conditional_logpdf(grid, z_p.expand_as(grid))
    integral = torch.trapz(torch.exp(logp), grid).item()
    assert abs(integral - 1.0) < 1e-3


def test_fit_recovers_correlation():
    rng = np.random.default_rng(0)
    rho = 0.7
    cov = np.array([[1.0, rho], [rho, 1.0]])
    data = rng.multivariate_normal([0, 0], cov, size=20000)
    z_p = torch.tensor(data[:, 0], dtype=torch.float32)
    z_t = torch.tensor(data[:, 1], dtype=torch.float32)
    xm = ConditionalGaussian.fit(z_p, z_t)
    assert abs(xm.cov_pt - rho) < 0.03
    assert xm.fitted is True
    assert xm.descriptor["type"] == "ConditionalGaussian"


def test_sample_conditional_statistics():
    xm = ConditionalGaussian.synthetic(0.5)
    z_p = torch.full((50000,), 1.0)
    z_t = xm.sample_conditional(z_p, RNGState(seed=4))
    assert abs(z_t.mean().item() - 0.5) < 0.02       # mean = rho * z_p = 0.5
    assert abs(z_t.var(unbiased=False).item() - 0.75) < 0.03  # var = 1 - rho^2


# ---------- failure cases ----------

def test_synthetic_rho_out_of_range():
    for bad in (-1.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            ConditionalGaussian.synthetic(bad)


def test_degenerate_conditional_variance_raises():
    # |cov| == sqrt(var_p var_t) -> conditional variance 0 -> reject
    with pytest.raises(ValueError):
        ConditionalGaussian(0.0, 0.0, 1.0, 1.0, 1.0, fitted=True)


def test_fit_requires_enough_rows():
    with pytest.raises(ValueError):
        ConditionalGaussian.fit(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
