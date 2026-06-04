"""Tests for lacuna.feasibility.xmodel_mv."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.xmodel_mv import MultivariateGaussianX, build_p1ra_corr


def test_build_p1ra_corr_is_psd_and_correct():
    for rho_orig in [0.0, 0.3, 0.9]:
        for rho_a in [0.0, 0.6, 0.95, 0.99]:
            corr = build_p1ra_corr(rho_orig, rho_a)
            MultivariateGaussianX(corr)  # constructs ⇒ PSD ok
            assert abs(corr[2, 0].item() - rho_orig) < 1e-12   # z_p,z_t
            assert abs(corr[2, 1].item() - rho_a) < 1e-12       # z_a,z_t
            assert abs(corr[0, 1].item() - rho_orig * rho_a) < 1e-12  # z_p,z_a (cond-indep value)


def test_sample_reproduces_correlation():
    xm = MultivariateGaussianX(build_p1ra_corr(0.9, 0.6))
    X = xm.sample(200000, RNGState(seed=0))
    c = torch.corrcoef(X.T)
    assert abs(c[2, 0].item() - 0.9) < 0.01
    assert abs(c[2, 1].item() - 0.6) < 0.01
    assert abs(c[0, 1].item() - 0.54) < 0.01  # 0.9*0.6


def test_conditional_single_obs_matches_bivariate_formula():
    xm = MultivariateGaussianX(build_p1ra_corr(0.7, 0.4))
    z_p = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    mean, var = xm.conditional_mean_var(2, (0,), z_p)   # z_t | z_p only
    assert torch.allclose(mean, 0.7 * z_p, atol=1e-9)   # corr(z_t,z_p)=0.7
    assert abs(var - (1 - 0.49)) < 1e-9


def test_conditional_two_obs_solves_normal_equations():
    xm = MultivariateGaussianX(build_p1ra_corr(0.8, 0.5))
    b, v = xm._conditioner(2, (0, 1))
    Soo = xm.corr[[0, 1]][:, [0, 1]]
    Sot = xm.corr[2, [0, 1]]
    assert torch.allclose(Soo @ b, Sot, atol=1e-9)       # b solves Σoo b = Σot
    assert 0.0 < v < 1.0


def test_non_psd_matrix_raises():
    bad = torch.tensor([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        MultivariateGaussianX(bad)


def test_non_symmetric_or_bad_diagonal_raises():
    with pytest.raises(ValueError):
        MultivariateGaussianX(torch.tensor([[1.0, 0.5], [0.4, 1.0]]))
    with pytest.raises(ValueError):
        MultivariateGaussianX(torch.tensor([[2.0, 0.5], [0.5, 1.0]]))


def test_build_rejects_out_of_range():
    with pytest.raises(ValueError):
        build_p1ra_corr(1.0, 0.5)
