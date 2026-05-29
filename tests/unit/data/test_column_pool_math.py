"""Tests for lacuna.data._column_pool_math — marginal-rate math shared by the column pools."""

import math

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data._column_pool_math import logit, comp_beta0, norm_ppf


# ---------------------------------------------------------------------------
# logit
# ---------------------------------------------------------------------------

def test_logit_known_values():
    assert logit(0.5) == pytest.approx(0.0)
    assert logit(0.25) == pytest.approx(math.log(1 / 3))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_logit_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="0 < p < 1"):
        logit(bad)


# ---------------------------------------------------------------------------
# comp_beta0 — centres E[sigmoid(beta0 + slope*Z)] at r
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", [0.1, 0.25, 0.4])
@pytest.mark.parametrize("slope", [0.5, 1.5, 2.5])
def test_comp_beta0_centres_marginal_at_r(r, slope):
    """The compensated intercept makes the realised sigmoid marginal ≈ r for N(0,1) Z."""
    z = RNGState(seed=7).randn(200000)
    p = torch.sigmoid(comp_beta0(r, slope) + slope * z)
    realised = p.mean().item()
    assert abs(realised - r) < 0.02, f"r={r} slope={slope}: realised {realised:.3f}"


def test_comp_beta0_zero_slope_is_plain_logit():
    assert comp_beta0(0.25, 0.0) == pytest.approx(logit(0.25))


# ---------------------------------------------------------------------------
# norm_ppf — standard-normal quantile
# ---------------------------------------------------------------------------

def test_norm_ppf_median_is_zero():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("p,expected", [(0.975, 1.959964), (0.025, -1.959964), (0.84134, 1.0)])
def test_norm_ppf_known_quantiles(p, expected):
    assert norm_ppf(p) == pytest.approx(expected, abs=1e-3)


def test_norm_ppf_is_inverse_of_normal_cdf():
    """norm_ppf round-trips through the standard-normal CDF."""
    normal = torch.distributions.Normal(0.0, 1.0)
    for p in (0.05, 0.3, 0.5, 0.7, 0.95):
        x = norm_ppf(p)
        assert normal.cdf(torch.tensor(x)).item() == pytest.approx(p, abs=1e-4)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
def test_norm_ppf_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="0 < p < 1"):
        norm_ppf(bad)
