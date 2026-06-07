"""Tests for lacuna.survey.regime_descriptors (D2 footprint-geometry metric).

L-moments are validated against CLOSED-FORM theoretical values using deterministic inverse-CDF
"samples" (plotting positions u_i = (i-0.5)/n) — no RNG (Rule 6). Known L-moment ratios:
  • Uniform:      τ3 = 0,      τ4 = 0
  • Exponential:  τ3 = 1/3,    τ4 = 1/6
  • Normal:       τ3 = 0,      τ4 ≈ 0.1226
"""

import math

import numpy as np
import pytest

from lacuna.survey import regime_descriptors as RD


def _inv_cdf_samples(ppf, n=20000):
    u = (np.arange(1, n + 1) - 0.5) / n
    return ppf(u)


def test_l_moments_uniform():
    x = _inv_cdf_samples(lambda u: u)  # Uniform(0,1)
    lm = RD.l_moments(x)
    assert abs(lm["l_skew"] - 0.0) < 1e-3
    assert abs(lm["l_kurt"] - 0.0) < 1e-3


def test_l_moments_exponential():
    x = _inv_cdf_samples(lambda u: -np.log(1 - u))  # Exponential(1)
    lm = RD.l_moments(x)
    assert abs(lm["l_skew"] - 1.0 / 3.0) < 2e-3
    assert abs(lm["l_kurt"] - 1.0 / 6.0) < 2e-3


def test_l_moments_normal_tau4():
    # Normal: τ3 = 0 (symmetry), τ4 ≈ 0.1226 — uses erfinv for the ppf, no RNG.
    ppf = lambda u: math.sqrt(2) * np.vectorize(_erfinv)(2 * u - 1)
    x = _inv_cdf_samples(ppf)
    lm = RD.l_moments(x)
    assert abs(lm["l_skew"]) < 1e-3
    assert abs(lm["l_kurt"] - 0.1226) < 3e-3


def _erfinv(z):
    # Winitzki approximation to erfinv (deterministic; adequate for a τ4 tolerance test).
    a = 0.147
    ln = math.log(1 - z * z)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), z)


def test_l_skew_sign_heavy_right_tail():
    # A right-skewed heavy tail (lognormal-like) ⇒ strongly positive L-skew, bounded L-kurt.
    x = _inv_cdf_samples(lambda u: np.exp(3.0 * (-np.log(1 - u))))  # heavy right tail
    d = RD.shape_descriptors(x)
    assert d["l_skew"] > 0.5
    assert -1.0 <= d["l_skew"] <= 1.0
    assert d["l_kurt"] < 1.0  # L-kurt stays bounded even though moment-kurtosis would explode
    assert d["excess_kurt"] > d["l_kurt"]  # moment-kurtosis far larger under the heavy tail


def test_descriptors_card_negatives_and_keys():
    x = np.concatenate([np.linspace(-5, 5, 200), [-10.0, 7.5]])
    d = RD.shape_descriptors(x)
    assert d["has_neg"] == 1.0
    assert d["card"] >= 30 and d["is_low_card"] == 0.0
    assert set(RD.PRIMARY_KEYS).issubset(d.keys())
    v = RD.regime_vector(x)
    assert v.shape == (4,)


def test_low_card_flag():
    x = np.array([1, 2, 3, 4, 5] * 40, dtype=float)  # Likert-ish, card 5
    d = RD.shape_descriptors(x)
    assert d["card"] == 5 and d["is_low_card"] == 1.0


def test_fail_loud_empty_and_constant():
    with pytest.raises(ValueError):
        RD.shape_descriptors([])
    with pytest.raises(ValueError):
        RD.shape_descriptors([3.0, 3.0, 3.0, 3.0, 3.0])
    with pytest.raises(ValueError):
        RD.shape_descriptors([1.0, 2.0])  # < MIN_ROWS


def test_determinism():
    x = np.concatenate([np.linspace(0, 1, 500), np.linspace(2, 3, 500)])
    assert RD.shape_descriptors(x) == RD.shape_descriptors(x.copy())
