"""Tests for lacuna.training.composition_calibration — the Stage-D calibration headline (ADR-0007)."""

import numpy as np
import pytest

from lacuna.models.composition_head import cant_tell_mass
import torch
from lacuna.training.composition_calibration import (
    apply_temperature, region_prob_ge, default_query_set, collect_query_pairs,
    expected_calibration_error, brier_score, reliability_curve, query_ece, fit_temperature,
)


# ---------------------------------------------------------------------------
# Region probability (exact Beta marginal) vs Monte Carlo
# ---------------------------------------------------------------------------

def test_region_prob_matches_monte_carlo():
    rng = np.random.default_rng(0)
    alpha = np.array([[2.0, 3.0, 5.0], [10.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    for c in range(3):
        for t in (0.2, 0.5, 0.8):
            exact = region_prob_ge(alpha, c, t)
            mc = np.array([(rng.dirichlet(a, size=40000)[:, c] >= t).mean() for a in alpha])
            assert np.allclose(exact, mc, atol=0.02), f"c={c} t={t}: {exact} vs {mc}"


def test_region_prob_concentrated():
    alpha = np.array([[1.0, 1.0, 60.0]])      # mass on MNAR
    assert region_prob_ge(alpha, 2, 0.7)[0] > 0.95
    assert region_prob_ge(alpha, 0, 0.3)[0] < 0.05


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def test_temperature_flattens_toward_uniform_and_changes_vacuity():
    """Evidential temperature interpolates toward the uniform prior (like logit temperature flattens
    the softmax): tau>1 raises the can't-tell mass and pulls the mean toward 1/K; tau<1 sharpens."""
    alpha = np.array([[6.0, 9.0, 15.0]])
    uniform = np.array([1 / 3, 1 / 3, 1 / 3])
    mean_id = alpha[0] / alpha[0].sum()
    mean_hot = apply_temperature(alpha, 6.0)[0]; mean_hot = mean_hot / mean_hot.sum()
    mean_cold = apply_temperature(alpha, 0.4)[0]; mean_cold = mean_cold / mean_cold.sum()
    # hotter => closer to uniform; colder => further
    assert np.abs(mean_hot - uniform).sum() < np.abs(mean_id - uniform).sum()
    assert np.abs(mean_cold - uniform).sum() > np.abs(mean_id - uniform).sum()
    # vacuity (can't-tell mass) monotone in temperature
    v_hi = float(cant_tell_mass(torch.tensor(apply_temperature(alpha, 4.0))))
    v_lo = float(cant_tell_mass(torch.tensor(apply_temperature(alpha, 0.5))))
    v_id = float(cant_tell_mass(torch.tensor(alpha)))
    assert v_hi > v_id > v_lo


# ---------------------------------------------------------------------------
# ECE / reliability
# ---------------------------------------------------------------------------

def test_ece_zero_when_calibrated():
    # preds all 0.5 with empirical frequency 0.5 -> perfectly calibrated
    preds = np.full(1000, 0.5)
    inds = np.array([0.0, 1.0] * 500)
    assert expected_calibration_error(preds, inds) < 1e-6


def test_ece_high_when_overconfident():
    preds = np.concatenate([np.full(500, 0.95), np.full(500, 0.05)])
    inds = np.concatenate([np.full(500, 0.5), np.full(500, 0.5)])   # actually 50/50
    assert expected_calibration_error(preds, inds) > 0.4


def test_brier_rewards_resolution_over_constant():
    """A perfectly-resolved predictor beats a constant base-rate predictor on Brier, even though
    both can be reliable — the reason Brier (not ECE) is the bar to beat prior-only."""
    inds = np.array([1.0, 1.0, 0.0, 0.0])
    resolved = np.array([0.99, 0.99, 0.01, 0.01])     # calibrated AND resolved
    constant = np.array([0.5, 0.5, 0.5, 0.5])         # base rate: reliable, no resolution
    assert brier_score(resolved, inds) < brier_score(constant, inds)


def test_reliability_curve_bins():
    preds = np.array([0.05, 0.15, 0.95, 0.85])
    inds = np.array([0.0, 0.0, 1.0, 1.0])
    curve = reliability_curve(preds, inds, n_bins=10)
    assert all(0.0 <= conf <= 1.0 and 0.0 <= freq <= 1.0 and cnt > 0 for conf, freq, cnt in curve)


# ---------------------------------------------------------------------------
# Fitting temperature reduces ECE
# ---------------------------------------------------------------------------

def test_fit_temperature_reduces_ece_on_overconfident_model():
    rng = np.random.default_rng(1)
    realized = rng.dirichlet([2.0, 2.0, 2.0], size=600)
    noisy = realized + rng.normal(0, 0.08, realized.shape)
    noisy = np.clip(noisy, 1e-3, None); noisy /= noisy.sum(1, keepdims=True)
    alpha = 1.0 + 60.0 * noisy                                  # very overconfident
    queries = default_query_set()
    tau = fit_temperature(alpha, realized, queries)
    ece_before = query_ece(alpha, realized, queries)
    ece_after = query_ece(apply_temperature(alpha, tau), realized, queries)
    assert tau > 1.0                                            # overconfident -> spread out
    assert ece_after < ece_before
    assert ece_after <= ece_before                              # never worse than uncalibrated


def test_default_query_set_size():
    assert len(default_query_set()) == 15                       # 3 classes x 5 thresholds


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.ones(3), np.zeros((2, 1))])
def test_bad_alpha_shape_raises(bad):
    with pytest.raises(ValueError, match=r"alpha must be"):
        region_prob_ge(bad, 0, 0.5)


def test_nonpositive_alpha_raises():
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(np.array([[1.0, 0.0, 2.0]]), 2.0)


def test_bad_tau_raises():
    with pytest.raises(ValueError, match="tau"):
        apply_temperature(np.array([[1.0, 1.0, 1.0]]), 0.0)


@pytest.mark.parametrize("t", [0.0, 1.0, 1.5])
def test_bad_threshold_raises(t):
    with pytest.raises(ValueError, match="threshold"):
        region_prob_ge(np.array([[1.0, 1.0, 1.0]]), 0, t)


def test_realized_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        collect_query_pairs(np.ones((4, 3)), np.ones((4, 2)) / 2, default_query_set())
