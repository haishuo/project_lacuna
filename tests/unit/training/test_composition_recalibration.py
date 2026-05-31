"""Tests for lacuna.training.composition_recalibration — Stage-D-v2 feature-conditional calibration."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.models.composition_head import composition_mean, cant_tell_mass
from lacuna.training.composition_loss import dirichlet_nll
from lacuna.training.composition_recalibration import FeatureTemperature, fit_feature_temperature


def _feasible_dataset(n=400, seed=0):
    """Half 'easy' (concentrated AND correct), half 'hard' (concentrated but WRONG); a footprint
    feature flags which. A global temperature cannot exploit the flag; a feature-conditional one can."""
    rng = np.random.default_rng(seed)
    realized = rng.dirichlet([2.0, 2.0, 2.0], size=n)
    hard = (rng.random(n) < 0.5)
    wrong = np.roll(realized, 1, axis=1)                      # a deterministically-wrong mean direction
    p = np.where(hard[:, None], wrong, realized)
    alpha = 1.0 + 30.0 * p                                    # concentrated either way (overconfident)
    feats = np.column_stack([hard.astype(float), rng.normal(size=(n, 4))])   # feature 0 = the signal
    return (torch.tensor(feats, dtype=torch.float32),
            torch.tensor(alpha, dtype=torch.float32),
            torch.tensor(realized, dtype=torch.float32),
            torch.tensor(hard))


# ---------------------------------------------------------------------------
# Module mechanics
# ---------------------------------------------------------------------------

def test_temperature_bounded():
    m = FeatureTemperature(5, tau_min=0.5, tau_max=25.0)
    m.train()  # batchnorm needs >1 sample in train; use eval for a single forward
    m.eval()
    tau = m(torch.randn(16, 5))
    assert tau.shape == (16,)
    assert bool((tau >= 0.5).all()) and bool((tau <= 25.0).all())


def test_recalibrate_preserves_argmax():
    """The no-overfit guarantee: temperature scaling cannot change which composition is predicted."""
    m = FeatureTemperature(5)
    m.eval()
    alpha = torch.tensor([[2.0, 9.0, 4.0], [10.0, 1.0, 3.0], [1.0, 2.0, 8.0]])
    feats = torch.randn(3, 5)
    cal = m.recalibrate(alpha, feats)
    assert torch.equal(composition_mean(alpha).argmax(-1), composition_mean(cal).argmax(-1))


def test_fit_is_deterministic():
    f, a, t, _ = _feasible_dataset()
    m1 = fit_feature_temperature(f, a, t, RNGState(seed=3), epochs=50)
    m2 = fit_feature_temperature(f, a, t, RNGState(seed=3), epochs=50)
    with torch.no_grad():
        assert torch.allclose(m1(f), m2(f))


# ---------------------------------------------------------------------------
# The headline: feature-conditional beats a global temperature when feasible
# ---------------------------------------------------------------------------

def _best_global_nll(alpha, target):
    best = float("inf")
    for tau in np.geomspace(0.25, 25, 60):
        a_cal = 1.0 + (alpha - 1.0) / float(tau)
        best = min(best, float(dirichlet_nll(a_cal, target).mean()))
    return best


def test_feature_temperature_beats_global_when_predictable():
    f, a, t, hard = _feasible_dataset(n=600, seed=1)
    model = fit_feature_temperature(f, a, t, RNGState(seed=2))
    with torch.no_grad():
        nll_feat = float(dirichlet_nll(model.recalibrate(a, f), t).mean())
    nll_global = _best_global_nll(a, t)
    assert nll_feat < nll_global, f"feature NLL {nll_feat:.3f} not below best global {nll_global:.3f}"
    # and it learns to spread the HARD (wrong) ones more than the easy ones
    with torch.no_grad():
        tau = model(f)
    assert float(tau[hard].mean()) > float(tau[~hard].mean())


def test_feature_temperature_improves_vacuity_error_ranking():
    """Vacuity should track error better after feature-conditional recalibration (the Stage-D-v2 goal)."""
    f, a, t, _ = _feasible_dataset(n=600, seed=4)
    err = (composition_mean(a) - t).abs().sum(-1)
    corr_before = float(torch.corrcoef(torch.stack([cant_tell_mass(a), err]))[0, 1])
    model = fit_feature_temperature(f, a, t, RNGState(seed=5))
    with torch.no_grad():
        cal = model.recalibrate(a, f)
    corr_after = float(torch.corrcoef(torch.stack([cant_tell_mass(cal), err]))[0, 1])
    assert corr_after > corr_before
    assert corr_after > 0.0          # uncertainty now positively tracks error


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_bad_tau_bounds_raise():
    with pytest.raises(ValueError, match="tau_min < tau_max"):
        FeatureTemperature(5, tau_min=2.0, tau_max=1.0)


def test_forward_bad_shape_raises():
    m = FeatureTemperature(5)
    with pytest.raises(ValueError, match="footprint must be"):
        m(torch.randn(5))


def test_fit_shape_mismatch_raises():
    with pytest.raises(ValueError, match="required with matching"):
        fit_feature_temperature(torch.randn(10, 5), torch.rand(10, 3) + 1, torch.rand(8, 3),
                                RNGState(seed=0))
