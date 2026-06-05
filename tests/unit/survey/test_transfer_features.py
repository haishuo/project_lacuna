"""Tests for lacuna.survey.transfer_features."""

import numpy as np
import pytest
import torch

from lacuna.survey.transfer_features import (
    FEATURE_NAMES,
    N_TRANSFER_FEATURES,
    compute_transfer_features,
)


def _xy(n=3000, seed=0, rho=0.6):
    g = np.random.default_rng(seed)
    p = g.standard_normal(n)
    t = rho * p + np.sqrt(1 - rho ** 2) * g.standard_normal(n)
    x = np.stack([p, t], axis=1).astype("float32")  # col0=predictor, col1=target
    return torch.from_numpy(x)


# ---------- shape / determinism ----------

def test_shape_and_schema():
    x = _xy()
    f = compute_transfer_features(x, torch.ones_like(x, dtype=torch.bool), 1)
    assert f.shape == (N_TRANSFER_FEATURES,) and f.dtype == torch.float32
    assert N_TRANSFER_FEATURES == len(FEATURE_NAMES) == 10


def test_determinism():
    x = _xy()
    r = torch.rand(x.shape) > 0.3
    a = compute_transfer_features(x, r, 1)
    b = compute_transfer_features(x, r, 1)
    assert torch.equal(a, b)


# ---------- scale/shift invariance ----------

def test_affine_invariance_of_target():
    x = _xy()
    r = torch.ones_like(x, dtype=torch.bool)
    base = compute_transfer_features(x, r, 1)
    x2 = x.clone(); x2[:, 1] = x[:, 1] * 5.0 - 2.0  # affine transform of the target
    scaled = compute_transfer_features(x2, r, 1)
    # all features are ratios / z-diffs / correlations ⇒ invariant to target affine scaling
    assert torch.allclose(base, scaled, atol=1e-3)


# ---------- missing handling ----------

def test_missing_rate_and_censored_excluded():
    x = _xy()
    r = torch.ones(x.shape, dtype=torch.bool)
    r[:900, 1] = False  # 30% target censored
    f = compute_transfer_features(x, r, 1)
    assert abs(float(f[0]) - 0.30) < 1e-6


def test_few_observed_safe():
    x = _xy(n=100)
    r = torch.ones(100, 2, dtype=torch.bool)
    r[6:, 1] = False  # only 6 observed (< MIN_OBS)
    f = compute_transfer_features(x, r, 1)
    assert torch.isfinite(f).all()
    assert float(f[0]) == pytest.approx(0.94)


# ---------- discriminative direction (a unit check, NOT a training target) ----------

def test_top_coding_makes_reach_deficit_negative():
    x = _xy(seed=2)
    r = torch.ones(x.shape, dtype=torch.bool)
    full = compute_transfer_features(x, r, 1)
    # top-code: censor the upper 30% of the TARGET (col 1)
    thr = torch.quantile(x[:, 1].double(), 0.70)
    r_tc = r.clone(); r_tc[:, 1] = x[:, 1].double() <= thr
    tc = compute_transfer_features(x, r_tc, 1)
    i_ruler = FEATURE_NAMES.index("ruler_max_diff")
    i_reach = FEATURE_NAMES.index("res_reach_deficit")
    # under upper truncation the target reaches less far up than the predictor, and the residual
    # upper reach is deficient — both move down vs the uncensored case
    assert tc[i_ruler] < full[i_ruler]
    assert tc[i_reach] < full[i_reach]


# ---------- failures ----------

def test_bad_target_idx():
    with pytest.raises(ValueError):
        compute_transfer_features(_xy(), torch.ones(3000, 2, dtype=torch.bool), 9)
