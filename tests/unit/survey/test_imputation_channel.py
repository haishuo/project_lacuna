"""Tests for lacuna.survey.imputation_channel (G1 evaluation instrumentation).

Normal: imputers recover a linear relation; MNAR-style punching produces the predicted signed paired
differences; MCAR-vs-MCAR pairing is ~zero. Edge: minimum sizes; determinism. Failure: degenerate
inputs raise (Rule 1).
"""

import numpy as np
import pytest

from lacuna.core.rng import RNGState
from lacuna.survey import imputation_channel as IC


def _toy(n=400, rho=0.8, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=n)
    t = rho * p + np.sqrt(1 - rho**2) * rng.normal(size=n)
    return np.column_stack([p, t]), 1  # target_idx = 1


def _self_censor_mask(t, rate=0.3, delta=4.0, seed=9):
    rng = np.random.default_rng(seed)
    z = (t - t.mean()) / t.std()
    logits = -1.5 + delta * (z > np.quantile(z, 1 - rate * 1.5))
    miss = rng.random(len(t)) < 1 / (1 + np.exp(-logits))
    return ~miss  # observed-mask


def test_linear_imputer_recovers_relation():
    X, t_idx = _toy()
    obs = np.ones(len(X), dtype=bool); obs[:80] = False  # punch 80 cells, arbitrary
    feats = IC.imputer_channel_features(X, t_idx, obs, obs, imputer="linear", seed=0)
    # identical masks => every paired difference is exactly zero
    assert all(abs(v) < 1e-12 for v in feats.values())


def test_mcar_vs_mcar_pair_near_zero_and_mnar_signed():
    X, t_idx = _toy()
    t = X[:, t_idx]
    mech = _self_censor_mask(t)            # upper-tail punching (MNAR-style)
    mcar = IC.mcar_pair_mask(mech, RNGState(seed=5))
    assert mech.sum() != len(mech) and (~mcar).sum() == (~mech).sum()
    for imp in ("linear", "rf"):
        f = IC.imputer_channel_features(X, t_idx, mech, mcar, imputer=imp, seed=0)
        # truth at mech-punched cells sits ABOVE the MAR prediction: mu - y < 0, PIT mass near 1
        assert f["d_B"] < 0, (imp, f)
        assert f["d_PIT_tail"] > 0, (imp, f)


def test_all_imputers_run_and_are_deterministic():
    X, t_idx = _toy(n=250)
    mech = _self_censor_mask(X[:, t_idx], seed=11)
    mcar = IC.mcar_pair_mask(mech, RNGState(seed=2))
    for imp in IC.IMPUTERS:
        a = IC.imputer_channel_features(X, t_idx, mech, mcar, imputer=imp, seed=7)
        b = IC.imputer_channel_features(X, t_idx, mech, mcar, imputer=imp, seed=7)
        assert a == b, imp
        assert set(a) == set(IC.FEATURE_NAMES)


def test_mask_stats_pit_uniform_under_correct_model():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=4000)
    stats = IC.mask_stats(truth, np.zeros(4000), np.ones(4000))  # correct N(0,1) predictive
    assert abs(stats["PIT_loc"]) < 0.03
    assert abs(stats["PIT_tail"] - 0.1) < 0.02
    assert abs(stats["cov80"] - 0.8) < 0.03
    assert abs(stats["B"]) < 0.05


def test_fail_loud():
    X, t_idx = _toy(n=60)
    with pytest.raises(ValueError):  # too few observed
        m = np.zeros(60, dtype=bool); m[:10] = True
        IC.imputer_channel_features(X, t_idx, m, m, imputer="linear", seed=0)
    with pytest.raises(ValueError):  # too few punched
        m = np.ones(60, dtype=bool); m[0] = False
        IC.imputer_channel_features(X, t_idx, m, m, imputer="linear", seed=0)
    with pytest.raises(ValueError):  # unknown imputer
        m = np.ones(60, dtype=bool); m[:20] = False
        IC.imputer_channel_features(X, t_idx, m, m, imputer="gan", seed=0)
    with pytest.raises(ValueError):  # 1 column = no predictors
        IC.imputer_channel_features(X[:, [1]], 0, m, m, imputer="linear", seed=0)
    with pytest.raises(ValueError):  # non-finite
        Xb = X.copy(); Xb[0, 0] = np.nan
        IC.imputer_channel_features(Xb, t_idx, m, m, imputer="linear", seed=0)


def test_mcar_pair_mask_guards():
    with pytest.raises(ValueError):
        IC.mcar_pair_mask(np.ones(30, dtype=bool), RNGState(seed=1))  # nothing punched


# ---------- no-truth-conditional station (empty-cell prereg e1374f9) ----------

def _view(X, t_idx, obs_mask):
    v = X.copy().astype(float)
    v[~obs_mask, t_idx] = np.nan
    return v


def test_no_truth_structural_truth_blindness():
    # outputs identical whatever values sit at punched positions BEFORE NaN-ing — and NaN is enforced.
    X, t_idx = _toy()
    mech = _self_censor_mask(X[:, t_idx])
    v = _view(X, t_idx, mech)
    a = IC.no_truth_features(v, t_idx, imputer="linear", seed=3, holdout_rng=RNGState(seed=11))
    X2 = X.copy(); X2[~mech, t_idx] = 1e9   # poison the truth
    b = IC.no_truth_features(_view(X2, t_idx, mech), t_idx, imputer="linear", seed=3,
                             holdout_rng=RNGState(seed=11))
    assert a == b  # cannot depend on the deleted values


def test_no_truth_determinism_and_names():
    X, t_idx = _toy(n=300)
    mech = _self_censor_mask(X[:, t_idx], seed=4)
    v = _view(X, t_idx, mech)
    a = IC.no_truth_features(v, t_idx, imputer="rf", seed=5, holdout_rng=RNGState(seed=7))
    b = IC.no_truth_features(v, t_idx, imputer="rf", seed=5, holdout_rng=RNGState(seed=7))
    assert a == b and set(a) == set(IC.NO_TRUTH_FEATURE_NAMES) and len(a) == 16


def test_no_truth_residual_signature_direction():
    # own-value upper-tail punching truncates observed residuals from above => negative skew/deficit.
    X, t_idx = _toy(n=600, rho=0.6)
    mech = _self_censor_mask(X[:, t_idx], delta=6.0, seed=2)
    f = IC.no_truth_features(_view(X, t_idx, mech), t_idx, imputer="linear", seed=0,
                             holdout_rng=RNGState(seed=3))
    assert f["r_skew"] < 0 and f["r_reach_deficit"] < 0


def test_no_truth_fail_loud():
    X, t_idx = _toy(n=200)
    mech = _self_censor_mask(X[:, t_idx])
    with pytest.raises(ValueError):  # NaN predictor forbidden (only the target is punched)
        v = _view(X, t_idx, mech); v[0, 0] = np.nan
        IC.no_truth_features(v, t_idx, imputer="linear", seed=0, holdout_rng=RNGState(seed=1))
    with pytest.raises(ValueError):  # nothing punched
        IC.no_truth_features(X.astype(float), t_idx, imputer="linear", seed=0,
                             holdout_rng=RNGState(seed=1))