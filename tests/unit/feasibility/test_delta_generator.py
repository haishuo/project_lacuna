"""Tests for lacuna.feasibility.delta_generator."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.feasibility.delta_generator import (
    apply_self_censor,
    expected_missing_rate,
    solve_beta0_for_rate,
)


def _independent_X(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((n, 2)), dtype=torch.float32)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 0.0
    return float((a * b).sum().item() / denom)


# ---------- normal cases ----------

def test_rate_matching_hits_target():
    X = _independent_X()
    from lacuna.data.semisynthetic import _zscore_columns

    Z = _zscore_columns(X)
    z_p, z_t = Z[:, 0], Z[:, 1]
    for target in (0.1, 0.3, 0.5):
        beta0 = solve_beta0_for_rate(z_p, z_t, beta1=1.0, beta2=1.0, target_rate=target)
        got = expected_missing_rate(z_p, z_t, beta0, 1.0, 1.0)
        assert abs(got - target) < 1e-3


def test_apply_shapes_and_single_column_censored():
    X = _independent_X()
    res = apply_self_censor(X, target_idx=1, predictor_idx=0, beta1=1.0, delta=1.0,
                            target_rate=0.3, rng=RNGState(seed=7))
    assert res.mask.shape == X.shape
    assert res.mask.dtype == torch.bool
    # only the target column may be missing; all other columns fully observed
    assert res.mask[:, 0].all()
    assert not res.mask[:, 1].all()
    assert abs(res.realized_rate - 0.3) < 0.03  # large-n sampled rate near target


def test_delta_zero_is_mar_independent_of_own_value():
    """At δ=0 missingness depends only on the observed predictor, not the own value."""
    X = _independent_X(seed=1)
    res = apply_self_censor(X, target_idx=1, predictor_idx=0, beta1=2.0, delta=0.0,
                            target_rate=0.3, rng=RNGState(seed=3))
    from lacuna.data.semisynthetic import _zscore_columns

    z_t = _zscore_columns(X)[:, 1]
    missing = (~res.mask[:, 1]).float()
    assert abs(_pearson(missing, z_t)) < 0.05  # ~0 correlation with own value


def test_delta_positive_is_mnar_correlated_with_own_value():
    X = _independent_X(seed=1)
    res = apply_self_censor(X, target_idx=1, predictor_idx=0, beta1=0.0, delta=2.0,
                            target_rate=0.3, rng=RNGState(seed=3))
    from lacuna.data.semisynthetic import _zscore_columns

    z_t = _zscore_columns(X)[:, 1]
    missing = (~res.mask[:, 1]).float()
    assert _pearson(missing, z_t) > 0.15  # clearly positive: own-value self-censoring


def test_determinism_same_seed_same_mask():
    X = _independent_X(seed=2)
    a = apply_self_censor(X, 1, 0, 1.0, 1.0, 0.3, RNGState(seed=99))
    b = apply_self_censor(X, 1, 0, 1.0, 1.0, 0.3, RNGState(seed=99))
    assert torch.equal(a.mask, b.mask)
    assert a.params == b.params


# ---------- failure cases ----------

def test_rejects_bad_target_rate():
    X = _independent_X(n=100)
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError):
            apply_self_censor(X, 1, 0, 1.0, 1.0, bad, RNGState(seed=1))


def test_rejects_same_target_predictor():
    X = _independent_X(n=100)
    with pytest.raises(ValueError):
        apply_self_censor(X, 1, 1, 1.0, 1.0, 0.3, RNGState(seed=1))


def test_rejects_too_few_columns():
    X = torch.randn(100, 1)
    with pytest.raises(ValueError):
        apply_self_censor(X, 0, 0, 1.0, 1.0, 0.3, RNGState(seed=1))


def test_rejects_negative_delta_or_beta1():
    X = _independent_X(n=100)
    with pytest.raises(ValueError):
        apply_self_censor(X, 1, 0, 1.0, -0.5, 0.3, RNGState(seed=1))
    with pytest.raises(ValueError):
        apply_self_censor(X, 1, 0, -1.0, 0.5, 0.3, RNGState(seed=1))


def test_rejects_constant_target_column():
    X = torch.randn(200, 2)
    X[:, 1] = 3.0  # constant target
    with pytest.raises(ValueError):
        apply_self_censor(X, target_idx=1, predictor_idx=0, beta1=1.0, delta=1.0,
                          target_rate=0.3, rng=RNGState(seed=1))


def test_unbracketable_rate_raises():
    X = _independent_X(n=200)
    from lacuna.data.semisynthetic import _zscore_columns

    Z = _zscore_columns(X)
    with pytest.raises(ValueError):
        solve_beta0_for_rate(Z[:, 0], Z[:, 1], 1.0, 1.0, 0.3, bracket=(10.0, 50.0))
