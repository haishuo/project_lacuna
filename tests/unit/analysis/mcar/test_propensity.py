"""Tests for propensity_mcar_test.

Covers the analytical-null default (fast, Mann-Whitney-U asymptotic)
and the opt-in permutation null. Both must produce sensible calibration
under MCAR and reject under MAR.
"""

import numpy as np
import pytest

from lacuna.analysis.mcar import (
    NonparametricMCARResult,
    propensity_mcar_test,
)


def _mcar_matrix(n: int = 60, d: int = 4, p_miss: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    mask = rng.random((n, d)) < p_miss
    X[mask] = np.nan
    return X


def _mar_matrix(n: int = 120, d: int = 4, seed: int = 0):
    """Column 1's missingness is driven by column 0's value."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    drop_prob = 1.0 / (1.0 + np.exp(-3.0 * X[:, 0]))
    mask = rng.random(n) < drop_prob
    X[mask, 1] = np.nan
    # Background MCAR in col 2 so we have multiple pooled-AUC contributors.
    mask2 = rng.random(n) < 0.15
    X[mask2, 2] = np.nan
    return X


# ---------------------------------------------------------------------
# Analytical null (default) — normal cases
# ---------------------------------------------------------------------


def test_analytical_on_mcar_data_does_not_reject():
    X = _mcar_matrix(n=80, d=4, p_miss=0.25, seed=0)
    r = propensity_mcar_test(X, seed=0, cv_folds=3)
    assert isinstance(r, NonparametricMCARResult)
    assert "Propensity-AUC" in r.method
    assert "analytical" in r.method
    assert 0.0 <= r.statistic <= 0.5
    assert r.rejected is False


def test_analytical_rejects_mar_data():
    X = _mar_matrix(n=200, d=4, seed=0)
    r = propensity_mcar_test(X, seed=0, cv_folds=3)
    assert r.rejected is True
    assert r.statistic > 0.05


def test_analytical_reproducible_same_seed():
    X = _mcar_matrix(n=60, d=4, seed=1)
    r1 = propensity_mcar_test(X, seed=42, cv_folds=3)
    r2 = propensity_mcar_test(X, seed=42, cv_folds=3)
    assert r1.statistic == r2.statistic
    assert r1.p_value == r2.p_value


def test_default_model_is_hgb():
    X = _mcar_matrix(n=60, d=4, seed=0)
    r = propensity_mcar_test(X, seed=0, cv_folds=3)
    assert r.extra["model"] == "hgb"
    assert "HGB" in r.method


# ---------------------------------------------------------------------
# Permutation null (opt-in) — still works
# ---------------------------------------------------------------------


def test_permutation_null_runs_and_rejects_mar():
    X = _mar_matrix(n=200, d=4, seed=0)
    r = propensity_mcar_test(
        X, null="permutation", n_permutations=49, seed=0, cv_folds=3,
    )
    assert r.rejected is True
    assert "perm=49" in r.method
    assert "permutation_null_mean_auc" in r.extra


def test_permutation_reproducible_same_seed():
    X = _mcar_matrix(n=60, d=4, seed=1)
    kw = dict(null="permutation", n_permutations=19, cv_folds=3)
    r1 = propensity_mcar_test(X, seed=42, **kw)
    r2 = propensity_mcar_test(X, seed=42, **kw)
    assert r1.p_value == r2.p_value


# ---------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------


def test_rf_model_option_runs():
    X = _mcar_matrix(n=60, d=4, seed=0)
    r = propensity_mcar_test(X, seed=0, cv_folds=3, model="rf")
    assert "RF" in r.method


def test_gbm_model_option_runs():
    X = _mcar_matrix(n=60, d=4, seed=0)
    r = propensity_mcar_test(X, seed=0, cv_folds=3, model="gbm")
    assert "GBM" in r.method


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


def test_column_without_missingness_is_ignored():
    X = _mcar_matrix(n=60, d=4, seed=0)
    X[:, 3] = np.linspace(0, 1, 60)  # fully observed
    r = propensity_mcar_test(X, seed=0, cv_folds=3)
    per_col = dict(r.extra["per_column_auc"])
    assert 3 not in per_col


# ---------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------


def test_raises_on_1d_input():
    with pytest.raises(ValueError, match="2D"):
        propensity_mcar_test(np.array([np.nan, 1.0, 2.0]))


def test_raises_on_small_n():
    X = _mcar_matrix(n=5, d=4)
    with pytest.raises(ValueError, match="at least 10 rows"):
        propensity_mcar_test(X)


def test_raises_on_single_column():
    X = np.full((30, 1), np.nan)
    X[::2] = 1.0
    with pytest.raises(ValueError, match="at least 2 columns"):
        propensity_mcar_test(X)


def test_raises_on_matrix_with_no_missingness():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    with pytest.raises(ValueError, match="missing and observed"):
        propensity_mcar_test(X)


def test_raises_on_invalid_model_name():
    X = _mcar_matrix(n=30, d=3, seed=0)
    with pytest.raises(ValueError, match="model must be"):
        propensity_mcar_test(X, model="xgboost")


def test_raises_on_invalid_null_name():
    X = _mcar_matrix(n=30, d=3, seed=0)
    with pytest.raises(ValueError, match="null must be"):
        propensity_mcar_test(X, null="bayesian")


def test_raises_on_invalid_alpha():
    X = _mcar_matrix(n=30, d=3, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        propensity_mcar_test(X, alpha=1.5)
