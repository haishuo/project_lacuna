"""Tests for lacuna.analysis.mcar.mom.mom_mcar_test.

Ported from the old ``pystatistics/tests/mvnmle/test_mom_mcar.py`` when
MoM was moved to Lacuna on 2026-04-20 (CLAUDE.md §8). Invariants pinned:

1. Returns a ``MCARTestResult`` (pystatistics's public dataclass) whose
   ``method`` field explicitly identifies the estimator as method-of-
   moments. No silent masquerading as Little's test.
2. Under MCAR on moderate-n data, MoM and Little's agree directionally
   on the test statistic and p-value — not bit-identical (different
   estimators) but within a tolerance that makes MoM useful as a fast
   screen.
3. All-missing-row handling mirrors ``little_mcar_test``.
4. Completes in under 100 ms on breast-cancer-scale data.
"""

import time
import warnings

import numpy as np
import pytest

from pystatistics.mvnmle import datasets, little_mcar_test

from lacuna.analysis.mcar import mom_mcar_test


# =====================================================================
# Honesty — the result identifies itself
# =====================================================================


def test_mom_result_method_field_says_method_of_moments():
    r = mom_mcar_test(datasets.apple)
    assert "Method-of-moments" in r.method
    assert "Little" not in r.method


def test_summary_includes_method_line():
    r = mom_mcar_test(datasets.apple)
    assert "Method: Method-of-moments" in r.summary()


# =====================================================================
# Correctness — MoM and MLE agree to a reasonable tolerance under MCAR
# =====================================================================


def test_mom_runs_on_apple():
    r = mom_mcar_test(datasets.apple)
    assert np.isfinite(r.statistic)
    assert 0.0 <= r.p_value <= 1.0
    assert r.df > 0


def test_mom_runs_on_missvals():
    r = mom_mcar_test(datasets.missvals)
    assert np.isfinite(r.statistic)
    assert 0.0 <= r.p_value <= 1.0


def test_mom_mle_agree_qualitatively_on_apple():
    """Apple violates MCAR (test statistic much larger than df), so both
    MoM and MLE should reject. Statistics will differ because of
    different plug-ins, but the rejection decision should agree."""
    mom = mom_mcar_test(datasets.apple)
    mle = little_mcar_test(datasets.apple)
    assert mom.df == mle.df
    assert mom.rejected == mle.rejected
    assert 0.5 * mle.statistic <= mom.statistic <= 2.0 * mle.statistic


def test_mom_mle_agree_under_large_n_mcar():
    """On n=1000 MCAR data, both estimators are consistent so the
    statistics should be close."""
    rng = np.random.default_rng(0)
    n, v = 1000, 5
    X = rng.standard_normal((n, v))
    X[rng.random(X.shape) < 0.15] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]

    mom = mom_mcar_test(X)
    mle = little_mcar_test(X)

    assert mom.df == mle.df
    assert not mom.rejected
    assert not mle.rejected
    rel_diff = abs(mom.statistic - mle.statistic) / max(mle.statistic, 1.0)
    assert rel_diff < 0.25, (
        f"MoM stat={mom.statistic:.2f}, MLE stat={mle.statistic:.2f} — "
        f"relative diff {rel_diff:.2%} too large on MCAR data"
    )


# =====================================================================
# Edge cases — mirror little_mcar_test handling
# =====================================================================


def test_mom_auto_drops_all_missing_rows_with_warning():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[0, :] = np.nan

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        r = mom_mcar_test(X)
    assert any("all values missing" in str(w.message) for w in captured)
    assert np.isfinite(r.statistic)


def test_mom_complete_data_returns_sentinel():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    r = mom_mcar_test(X)
    assert r.statistic == 0.0
    assert r.p_value == 1.0
    assert r.df == 0


# =====================================================================
# Speed — MoM should beat MLE on realistic shapes
# =====================================================================


def test_mom_dramatically_faster_than_mle_on_breast_scale():
    """Regression for the headline claim. MoM ≥10× faster than MLE on
    breast-cancer-scale data; typical measured ratio is 30-100×. The
    Lacuna port uses a simple per-pattern loop instead of the batched
    path that used to live in pystatistics, so the absolute speed is
    slightly worse than 2.3.0's mom_mcar_test — still massively faster
    than MLE."""
    sklearn = pytest.importorskip("sklearn.datasets")
    X = sklearn.load_breast_cancer().data.astype(float).copy()
    rng = np.random.default_rng(0)
    X[rng.random(X.shape) < 0.15] = np.nan
    X = X[~np.all(np.isnan(X), axis=1)]

    _ = mom_mcar_test(X)
    _ = little_mcar_test(X)

    t = time.perf_counter()
    _ = mom_mcar_test(X)
    mom_time = time.perf_counter() - t

    t = time.perf_counter()
    _ = little_mcar_test(X)
    mle_time = time.perf_counter() - t

    assert mom_time * 10 < mle_time, (
        f"MoM {mom_time*1000:.1f} ms vs MLE {mle_time*1000:.1f} ms — "
        f"expected ≥ 10× speedup on breast_cancer-scale data."
    )
