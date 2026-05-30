"""Tests for lacuna.data.real_mask_sampler — analyst-realistic block sampling (ADR-0007 Stage A)."""

import numpy as np
import pytest

from lacuna.data.real_mask_sampler import sample_mask_blocks


def _table(n=3000, d=50, seed=0):
    """A synthetic 'survey' table: standard-normal values with per-column missing rates ~U(0, 0.5)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    rates = rng.uniform(0.0, 0.5, size=d)
    x[rng.random((n, d)) < rates] = np.nan
    return x


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_blocks_respect_shape_bounds():
    blocks = sample_mask_blocks(_table(), np.random.default_rng(1), n_blocks=20,
                                min_cols=4, max_cols=20, min_rows=200, max_rows=1500)
    assert len(blocks) > 0
    for x, r in blocks:
        assert 4 <= x.shape[1] <= 20
        assert x.shape[0] >= 200


def test_blocks_in_missingness_band_with_both_present():
    blocks = sample_mask_blocks(_table(), np.random.default_rng(2), n_blocks=25,
                                miss_lo=0.02, miss_hi=0.5)
    assert len(blocks) > 0
    for x, r in blocks:
        m = np.isnan(x).mean()
        assert 0.02 <= m <= 0.5
        assert r.any() and (~r).any()           # both observed and missing present


def test_mask_matches_nan_pattern():
    blocks = sample_mask_blocks(_table(), np.random.default_rng(3), n_blocks=5)
    for x, r in blocks:
        assert np.array_equal(r, ~np.isnan(x))


def test_deterministic_under_same_seed():
    a = sample_mask_blocks(_table(seed=5), np.random.default_rng(9), n_blocks=8)
    b = sample_mask_blocks(_table(seed=5), np.random.default_rng(9), n_blocks=8)
    assert len(a) == len(b)
    for (xa, ra), (xb, rb) in zip(a, b):
        assert np.array_equal(xa, xb, equal_nan=True)
        assert np.array_equal(ra, rb)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_complete_table_yields_nothing():
    """A fully-observed table has no missingness to sample → no blocks pass the band."""
    complete = np.random.default_rng(4).normal(size=(2000, 40))
    assert sample_mask_blocks(complete, np.random.default_rng(0), n_blocks=10) == []


def test_table_too_small_returns_empty():
    small = _table(n=50, d=3)                    # d < min_cols and n < min_rows
    assert sample_mask_blocks(small, np.random.default_rng(0), n_blocks=5) == []


def test_high_band_excludes_low_missing_blocks():
    """Requiring miss in a high band on a low-missing table yields few/zero blocks (not a crash)."""
    low = np.random.default_rng(6).normal(size=(3000, 50))
    low[np.random.default_rng(6).random((3000, 50)) < 0.02] = np.nan  # ~2% missing
    blocks = sample_mask_blocks(low, np.random.default_rng(0), n_blocks=10, miss_lo=0.4, miss_hi=0.9)
    assert all(0.4 <= np.isnan(x).mean() <= 0.9 for x, _ in blocks)  # whatever passes is in-band


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_non_2d_raises():
    with pytest.raises(ValueError, match="2-D"):
        sample_mask_blocks(np.zeros(10), np.random.default_rng(0), n_blocks=1)


def test_bad_n_blocks_raises():
    with pytest.raises(ValueError, match="n_blocks"):
        sample_mask_blocks(_table(), np.random.default_rng(0), n_blocks=0)


def test_incoherent_bounds_raise():
    with pytest.raises(ValueError, match="min_cols"):
        sample_mask_blocks(_table(), np.random.default_rng(0), n_blocks=1, min_cols=20, max_cols=4)


def test_bad_missingness_band_raises():
    with pytest.raises(ValueError, match="miss_lo"):
        sample_mask_blocks(_table(), np.random.default_rng(0), n_blocks=1, miss_lo=0.7, miss_hi=0.3)
