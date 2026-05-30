"""
Tests for lacuna.data.missingness_footprint — the observable footprint extractor (ADR-0007 Stage A).

The footprint must (a) honour its contract (exact key set, finite floats, torch==numpy,
deterministic), (b) actually *respond* to the structures it is meant to detect (block co-missingness,
monotone dropout, MAR coupling, MNAR observed-distortion), and (c) fail loud on malformed input.
"""

import numpy as np
import pytest
import torch

from lacuna.data.missingness_footprint import missingness_footprint, FOOTPRINT_FEATURES


def _fp(x, r, **kw):
    return missingness_footprint(x, r, **kw)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_output_key_set_and_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    r = rng.random((200, 6)) > 0.25
    out = _fp(x, r)
    assert list(out.keys()) == list(FOOTPRINT_FEATURES)      # exact order
    assert len(FOOTPRINT_FEATURES) == 20
    assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())


def test_torch_and_numpy_agree():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(150, 5))
    r = rng.random((150, 5)) > 0.3
    a = _fp(x, r)
    b = _fp(torch.from_numpy(x), torch.from_numpy(r))
    assert a == b


def test_deterministic():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(120, 4)); r = rng.random((120, 4)) > 0.2
    assert _fp(x, r) == _fp(x, r)


def test_nan_at_missing_equals_zero_at_missing():
    """Real CSVs encode missing as NaN; the synthetic path zeroes them. Only x[r] is read, so the
    footprint must be identical either way."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 5)); r = rng.random((200, 5)) > 0.3
    x_nan = x.copy(); x_nan[~r] = np.nan
    x_zero = x * r
    assert _fp(x_nan, r) == _fp(x_zero, r)


# ---------------------------------------------------------------------------
# Normal cases — known structures produce the expected footprint
# ---------------------------------------------------------------------------

def test_complete_dataset():
    x = np.random.default_rng(4).normal(size=(100, 5))
    r = np.ones((100, 5), dtype=bool)
    out = _fp(x, r)
    assert out["col_rate_mean"] == 0.0
    assert out["frac_cols_complete"] == 1.0
    assert out["frac_rows_complete"] == 1.0
    assert out["top1_pattern_frac"] == 1.0            # one (empty) pattern
    assert out["monotone_row_frac"] == 1.0
    assert out["miss_corr_mean_abs"] == 0.0
    assert out["mar_coupling_mean"] == 0.0


def test_block_comissingness_beats_independent():
    """Columns that go missing together (a block/module) raise the co-missingness correlation."""
    rng = np.random.default_rng(5)
    n, d = 400, 6
    x = rng.normal(size=(n, d))
    indep = rng.random((n, d)) > 0.25                  # each cell independent
    block = np.ones((n, d), dtype=bool)
    drop = rng.random(n) < 0.25                         # a shared "drop this module" event
    block[drop, 0] = block[drop, 1] = block[drop, 2] = False   # cols 0-2 missing together
    block[rng.random((n, d)) < 0.05] = False           # a little independent noise elsewhere
    assert _fp(x, block)["miss_corr_mean_abs"] > _fp(x, indep)["miss_corr_mean_abs"] + 0.1


def test_monotone_dropout():
    """A staircase mask (each row observes a prefix, misses a suffix) is fully monotone; a random
    mask is not."""
    rng = np.random.default_rng(6)
    n, d = 300, 6
    x = rng.normal(size=(n, d))
    # staircase: row i is observed up to a random cutoff, missing after — and later columns miss more
    cutoff = rng.integers(1, d + 1, size=n)
    stair = np.arange(d)[None, :] < cutoff[:, None]    # True (observed) before cutoff
    rand = rng.random((n, d)) > 0.3
    assert _fp(x, stair)["monotone_row_frac"] == 1.0
    assert _fp(x, rand)["monotone_row_frac"] < 1.0


def test_mar_coupling_detects_predictor():
    """A column whose missingness tracks another column's observed values scores high on the MAR
    axis; a randomly-missing column scores low."""
    rng = np.random.default_rng(7)
    n, d = 600, 5
    x = rng.normal(size=(n, d))
    mar = np.ones((n, d), dtype=bool)
    mar[:, 4] = x[:, 0] <= 0.0                          # col 4 missing iff col 0 (observed) is high
    mcar = np.ones((n, d), dtype=bool)
    mcar[:, 4] = rng.random(n) > 0.5                    # col 4 missing at random
    hi = _fp(x, mar)["mar_coupling_max"]
    lo = _fp(x, mcar)["mar_coupling_max"]
    assert hi > 0.4 and hi > lo + 0.2


def test_mnar_observed_distortion():
    """A self-censored column (the top values are the ones that go missing) leaves a skewed observed
    marginal; the MCAR version does not."""
    rng = np.random.default_rng(8)
    n = 800
    x = rng.normal(size=(n, 2))
    sc = np.ones((n, 2), dtype=bool)
    sc[:, 1] = x[:, 1] <= np.quantile(x[:, 1], 0.7)    # drop the top 30% of col 1 (truncation)
    mcar = np.ones((n, 2), dtype=bool)
    mcar[:, 1] = rng.random(n) > 0.3                    # same rate, at random
    assert _fp(x, sc)["obs_abs_skew_mean"] > _fp(x, mcar)["obs_abs_skew_mean"] + 0.1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_column_does_not_crash():
    x = np.random.default_rng(9).normal(size=(50, 1))
    r = np.random.default_rng(9).random((50, 1)) > 0.3
    out = _fp(x, r)
    assert list(out.keys()) == list(FOOTPRINT_FEATURES)
    assert out["miss_corr_mean_abs"] == 0.0            # no pairs
    assert out["mar_coupling_mean"] == 0.0             # nothing to couple to


def test_single_row_does_not_crash():
    out = _fp(np.zeros((1, 5)), np.array([[True, False, True, True, False]]))
    assert all(np.isfinite(v) for v in out.values())


def test_fully_missing_column_handled():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(100, 4)); r = rng.random((100, 4)) > 0.3
    r[:, 2] = False                                    # column 2 never observed
    out = _fp(x, r)
    assert all(np.isfinite(v) for v in out.values())
    assert out["frac_cols_complete"] < 1.0


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        _fp(np.zeros((10, 3)), np.ones((10, 4), dtype=bool))


def test_non_2d_raises():
    with pytest.raises(ValueError, match="2-D"):
        _fp(np.zeros((10,)), np.ones((10,), dtype=bool))


def test_empty_raises():
    with pytest.raises(ValueError, match="n >= 1 and d >= 1"):
        _fp(np.zeros((0, 3)), np.ones((0, 3), dtype=bool))
