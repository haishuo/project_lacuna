"""Tests for lacuna.analysis.realism_gate.fidelity."""

import numpy as np
import pytest

from lacuna.analysis.realism_gate.fidelity import run_fidelity


def test_identical_masks_zero_gaps():
    rng = np.random.default_rng(0)
    M = (rng.random((1000, 8)) < 0.15).astype(np.uint8)
    res = run_fidelity(M, M.copy(), seed=1)
    assert res.attr_max_gap == pytest.approx(0.0)
    assert res.attr_mean_gap == pytest.approx(0.0)
    assert res.bivar_max_gap == pytest.approx(0.0, abs=1e-9)
    assert res.runlen_tv == pytest.approx(0.0, abs=1e-9)
    assert res.rowcount_tv == pytest.approx(0.0, abs=1e-9)
    assert res.overall_rate_real == pytest.approx(res.overall_rate_gen)


def test_rate_shift_increases_attr_gap():
    rng = np.random.default_rng(0)
    M_real = (rng.random((1000, 8)) < 0.05).astype(np.uint8)
    M_gen = (rng.random((1000, 8)) < 0.40).astype(np.uint8)
    res = run_fidelity(M_real, M_gen, seed=1)
    assert res.attr_max_gap > 0.2
    assert res.overall_rate_gen > res.overall_rate_real


def test_authenticity_fields_present():
    rng = np.random.default_rng(0)
    M_real = (rng.random((1000, 8)) < 0.1).astype(np.uint8)
    M_gen = (rng.random((1000, 8)) < 0.1).astype(np.uint8)
    res = run_fidelity(M_real, M_gen, seed=1)
    assert 0.0 <= res.auth_exact_dup_frac <= 1.0
    assert res.auth_median_nn >= 0.0


def test_column_mismatch_raises():
    with pytest.raises(ValueError):
        run_fidelity(np.zeros((10, 4), dtype=np.uint8),
                     np.zeros((10, 5), dtype=np.uint8), seed=1)


def test_determinism():
    rng = np.random.default_rng(0)
    M_real = (rng.random((500, 6)) < 0.1).astype(np.uint8)
    M_gen = (rng.random((500, 6)) < 0.2).astype(np.uint8)
    a = run_fidelity(M_real, M_gen, seed=5)
    b = run_fidelity(M_real, M_gen, seed=5)
    assert a == b
