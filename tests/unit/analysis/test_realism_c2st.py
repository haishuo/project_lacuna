"""Tests for lacuna.analysis.realism_gate.c2st."""

import numpy as np
import pytest

from lacuna.analysis.realism_gate.c2st import run_c2st


def _blocks(n, n_blocks, seed):
    return np.random.default_rng(seed).integers(0, n_blocks, n)


def test_identical_distributions_near_chance():
    # real and gen drawn from the SAME Bernoulli(0.1) -> indistinguishable.
    rng = np.random.default_rng(0)
    d = 10
    M_real = (rng.random((3000, d)) < 0.1).astype(np.uint8)
    M_gen = (rng.random((3000, d)) < 0.1).astype(np.uint8)
    cols = tuple(f"c{j}" for j in range(d))
    res = run_c2st(M_real, M_gen, _blocks(3000, 4, 1), _blocks(3000, 4, 2), cols, seed=7)
    assert res.auc < 0.6           # near chance
    assert res.p_value > 0.01      # cannot reject equality


def test_shifted_distributions_separable():
    rng = np.random.default_rng(0)
    d = 10
    M_real = (rng.random((3000, d)) < 0.05).astype(np.uint8)
    M_gen = (rng.random((3000, d)) < 0.40).astype(np.uint8)  # much higher rate
    cols = tuple(f"c{j}" for j in range(d))
    res = run_c2st(M_real, M_gen, _blocks(3000, 4, 1), _blocks(3000, 4, 2), cols, seed=7)
    assert res.auc > 0.8
    assert res.p_value < 1e-3
    # localisation: the rate gap should be large and positive
    assert res.top_rate_gaps[0][1] > 0.2


def test_determinism_same_seed():
    rng = np.random.default_rng(0)
    d = 8
    M_real = (rng.random((1500, d)) < 0.1).astype(np.uint8)
    M_gen = (rng.random((1500, d)) < 0.2).astype(np.uint8)
    cols = tuple(f"c{j}" for j in range(d))
    a = run_c2st(M_real, M_gen, _blocks(1500, 3, 1), _blocks(1500, 3, 2), cols, seed=11)
    b = run_c2st(M_real, M_gen, _blocks(1500, 3, 1), _blocks(1500, 3, 2), cols, seed=11)
    assert a.auc == b.auc and a.p_value == b.p_value


def test_block_aware_flag_single_block_falls_back():
    rng = np.random.default_rng(0)
    d = 6
    M_real = (rng.random((800, d)) < 0.1).astype(np.uint8)
    M_gen = (rng.random((800, d)) < 0.1).astype(np.uint8)
    cols = tuple(f"c{j}" for j in range(d))
    res = run_c2st(M_real, M_gen, np.zeros(800, int), np.zeros(800, int), cols, seed=3)
    assert res.block_aware is False   # only one block -> stratified fallback


def test_column_mismatch_raises():
    M_real = np.zeros((100, 5), dtype=np.uint8)
    M_gen = np.zeros((100, 6), dtype=np.uint8)
    with pytest.raises(ValueError):
        run_c2st(M_real, M_gen, np.zeros(100, int), np.zeros(100, int),
                 tuple(f"c{j}" for j in range(5)), seed=1)
