"""Tests for the T2 per-column mechanism + mixture generator."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.survey.skip_logic_generator import (FAMILIES, RATE_LO, RATE_HI,
                                                generate_table)


def _X(n=384, d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _gen(X, seed, retries=10):
    """Mirror the runner's retry-on-infeasible-draw contract (T1 semantics)."""
    base = RNGState(seed=seed)
    for _ in range(retries):
        try:
            return generate_table(X, base.spawn())
        except ValueError as err:
            if "d>=3" in str(err) or "n>=30" in str(err) or "finite" in str(err):
                raise                                       # structural, not a draw — re-raise
    raise AssertionError("10 infeasible draws")


# ---- normal cases ----
def test_returns_mask_and_flags():
    R, flagged = _gen(_X(), 1)
    assert R.shape == (384, 5) and R.dtype == torch.bool
    assert 1 <= len(flagged) <= 2
    for f in flagged:
        assert f.family in FAMILIES
        assert f.dbin in (0, 1)


def test_flagged_columns_in_rate_band_unflagged_complete():
    R, flagged = _gen(_X(), 3)
    flagged_cols = {f.col for f in flagged}
    for f in flagged:
        assert RATE_LO <= f.rate <= RATE_HI
    for j in range(R.shape[1]):
        if j not in flagged_cols:
            assert bool(R[:, j].all())                     # unflagged ⇒ fully observed


def test_null_family_is_always_dbin0():
    seen = 0
    for s in range(200):
        _, flagged = _gen(_X(seed=s), s)
        for f in flagged:
            if f.family == "null":
                seen += 1
                assert f.dbin == 0
    assert seen > 0


def test_all_families_appear_over_many_draws():
    fams = set()
    for s in range(200):
        _, flagged = _gen(_X(seed=s), s + 7)
        fams.update(f.family for f in flagged)
    assert fams == set(FAMILIES)


def test_mixture_sizes_one_and_two_both_occur():
    sizes = set()
    for s in range(100):
        _, flagged = _gen(_X(seed=s), s + 11)
        sizes.add(len(flagged))
    assert sizes == {1, 2}


def test_deterministic_given_seed():
    R1, f1 = generate_table(_X(seed=4), RNGState(seed=99))
    R2, f2 = generate_table(_X(seed=4), RNGState(seed=99))
    assert torch.equal(R1, R2)
    assert [(f.col, f.family, f.dbin) for f in f1] == [(f.col, f.family, f.dbin) for f in f2]


def test_active_self_driven_separates_observed_values():
    """An active self_driven (own_value) column should censor by its own value (mean shift)."""
    shifts = []
    for s in range(300):
        X = _X(seed=s)
        R, flagged = _gen(X, s + 5)
        for f in flagged:
            if f.family == "self_driven" and f.dbin == 1 and f.detail == "own_value":
                col = X[:, f.col]
                shifts.append(float(col[R[:, f.col]].mean() - col[~R[:, f.col]].mean()))
    # observed mean should sit below the missing mean (high values censored) on average
    assert len(shifts) > 5 and np.mean(shifts) < 0


# ---- failure cases (Rule 1) ----
def test_too_few_columns_raises():
    with pytest.raises(ValueError, match="d>=3"):
        generate_table(_X(d=2), RNGState(seed=1))


def test_too_few_rows_raises():
    with pytest.raises(ValueError, match="n>=30"):
        generate_table(_X(n=10), RNGState(seed=1))


def test_non_finite_raises():
    X = _X()
    X[0, 0] = float("nan")
    with pytest.raises(ValueError, match="complete/finite"):
        generate_table(X, RNGState(seed=1))
