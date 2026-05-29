"""Tests for lacuna.data.mnar_column_pool — diverse per-column MNAR generator sampling."""

from collections import Counter

import pytest

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.data.mnar_column_pool import (
    MNAR_SUBTYPES,
    sample_mnar_column_generator,
    _builders,
)


@pytest.fixture
def Z():
    """A reproducible z-scored standard-normal matrix (the composer's predictor view)."""
    return _zscore_columns(RNGState(seed=11).randn(1500, 5))


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_returns_subtype_name_and_mnar_generator():
    name, gen = sample_mnar_column_generator(2, RNGState(seed=0))
    assert name in MNAR_SUBTYPES
    assert gen.class_id == MNAR


def test_targets_only_the_named_column(Z):
    _, gen = sample_mnar_column_generator(3, RNGState(seed=0), target_miss_rate=0.25)
    R = gen.apply_to(Z, RNGState(seed=5))
    missing_cols = (~R.all(dim=0)).nonzero().flatten().tolist()
    assert missing_cols == [3]


def test_all_subtypes_reachable_over_many_draws():
    """Over many draws the pool should exercise the full subtype set (no dead branch)."""
    rng = RNGState(seed=1)
    seen = Counter(sample_mnar_column_generator(1, rng.spawn())[0] for _ in range(600))
    # All 8 subtypes should appear at least once.
    assert set(seen) == set(MNAR_SUBTYPES), f"missing: {set(MNAR_SUBTYPES) - set(seen)}"


def test_builders_cover_declared_subtypes():
    names = [n for n, _ in _builders(0, 0.25, 1.5)]
    assert set(names) == set(MNAR_SUBTYPES)
    assert len(names) == len(MNAR_SUBTYPES)


def test_every_subtype_actually_censors_target(Z):
    """Each builder, when applied, removes some (but not all) of the target column."""
    for name, gen in _builders(target_col_idx=2, target_miss_rate=0.25, strength=1.5):
        R = gen.apply_to(Z, RNGState(seed=3))
        obs_frac = R[:, 2].float().mean().item()
        assert 0.0 < obs_frac < 1.0, f"{name}: obs frac {obs_frac}"
        # no other column touched
        other = [c for c in (~R.all(dim=0)).nonzero().flatten().tolist() if c != 2]
        assert other == [], f"{name} also censored {other}"


def test_miss_rate_roughly_matches_target(Z):
    """Confound control: every subtype lands near target_miss_rate (within a tolerance)."""
    for name, gen in _builders(target_col_idx=2, target_miss_rate=0.25, strength=1.5):
        R = gen.apply_to(Z, RNGState(seed=4))
        rate = 1.0 - R[:, 2].float().mean().item()
        assert 0.15 <= rate <= 0.40, f"{name}: realized rate {rate:.3f} far from 0.25"


def test_deterministic_under_same_seed(Z):
    n1, g1 = sample_mnar_column_generator(2, RNGState(seed=42), target_miss_rate=0.25)
    n2, g2 = sample_mnar_column_generator(2, RNGState(seed=42), target_miss_rate=0.25)
    assert n1 == n2
    import torch
    assert torch.equal(g1.apply_to(Z, RNGState(seed=9)), g2.apply_to(Z, RNGState(seed=9)))


def test_negative_index_after_wrap_is_callers_job():
    """The pool requires a concrete non-negative column index (composer passes one)."""
    with pytest.raises(ValueError, match=">= 0"):
        sample_mnar_column_generator(-1, RNGState(seed=0))


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_non_int_target_raises():
    with pytest.raises(ValueError, match="must be an int"):
        sample_mnar_column_generator(2.0, RNGState(seed=0))


def test_bool_target_raises():
    with pytest.raises(ValueError, match="must be an int"):
        sample_mnar_column_generator(True, RNGState(seed=0))


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.1, 1.5])
def test_bad_miss_rate_raises(bad_rate):
    with pytest.raises(ValueError, match="target_miss_rate"):
        sample_mnar_column_generator(2, RNGState(seed=0), target_miss_rate=bad_rate)
