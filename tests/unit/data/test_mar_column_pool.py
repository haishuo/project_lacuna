"""Tests for lacuna.data.mar_column_pool — diverse per-column MAR generator sampling (ADR-0006)."""

from collections import Counter

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.data.mar_column_pool import (
    MAR_SUBTYPES,
    sample_mar_column_generator,
    _builders,
)


@pytest.fixture
def Z():
    """A reproducible z-scored standard-normal matrix (the composer's predictor view)."""
    return _zscore_columns(RNGState(seed=11).randn(1500, 5))


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_returns_subtype_name_and_mar_generator():
    name, gen = sample_mar_column_generator(2, 0, RNGState(seed=0))
    assert name in MAR_SUBTYPES
    assert gen.class_id == MAR


def test_targets_only_the_named_column(Z):
    _, gen = sample_mar_column_generator(3, 0, RNGState(seed=0), target_miss_rate=0.25)
    R = gen.apply_to(Z, RNGState(seed=5))
    missing_cols = (~R.all(dim=0)).nonzero().flatten().tolist()
    assert missing_cols == [3]


def test_all_subtypes_reachable_over_many_draws():
    """Over many draws the pool exercises the full subtype set (no dead branch)."""
    rng = RNGState(seed=1)
    seen = Counter(sample_mar_column_generator(1, 0, rng.spawn())[0] for _ in range(1500))
    assert set(seen) == set(MAR_SUBTYPES), f"missing: {set(MAR_SUBTYPES) - set(seen)}"


def test_builders_cover_declared_subtypes():
    names = [n for n, _ in _builders(0, 1, 0.25, 1.5)]
    assert set(names) == set(MAR_SUBTYPES)
    assert len(names) == len(MAR_SUBTYPES)


def test_pool_is_diverse():
    """Diverse MAR pool (linear / probit / polynomial / threshold / step / predictor-type /
    realistic), not the single MARLogistic monoculture of Stage 0-3."""
    assert len(MAR_SUBTYPES) >= 6
    for required in ("mar_logistic", "mar_probit", "mar_threshold", "mar_realistic"):
        assert required in MAR_SUBTYPES


def test_every_subtype_actually_censors_target(Z):
    """Each builder removes some (but not all) of the target column, none touches others."""
    for name, gen in _builders(target_col_idx=2, predictor_col_idx=0, target_miss_rate=0.25, strength=1.5):
        R = gen.apply_to(Z, RNGState(seed=3))
        obs_frac = R[:, 2].float().mean().item()
        assert 0.0 < obs_frac < 1.0, f"{name}: obs frac {obs_frac}"
        other = [c for c in (~R.all(dim=0)).nonzero().flatten().tolist() if c != 2]
        assert other == [], f"{name} also censored {other}"


def test_miss_rate_in_tolerance_across_seeds():
    """Confound control: every subtype lands within [0.15, 0.40] across data seeds."""
    for sd in range(5):
        Zs = _zscore_columns(RNGState(seed=300 + sd).randn(1500, 5))
        for name, gen in _builders(target_col_idx=2, predictor_col_idx=0,
                                   target_miss_rate=0.25, strength=1.5):
            R = gen.apply_to(Zs, RNGState(seed=13 + sd))
            rate = 1.0 - R[:, 2].float().mean().item()
            assert 0.15 <= rate <= 0.40, f"{name} seed {sd}: realized rate {rate:.3f} out of band"


def test_missingness_depends_on_predictor_not_target(Z):
    """MAR signature: the targeted column's missingness tracks the PREDICTOR column's values,
    not its own (that would be MNAR). Check that rows missing in the target differ in predictor
    mean from observed rows."""
    _, gen = sample_mar_column_generator(2, 0, RNGState(seed=2), target_miss_rate=0.25, strength=2.0)
    R = gen.apply_to(Z, RNGState(seed=5))
    miss = ~R[:, 2]
    pred_missing_mean = Z[miss, 0].mean().item()
    pred_obs_mean = Z[~miss, 0].mean().item()
    assert abs(pred_missing_mean - pred_obs_mean) > 0.2, "predictor should separate missing/observed"


def test_deterministic_under_same_seed(Z):
    n1, g1 = sample_mar_column_generator(2, 0, RNGState(seed=42), target_miss_rate=0.25)
    n2, g2 = sample_mar_column_generator(2, 0, RNGState(seed=42), target_miss_rate=0.25)
    assert n1 == n2
    assert torch.equal(g1.apply_to(Z, RNGState(seed=9)), g2.apply_to(Z, RNGState(seed=9)))


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_negative_target_raises():
    with pytest.raises(ValueError, match="target_col_idx must be >= 0"):
        sample_mar_column_generator(-1, 0, RNGState(seed=0))


def test_negative_predictor_raises():
    with pytest.raises(ValueError, match="predictor_col_idx must be >= 0"):
        sample_mar_column_generator(2, -1, RNGState(seed=0))


def test_equal_target_predictor_raises():
    with pytest.raises(ValueError, match="must differ from target"):
        sample_mar_column_generator(2, 2, RNGState(seed=0))


def test_non_int_target_raises():
    with pytest.raises(ValueError, match="must be an int"):
        sample_mar_column_generator(2.0, 0, RNGState(seed=0))


def test_bool_index_raises():
    with pytest.raises(ValueError, match="must be an int"):
        sample_mar_column_generator(True, 0, RNGState(seed=0))


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.1, 1.5])
def test_bad_miss_rate_raises(bad_rate):
    with pytest.raises(ValueError, match="target_miss_rate"):
        sample_mar_column_generator(2, 0, RNGState(seed=0), target_miss_rate=bad_rate)
