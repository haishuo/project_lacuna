"""Tests for lacuna.data.composition_target — the Stage-B simplex × rate prior (ADR-0007)."""

import numpy as np
import pytest

from lacuna.core.rng import RNGState
from lacuna.data.composition_target import CompositionTarget, sample_composition_target


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_draw_is_valid_composition_and_rate():
    t = sample_composition_target(RNGState(seed=0))
    assert 0.0 <= t.f_mcar <= 1.0 and 0.0 <= t.f_mar <= 1.0 and 0.0 <= t.f_mnar <= 1.0
    assert abs(sum(t.as_fractions()) - 1.0) < 1e-9
    assert 0.05 <= t.miss_rate <= 0.6


def test_as_fractions_order_matches_class_ids():
    """as_fractions() must be indexable by MCAR=0 / MAR=1 / MNAR=2."""
    from lacuna.core.types import MCAR, MAR, MNAR
    t = CompositionTarget(f_mcar=0.2, f_mar=0.3, f_mnar=0.5, miss_rate=0.25)
    fr = t.as_fractions()
    assert fr[MCAR] == 0.2 and fr[MAR] == 0.3 and fr[MNAR] == 0.5


def test_deterministic_under_same_seed():
    a = sample_composition_target(RNGState(seed=7))
    b = sample_composition_target(RNGState(seed=7))
    assert a == b


def test_different_seeds_give_different_draws():
    a = sample_composition_target(RNGState(seed=1))
    b = sample_composition_target(RNGState(seed=2))
    assert a != b


def test_uniform_prior_is_broad_and_covers_simplex():
    """concentration=1 should spread mass across the whole simplex (no corner starvation) —
    the broad / prior-agnostic training prior of ADR-0007 commitment 4."""
    rng = RNGState(seed=3)
    draws = np.array([sample_composition_target(rng.spawn()).as_fractions() for _ in range(3000)])
    # Each marginal mean ~1/3 under a symmetric Dirichlet(1,1,1); spread should be wide.
    assert np.allclose(draws.mean(axis=0), 1 / 3, atol=0.05)
    assert (draws.std(axis=0) > 0.18).all()           # genuinely broad, not pinned to the centre
    assert (draws.max(axis=0) > 0.95).all()            # near-pure datasets are reachable...
    assert (draws.min(axis=0) < 0.05).all()            # ...as are near-absent ones


def test_low_concentration_pushes_to_corners():
    rng = RNGState(seed=4)
    lo = np.array([sample_composition_target(rng.spawn(), concentration=0.2).as_fractions()
                   for _ in range(2000)])
    hi = np.array([sample_composition_target(rng.spawn(), concentration=5.0).as_fractions()
                   for _ in range(2000)])
    # Lower concentration => more dispersed (closer to corners) => higher marginal variance.
    assert lo.std(axis=0).mean() > hi.std(axis=0).mean()


def test_miss_rate_range_respected():
    rng = RNGState(seed=5)
    rates = [sample_composition_target(rng.spawn(), miss_rate_range=(0.2, 0.3)).miss_rate
             for _ in range(500)]
    assert all(0.2 <= r <= 0.3 for r in rates)


# ---------------------------------------------------------------------------
# Edge / failure cases
# ---------------------------------------------------------------------------

def test_composition_target_validates_fraction_sum():
    with pytest.raises(ValueError, match="sum to 1"):
        CompositionTarget(f_mcar=0.5, f_mar=0.5, f_mnar=0.5, miss_rate=0.25)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_composition_target_validates_fraction_range(bad):
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        CompositionTarget(f_mcar=bad, f_mar=0.0, f_mnar=1.0 - bad, miss_rate=0.25)


@pytest.mark.parametrize("bad_rate", [0.0, 1.0, -0.2, 1.5])
def test_composition_target_validates_miss_rate(bad_rate):
    with pytest.raises(ValueError, match="miss_rate"):
        CompositionTarget(f_mcar=0.34, f_mar=0.33, f_mnar=0.33, miss_rate=bad_rate)


@pytest.mark.parametrize("bad_conc", [0.0, -1.0])
def test_bad_concentration_raises(bad_conc):
    with pytest.raises(ValueError, match="concentration"):
        sample_composition_target(RNGState(seed=0), concentration=bad_conc)


@pytest.mark.parametrize("bad_range", [(0.5, 0.4), (0.0, 0.5), (0.5, 1.0), (-0.1, 0.5)])
def test_bad_miss_rate_range_raises(bad_range):
    with pytest.raises(ValueError, match="miss_rate_range"):
        sample_composition_target(RNGState(seed=0), miss_rate_range=bad_range)
