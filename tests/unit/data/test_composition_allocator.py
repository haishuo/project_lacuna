"""Tests for lacuna.data.composition_allocator — the Stage-B planning brain (ADR-0007).

The headline invariant here is the PLANNING half of the shared-denominator discipline: the plan's
EXPECTED by-cell composition must track the drawn target. (The realised composition is checked end-
to-end in test_composition_sampler.py.)
"""

import numpy as np
import pytest

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.composition_target import CompositionTarget, sample_composition_target
from lacuna.data.composition_allocator import (
    plan_allocation, _largest_remainder, _MIN_BLOCK_WIDTH,
)


def _l1(a, b):
    return float(np.abs(np.array(a) - np.array(b)).sum())


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_columns_partition_exactly_once():
    """Observed columns + every unit's columns tile [0, d) with no overlap (clean tagging needs
    each missing cell owned by exactly one mechanism)."""
    t = sample_composition_target(RNGState(seed=1))
    plan = plan_allocation(800, 20, t, RNGState(seed=2))
    seen = list(plan.observed_cols)
    for u in plan.units:
        seen.extend(u.cols)
    assert sorted(seen) == list(range(20)), "columns not a clean partition"
    assert len(seen) == len(set(seen)), "a column is owned by two mechanisms"


def test_expected_composition_tracks_target_across_seeds():
    """The plan's expected by-cell composition matches the drawn target (the column-partition +
    common-rate control). Reported as a distribution, never a single seed."""
    l1s = []
    rng = RNGState(seed=10)
    for _ in range(40):
        t = sample_composition_target(rng.spawn())
        plan = plan_allocation(1000, 20, t, rng.spawn())
        l1s.append(_l1(plan.expected_composition, t.as_fractions()))
    l1s = np.array(l1s)
    # Per-class rate rescaling pins the budget, so the expected composition is near-exact (the
    # residual is rate-clipping + the small-d quantization of a near-empty class).
    assert l1s.mean() < 0.04, f"mean expected-composition L1 {l1s.mean():.3f} too high"
    assert l1s.max() < 0.20, f"worst expected-composition L1 {l1s.max():.3f} too high"


def test_column_partition_matches_target_within_quantization():
    """Each class's column fraction is within one column of its target fraction (largest-remainder)."""
    t = CompositionTarget(f_mcar=0.5, f_mar=0.2, f_mnar=0.3, miss_rate=0.3)
    plan = plan_allocation(500, 22, t, RNGState(seed=3))
    w = sum(len(u.cols) for u in plan.units)
    cols_c = [0, 0, 0]
    for u in plan.units:
        cols_c[u.cls] += len(u.cols)
    for cls in (MCAR, MAR, MNAR):
        assert abs(cols_c[cls] / w - t.as_fractions()[cls]) <= 1.0 / w + 1e-9


def test_expected_miss_rate_near_target():
    rng = RNGState(seed=4)
    errs = []
    for _ in range(30):
        t = sample_composition_target(rng.spawn())
        plan = plan_allocation(1000, 20, t, rng.spawn())
        errs.append(abs(plan.expected_miss_rate - t.miss_rate))
    assert np.mean(errs) < 0.04


def test_mcar_is_per_column_only():
    """The honest seam: MCAR is the random anchor — no block structure."""
    t = CompositionTarget(f_mcar=0.8, f_mar=0.1, f_mnar=0.1, miss_rate=0.3)
    plan = plan_allocation(500, 20, t, RNGState(seed=5))
    for u in plan.units:
        if u.cls == MCAR:
            assert u.kind == "column" and len(u.cols) == 1


def test_mar_and_mnar_have_blocks_when_wide():
    """With enough columns, MAR and MNAR each produce at least one block (co-missingness)."""
    t = CompositionTarget(f_mcar=0.1, f_mar=0.45, f_mnar=0.45, miss_rate=0.35)
    plan = plan_allocation(800, 26, t, RNGState(seed=6))
    mar_blocks = [u for u in plan.units if u.cls == MAR and u.kind != "column"]
    mnar_blocks = [u for u in plan.units if u.cls == MNAR and u.kind != "column"]
    assert mar_blocks and mnar_blocks
    for u in mar_blocks + mnar_blocks:
        assert len(u.cols) >= _MIN_BLOCK_WIDTH


def test_mar_per_column_units_have_clean_observed_predictor():
    t = CompositionTarget(f_mcar=0.2, f_mar=0.6, f_mnar=0.2, miss_rate=0.3)
    plan = plan_allocation(500, 18, t, RNGState(seed=7))
    for u in plan.units:
        if u.cls == MAR and u.kind == "column":
            assert u.predictor in plan.observed_cols
            assert u.predictor not in u.cols


def test_blocks_carry_no_predictor():
    t = CompositionTarget(f_mcar=0.1, f_mar=0.4, f_mnar=0.5, miss_rate=0.35)
    plan = plan_allocation(800, 26, t, RNGState(seed=8))
    for u in plan.units:
        if u.kind != "column":
            assert u.predictor is None


def test_rates_vary_for_band2_realism():
    """Per-column rates span a wide range (Stage-A Band 2): some low, some heavily missing."""
    t = CompositionTarget(f_mcar=0.34, f_mar=0.33, f_mnar=0.33, miss_rate=0.4)
    plan = plan_allocation(1000, 25, t, RNGState(seed=9))
    rates = np.array([u.rate for u in plan.units])
    assert rates.std() > 0.12
    assert (rates > 0.5).any() and (rates < 0.2).any()


def test_deterministic_under_same_seed():
    t = sample_composition_target(RNGState(seed=11))
    p1 = plan_allocation(600, 18, t, RNGState(seed=12))
    p2 = plan_allocation(600, 18, t, RNGState(seed=12))
    assert p1.units == p2.units and p1.observed_cols == p2.observed_cols


def test_largest_remainder_sums_and_is_close():
    assert _largest_remainder((0.5, 0.3, 0.2), 10) == (5, 3, 2)
    assert sum(_largest_remainder((0.333, 0.333, 0.334), 17)) == 17
    assert sum(_largest_remainder((0.7, 0.2, 0.1), 13)) == 13


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_d", [0, 1, 3])
def test_small_d_raises(bad_d):
    t = CompositionTarget(f_mcar=0.34, f_mar=0.33, f_mnar=0.33, miss_rate=0.3)
    with pytest.raises(ValueError, match="d >= 4"):
        plan_allocation(500, bad_d, t, RNGState(seed=0))


def test_zero_n_raises():
    t = CompositionTarget(f_mcar=0.34, f_mar=0.33, f_mnar=0.33, miss_rate=0.3)
    with pytest.raises(ValueError, match="n must be >= 1"):
        plan_allocation(0, 10, t, RNGState(seed=0))


def test_bad_frac_observed_range_raises():
    t = CompositionTarget(f_mcar=0.34, f_mar=0.33, f_mnar=0.33, miss_rate=0.3)
    with pytest.raises(ValueError, match="frac_observed_range"):
        plan_allocation(500, 10, t, RNGState(seed=0), frac_observed_range=(0.5, 0.2))


# ---------------------------------------------------------------------------
# mnar_block_share — per-class block override (Stage-F per-column-only MNAR)
# ---------------------------------------------------------------------------

# MNAR-heavy + wide so MNAR gets enough columns to host blocks under the default share.
_MNAR_HEAVY = CompositionTarget(f_mcar=0.2, f_mar=0.3, f_mnar=0.5, miss_rate=0.3)


def test_mnar_block_share_zero_forces_per_column_mnar():
    """mnar_block_share=0.0 => every MNAR unit is a per-column 'column' unit (no joint MNAR blocks)."""
    plan = plan_allocation(800, 40, _MNAR_HEAVY, RNGState(seed=2), mnar_block_share=0.0)
    mnar_units = [u for u in plan.units if u.cls == MNAR]
    assert mnar_units, "no MNAR units produced; check the heavy target"
    assert all(u.kind == "column" for u in mnar_units), \
        f"MNAR blocks survived: {[u.kind for u in mnar_units]}"


def test_mnar_block_share_none_allows_blocks():
    """Default (None) => MNAR may form blocks like MAR (bit-identical baseline)."""
    plan = plan_allocation(800, 40, _MNAR_HEAVY, RNGState(seed=2))
    mnar_kinds = {u.kind for u in plan.units if u.cls == MNAR}
    assert mnar_kinds - {"column"}, f"expected some MNAR block kinds, got {mnar_kinds}"


def test_mnar_block_share_zero_leaves_mar_blocks_intact():
    """The override is MNAR-only: MAR still forms its skip-logic blocks."""
    plan = plan_allocation(800, 40, _MNAR_HEAVY, RNGState(seed=2), mnar_block_share=0.0)
    mar_kinds = {u.kind for u in plan.units if u.cls == MAR}
    assert mar_kinds - {"column"}, f"MAR blocks were wrongly suppressed: {mar_kinds}"


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_mnar_block_share_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="mnar_block_share"):
        plan_allocation(500, 10, _MNAR_HEAVY, RNGState(seed=0), mnar_block_share=bad)
