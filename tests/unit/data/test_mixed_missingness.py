"""Tests for lacuna.data.mixed_missingness — per-column mixed mechanism composition."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.ingestion import RawDataset
from lacuna.data.mixed_missingness import (
    OBSERVED,
    MixedMissingnessResult,
    compose_mixed_missingness,
)


def _make_raw(n: int = 120, d: int = 6, seed: int = 0, name: str = "synth") -> RawDataset:
    """Build a complete synthetic RawDataset (standard-normal columns)."""
    data = RNGState(seed=seed).randn(n, d).numpy()
    return RawDataset(
        data=data,
        feature_names=tuple(f"c{j}" for j in range(d)),
        name=name,
    )


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

def test_returns_result_with_expected_shapes():
    raw = _make_raw()
    classes = (OBSERVED, MCAR, MCAR, MAR, MNAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=1))

    assert isinstance(res, MixedMissingnessResult)
    assert res.observed.n == raw.n and res.observed.d == raw.d
    assert res.observed.r.shape == (raw.n, raw.d)
    assert res.column_classes == classes
    assert len(res.per_column_miss_rate) == raw.d
    assert res.source_name == "synth"


def test_observed_columns_have_no_missingness():
    raw = _make_raw()
    classes = (OBSERVED, OBSERVED, MCAR, MAR, MNAR, MCAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=2))
    r = res.observed.r
    for j, c in enumerate(classes):
        if c == OBSERVED:
            assert bool(r[:, j].all()), f"OBSERVED column {j} should be fully observed"
            assert res.per_column_miss_rate[j] == 0.0


def test_non_observed_columns_acquire_missingness():
    raw = _make_raw()
    classes = (OBSERVED, MCAR, MAR, MNAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=3))
    for j, c in enumerate(classes):
        if c != OBSERVED:
            assert res.per_column_miss_rate[j] > 0.0, f"column {j} ({c}) should have missingness"


def test_mar_predictors_are_clean_and_not_self():
    raw = _make_raw()
    classes = (OBSERVED, MCAR, MAR, MAR, MNAR, OBSERVED)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=4))
    clean_pool = {j for j, c in enumerate(classes) if c in (OBSERVED, MCAR)}
    assert set(res.mar_predictors.keys()) == {2, 3}  # the two MAR columns
    for target, predictor in res.mar_predictors.items():
        assert predictor != target
        assert predictor in clean_pool, "MAR predictor must be a clean (observed/MCAR) column"


def test_deterministic_under_same_seed():
    raw = _make_raw()
    classes = (OBSERVED, MCAR, MAR, MNAR, MNAR, MCAR)
    r1 = compose_mixed_missingness(raw, classes, RNGState(seed=7)).observed.r
    r2 = compose_mixed_missingness(raw, classes, RNGState(seed=7)).observed.r
    assert torch.equal(r1, r2)


def test_different_seed_changes_mask():
    raw = _make_raw()
    classes = (OBSERVED, MCAR, MAR, MNAR, MNAR, MCAR)
    r1 = compose_mixed_missingness(raw, classes, RNGState(seed=7)).observed.r
    r2 = compose_mixed_missingness(raw, classes, RNGState(seed=8)).observed.r
    assert not torch.equal(r1, r2)


def test_miss_rate_tracks_target_for_all_mechanisms():
    # Larger n for a stable marginal estimate. All three mechanisms should land in a
    # comparable band around the target (controls the miss-rate confound).
    raw = _make_raw(n=2000, d=6, seed=11)
    classes = (OBSERVED, MCAR, MAR, MNAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=5), target_miss_rate=0.25)
    for j, c in enumerate(classes):
        if c != OBSERVED:
            assert 0.10 <= res.per_column_miss_rate[j] <= 0.45, (
                f"column {j} ({c}) rate {res.per_column_miss_rate[j]} out of band"
            )


def test_observed_values_zeroed_where_missing():
    raw = _make_raw()
    classes = (OBSERVED, MCAR, MAR, MNAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=6))
    x, r = res.observed.x, res.observed.r
    assert torch.all(x[~r] == 0.0), "missing cells must be zeroed"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_minimal_two_columns_mar():
    raw = _make_raw(n=100, d=2)
    res = compose_mixed_missingness(raw, (OBSERVED, MAR), RNGState(seed=9))
    assert res.mar_predictors[1] == 0


def test_no_column_left_fully_missing_at_high_rate():
    # High target rate + small n could zero a whole column; the guard must prevent it.
    raw = _make_raw(n=12, d=4, seed=13)
    classes = (OBSERVED, MCAR, MNAR, MCAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=3), target_miss_rate=0.95)
    for j in range(raw.d):
        assert bool(res.observed.r[:, j].any()), f"column {j} must keep >=1 observed cell"


def test_mcar_columns_can_serve_as_clean_predictor_when_no_observed():
    # No OBSERVED column, but an MCAR column exists -> MAR is allowed (clean pool = MCAR).
    raw = _make_raw(n=100, d=4)
    classes = (MCAR, MAR, MNAR, MCAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=10))
    assert res.mar_predictors[1] in (0, 3)


# ---------------------------------------------------------------------------
# Failure cases (fail loud — Coding Bible Rule 1)
# ---------------------------------------------------------------------------

def test_wrong_length_raises():
    raw = _make_raw(d=5)
    with pytest.raises(ValueError, match="length"):
        compose_mixed_missingness(raw, (MCAR, MAR), RNGState(seed=0))


def test_invalid_class_raises():
    raw = _make_raw(d=4)
    with pytest.raises(ValueError, match="invalid"):
        compose_mixed_missingness(raw, (OBSERVED, MCAR, 99, MAR), RNGState(seed=0))


def test_mar_without_clean_predictor_raises():
    raw = _make_raw(d=3)
    with pytest.raises(ValueError, match="clean predictor"):
        compose_mixed_missingness(raw, (MAR, MAR, MNAR), RNGState(seed=0))


def test_bad_target_miss_rate_raises():
    raw = _make_raw(d=4)
    with pytest.raises(ValueError, match="target_miss_rate"):
        compose_mixed_missingness(
            raw, (OBSERVED, MCAR, MAR, MNAR), RNGState(seed=0), target_miss_rate=1.5
        )


def test_mar_requires_two_columns():
    raw = _make_raw(n=50, d=1)
    with pytest.raises(ValueError, match="d >= 2"):
        compose_mixed_missingness(raw, (MAR,), RNGState(seed=0))


# ---------------------------------------------------------------------------
# Diverse-MNAR regime (Stage 4 follow-up)
# ---------------------------------------------------------------------------

def test_default_mnar_is_self_censoring_only():
    """Without mnar_diverse, every MNAR column is self-censoring (Stage 0-3 behaviour)."""
    raw = _make_raw()
    classes = (OBSERVED, MNAR, MNAR, MNAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=1))
    assert set(res.mnar_subtypes.keys()) == {1, 2, 3, 5}
    assert set(res.mnar_subtypes.values()) == {"self_censoring"}


def test_diverse_mnar_records_subtype_per_mnar_column():
    raw = _make_raw()
    classes = (OBSERVED, MNAR, MNAR, MNAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=1), mnar_diverse=True)
    # one subtype recorded per MNAR column, no others
    assert set(res.mnar_subtypes.keys()) == {1, 2, 3, 5}
    from lacuna.data.mnar_column_pool import MNAR_SUBTYPES
    assert all(v in MNAR_SUBTYPES for v in res.mnar_subtypes.values())


def test_diverse_mnar_labels_stay_mnar():
    """The per-column CLASS is still MNAR regardless of the realised subtype."""
    raw = _make_raw()
    classes = (OBSERVED, MNAR, MNAR, MNAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=2), mnar_diverse=True)
    assert res.column_classes == classes  # labels unchanged


def test_diverse_mnar_columns_actually_go_missing():
    raw = _make_raw(n=300)
    classes = (OBSERVED, MNAR, MNAR, MNAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=3), mnar_diverse=True)
    for j in (1, 2, 3, 5):
        assert res.per_column_miss_rate[j] > 0.0, (j, res.per_column_miss_rate)


def test_diverse_mnar_produces_subtype_variety_across_seeds():
    """Across many composes, MNAR columns realise more than one subtype (not collapsed)."""
    raw = _make_raw(n=200)
    classes = (OBSERVED, MNAR, MNAR, MNAR, MNAR, MCAR)
    seen = set()
    for s in range(40):
        res = compose_mixed_missingness(raw, classes, RNGState(seed=s), mnar_diverse=True)
        seen.update(res.mnar_subtypes.values())
    assert len(seen) >= 4, f"diverse regime collapsed to {seen}"


def test_diverse_mnar_deterministic_under_same_seed():
    raw = _make_raw()
    classes = (OBSERVED, MNAR, MNAR, MAR, MCAR, MNAR)
    a = compose_mixed_missingness(raw, classes, RNGState(seed=7), mnar_diverse=True)
    b = compose_mixed_missingness(raw, classes, RNGState(seed=7), mnar_diverse=True)
    assert a.mnar_subtypes == b.mnar_subtypes
    assert torch.equal(a.observed.r, b.observed.r)
