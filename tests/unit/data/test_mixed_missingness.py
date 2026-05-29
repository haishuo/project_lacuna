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


# ---------------------------------------------------------------------------
# Diverse MAR (Stage 5) — mirrors the diverse-MNAR contract
# ---------------------------------------------------------------------------

def test_default_mar_is_logistic_only():
    """Without mar_diverse, every MAR column is the single logistic family (Stage 0-4b)."""
    raw = _make_raw()
    classes = (OBSERVED, MAR, MAR, MAR, MCAR, MAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=1))
    assert set(res.mar_subtypes.keys()) == {1, 2, 3, 5}
    assert set(res.mar_subtypes.values()) == {"logistic"}


def test_diverse_mar_records_subtype_per_mar_column():
    raw = _make_raw()
    classes = (OBSERVED, MAR, MAR, MAR, MCAR, MAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=1), mar_diverse=True)
    assert set(res.mar_subtypes.keys()) == {1, 2, 3, 5}
    from lacuna.data.mar_column_pool import MAR_SUBTYPES
    assert all(v in MAR_SUBTYPES for v in res.mar_subtypes.values())


def test_diverse_mar_labels_stay_mar():
    raw = _make_raw()
    classes = (OBSERVED, MAR, MAR, MAR, MCAR, MAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=2), mar_diverse=True)
    assert res.column_classes == classes


def test_diverse_mar_predictors_stay_clean():
    """The clean-MAR regime must hold under diversity: every MAR predictor is OBSERVED or MCAR
    and never the column itself."""
    raw = _make_raw()
    classes = (OBSERVED, MAR, MAR, MAR, MCAR, MAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=4), mar_diverse=True)
    for target, predictor in res.mar_predictors.items():
        assert predictor != target
        assert classes[predictor] in (OBSERVED, MCAR), (target, predictor, classes[predictor])


def test_diverse_mar_columns_actually_go_missing():
    raw = _make_raw(n=300)
    classes = (OBSERVED, MAR, MAR, MAR, MCAR, MAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=3), mar_diverse=True)
    for j in (1, 2, 3, 5):
        assert res.per_column_miss_rate[j] > 0.0, (j, res.per_column_miss_rate)


def test_diverse_mar_produces_subtype_variety_across_seeds():
    raw = _make_raw(n=200)
    classes = (OBSERVED, MAR, MAR, MAR, MAR, MCAR)
    seen = set()
    for s in range(40):
        res = compose_mixed_missingness(raw, classes, RNGState(seed=s), mar_diverse=True)
        seen.update(res.mar_subtypes.values())
    assert len(seen) >= 4, f"diverse MAR regime collapsed to {seen}"


def test_diverse_mar_deterministic_under_same_seed():
    raw = _make_raw()
    classes = (OBSERVED, MAR, MAR, MNAR, MCAR, MAR)
    a = compose_mixed_missingness(raw, classes, RNGState(seed=7), mar_diverse=True)
    b = compose_mixed_missingness(raw, classes, RNGState(seed=7), mar_diverse=True)
    assert a.mar_subtypes == b.mar_subtypes
    assert torch.equal(a.observed.r, b.observed.r)


def test_compensate_rate_default_is_logit_intercept():
    """Default (compensate_rate=False) is bit-identical to the legacy logit-intercept path."""
    raw = _make_raw(seed=9)
    classes = (OBSERVED, MAR, MNAR, MCAR, MAR, MNAR)
    a = compose_mixed_missingness(raw, classes, RNGState(seed=1))
    b = compose_mixed_missingness(raw, classes, RNGState(seed=1), compensate_rate=False)
    assert torch.equal(a.observed.r, b.observed.r)


def test_compensate_rate_brings_direct_paths_to_target():
    """compensate_rate makes the direct logistic-MAR / self-censoring-MNAR marginals hit ~target,
    instead of overshooting to ~0.30 (the slope-inflation the legacy path exhibits)."""
    raw = _make_raw(n=4000, d=6, seed=5)
    classes = (OBSERVED, MAR, MAR, MNAR, MNAR, MCAR)
    legacy = compose_mixed_missingness(raw, classes, RNGState(seed=2), target_miss_rate=0.25)
    comp = compose_mixed_missingness(raw, classes, RNGState(seed=2), target_miss_rate=0.25,
                                     compensate_rate=True)
    # Legacy MAR/MNAR overshoot; compensated land near 0.25.
    for j in (1, 2, 3, 4):
        assert legacy.per_column_miss_rate[j] > 0.27, (j, legacy.per_column_miss_rate[j])
        assert 0.20 <= comp.per_column_miss_rate[j] <= 0.30, (j, comp.per_column_miss_rate[j])


def test_compensate_rate_composes_with_diversity():
    """compensate_rate only touches the DIRECT paths; with diversity on it is a no-op on labels
    (the pools already compensate) and still produces valid per-column subtypes."""
    raw = _make_raw(n=300)
    classes = (OBSERVED, MAR, MNAR, MCAR, MAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=3),
                                    mar_diverse=True, mnar_diverse=True, compensate_rate=True)
    assert res.column_classes == classes
    for j in (1, 2, 4, 5):
        assert res.per_column_miss_rate[j] > 0.0


def test_full_diversity_both_mar_and_mnar():
    """Stage 5 regime: diverse MAR AND diverse MNAR coexist across columns in one dataset."""
    raw = _make_raw(n=300)
    classes = (OBSERVED, MAR, MNAR, MAR, MCAR, MNAR)
    res = compose_mixed_missingness(raw, classes, RNGState(seed=5),
                                    mar_diverse=True, mnar_diverse=True)
    from lacuna.data.mar_column_pool import MAR_SUBTYPES
    from lacuna.data.mnar_column_pool import MNAR_SUBTYPES
    assert set(res.mar_subtypes.keys()) == {1, 3}
    assert set(res.mnar_subtypes.keys()) == {2, 5}
    assert all(v in MAR_SUBTYPES for v in res.mar_subtypes.values())
    assert all(v in MNAR_SUBTYPES for v in res.mnar_subtypes.values())
    for j in (1, 2, 3, 5):
        assert res.per_column_miss_rate[j] > 0.0
