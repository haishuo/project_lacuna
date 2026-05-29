"""
Tests for per-column targeting of threshold/detection MNAR generators (ADR-0006).

Covers `lacuna.generators.families.mnar._affected_cols.resolve_affected_cols` and the
`target_col_idx` behaviour it gives the threshold (`censoring.py`) and detection-limit
(`detection.py`) MNAR generators. The contract:

  - DEFAULT (no `target_col_idx`): unchanged — a random `affected_frac` fraction of columns.
  - TARGETED (`target_col_idx=k`): exactly column k is affected; negative wraps from the end;
    out-of-range wraps modulo d.

Normal / edge / failure cases per Coding Bible Rule 7.
"""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mnar._affected_cols import resolve_affected_cols
from lacuna.generators.families.mnar.censoring import (
    MNARThresholdLeft,
    MNARThresholdRight,
    MNARThresholdTwoSided,
    MNARSoftThreshold,
    MNARQuantile70,
    MNARQuantile80,
    MNARQuantile90,
    MNARColumnSpecificThreshold,
)
from lacuna.generators.families.mnar.detection import (
    MNARDetectionLower,
    MNARDetectionUpper,
    MNARDetectionBoth,
)
from lacuna.core.types import MNAR


# Generators that take percentile-style params and support target_col_idx. Each is constructed
# with a strong miss probability so the targeted column reliably shows missingness.
_TARGETABLE = [
    (MNARThresholdLeft, dict(percentile=60, miss_prob=0.95)),
    (MNARThresholdRight, dict(percentile=40, miss_prob=0.95)),
    (MNARThresholdTwoSided, dict(lower_percentile=20, upper_percentile=80, miss_prob=0.95)),
    (MNARSoftThreshold, dict(percentile=50, steepness=5.0)),
    (MNARQuantile70, dict(miss_prob=0.95)),
    (MNARQuantile80, dict(miss_prob=0.95)),
    (MNARQuantile90, dict(miss_prob=0.95)),
    (MNARColumnSpecificThreshold, dict(percentile_range=[40, 60], miss_prob=0.95)),
    (MNARDetectionLower, dict(detection_percentile=30)),
    (MNARDetectionUpper, dict(detection_percentile=70)),
    (MNARDetectionBoth, dict(lower_percentile=20, upper_percentile=80)),
]


@pytest.fixture
def X():
    """A reproducible 600x6 standard-normal matrix."""
    return RNGState(seed=123).randn(600, 6)


def _missing_cols(R: torch.Tensor) -> list:
    return (~R.all(dim=0)).nonzero().flatten().tolist()


# =============================================================================
# resolve_affected_cols — the shared helper
# =============================================================================

class TestResolveAffectedCols:
    def test_default_mode_returns_random_fraction(self):
        """No target_col_idx -> ~affected_frac*d columns, numpy int array."""
        params = GeneratorParams(affected_frac=0.5)
        cols = resolve_affected_cols(params, d=6, rng=RNGState(seed=1))
        assert isinstance(cols, np.ndarray)
        assert len(cols) == 3  # 6 * 0.5
        assert all(0 <= int(c) < 6 for c in cols)

    def test_default_mode_at_least_one_column(self):
        """affected_frac small still yields >= 1 column (max(1, ...))."""
        params = GeneratorParams(affected_frac=0.01)
        cols = resolve_affected_cols(params, d=6, rng=RNGState(seed=1))
        assert len(cols) == 1

    def test_default_mode_is_deterministic(self):
        """Same seed -> identical random selection."""
        params = GeneratorParams(affected_frac=0.5)
        a = resolve_affected_cols(params, 6, RNGState(seed=42))
        b = resolve_affected_cols(params, 6, RNGState(seed=42))
        assert list(a) == list(b)

    def test_targeted_returns_single_column(self):
        params = GeneratorParams(target_col_idx=3)
        cols = resolve_affected_cols(params, d=6, rng=RNGState(seed=1))
        assert list(cols) == [3]

    def test_targeted_ignores_rng(self):
        """Targeted mode is deterministic regardless of RNG state."""
        params = GeneratorParams(target_col_idx=2)
        a = resolve_affected_cols(params, 6, RNGState(seed=1))
        b = resolve_affected_cols(params, 6, RNGState(seed=999))
        assert list(a) == list(b) == [2]

    def test_negative_index_wraps_from_end(self):
        assert list(resolve_affected_cols(GeneratorParams(target_col_idx=-1), 6, RNGState(seed=1))) == [5]
        assert list(resolve_affected_cols(GeneratorParams(target_col_idx=-2), 6, RNGState(seed=1))) == [4]

    def test_out_of_range_wraps_modulo(self):
        assert list(resolve_affected_cols(GeneratorParams(target_col_idx=7), 6, RNGState(seed=1))) == [1]
        assert list(resolve_affected_cols(GeneratorParams(target_col_idx=6), 6, RNGState(seed=1))) == [0]

    def test_target_ignores_affected_frac(self):
        """When targeted, affected_frac is irrelevant — still exactly one column."""
        params = GeneratorParams(target_col_idx=0, affected_frac=1.0)
        assert list(resolve_affected_cols(params, 6, RNGState(seed=1))) == [0]

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError, match="d >= 1"):
            resolve_affected_cols(GeneratorParams(target_col_idx=0), d=0, rng=RNGState(seed=1))

    def test_non_int_target_raises(self):
        with pytest.raises(ValueError, match="must be an int"):
            resolve_affected_cols(GeneratorParams(target_col_idx=1.5), d=6, rng=RNGState(seed=1))

    def test_bool_target_raises(self):
        """bool is an int subclass in Python — reject it explicitly (a True index is a bug)."""
        with pytest.raises(ValueError, match="must be an int"):
            resolve_affected_cols(GeneratorParams(target_col_idx=True), d=6, rng=RNGState(seed=1))


# =============================================================================
# target_col_idx behaviour across all targetable threshold/detection generators
# =============================================================================

class TestGeneratorTargeting:
    @pytest.mark.parametrize("gen_cls,extra", _TARGETABLE)
    def test_targets_only_named_column(self, gen_cls, extra, X):
        """With target_col_idx=k, ONLY column k may carry missingness."""
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=2, **extra))
        R = gen.apply_to(X, RNGState(seed=5))
        assert _missing_cols(R) == [2], f"{gen_cls.__name__}: {_missing_cols(R)}"

    @pytest.mark.parametrize("gen_cls,extra", _TARGETABLE)
    def test_targeted_column_actually_loses_values(self, gen_cls, extra, X):
        """The targeted column should actually go partially missing (rule fired)."""
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=4, **extra))
        R = gen.apply_to(X, RNGState(seed=5))
        col_obs = R[:, 4].float().mean().item()
        assert 0.0 < col_obs < 1.0, f"{gen_cls.__name__}: obs frac {col_obs}"

    @pytest.mark.parametrize("gen_cls,extra", _TARGETABLE)
    def test_negative_target_wraps(self, gen_cls, extra, X):
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=-1, **extra))
        R = gen.apply_to(X, RNGState(seed=5))
        assert _missing_cols(R) == [5], f"{gen_cls.__name__}: {_missing_cols(R)}"

    @pytest.mark.parametrize("gen_cls,extra", _TARGETABLE)
    def test_default_mode_can_affect_multiple_columns(self, gen_cls, extra, X):
        """Without target_col_idx, the default affected_frac=0.5 affects multiple of 6 columns."""
        gen = gen_cls(0, "t", GeneratorParams(**extra))
        R = gen.apply_to(X, RNGState(seed=5))
        # default affected_frac 0.5 of d=6 -> 3 columns selected (some may not fire, but >1 typical)
        assert len(_missing_cols(R)) >= 1, f"{gen_cls.__name__}: {_missing_cols(R)}"

    @pytest.mark.parametrize("gen_cls,extra", _TARGETABLE)
    def test_targeting_is_deterministic(self, gen_cls, extra, X):
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=1, **extra))
        R1 = gen.apply_to(X, RNGState(seed=8))
        R2 = gen.apply_to(X, RNGState(seed=8))
        assert torch.equal(R1, R2)

    @pytest.mark.parametrize("gen_cls,extra", _TARGETABLE)
    def test_class_id_is_mnar(self, gen_cls, extra):
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=0, **extra))
        assert gen.class_id == MNAR
