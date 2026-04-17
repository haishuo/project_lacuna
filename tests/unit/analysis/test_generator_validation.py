"""
Tests for lacuna.analysis.generator_validation

Covers:
    Normal cases:
        - A true MCAR generator passes validation.
        - A true MAR generator passes validation (rejects consistently).
    Edge cases:
        - All-observed and all-missing samples are skipped, not counted.
        - Fewer than 10 datasets is rejected.
        - n_rows <= n_cols is rejected.
    Failure cases:
        - An MCAR-labeled generator that actually emits MAR patterns fails.
        - Non-boolean `r` tensor fails loudly.
"""

from typing import Tuple

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from lacuna.analysis.generator_validation import (
    ValidationResult,
    VERDICT_PASS,
    VERDICT_FAIL,
    VERDICT_INDETERMINATE,
    validate_generator,
    _tensor_to_nan_array,
    format_results_table,
    summarize,
)


# =============================================================================
# Test generators (minimal, deterministic, no registry required)
# =============================================================================


class _TrueMCAR(Generator):
    """Bernoulli missingness independent of values — true MCAR."""

    def __init__(self, generator_id: int = 0, miss_rate: float = 0.2):
        super().__init__(
            generator_id=generator_id,
            name=f"true_mcar_{generator_id}",
            class_id=MCAR,
            params=GeneratorParams(miss_rate=miss_rate),
        )
        self._miss_rate = miss_rate

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rng.randn(n, d)
        r = rng.rand(n, d) > self._miss_rate
        return x, r


class _TrueMAR(Generator):
    """Missingness in cols 1..d-1 driven by col 0 — true MAR."""

    def __init__(self, generator_id: int = 1):
        super().__init__(
            generator_id=generator_id,
            name=f"true_mar_{generator_id}",
            class_id=MAR,
            params=GeneratorParams(miss_rate=0.3),
        )

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rng.randn(n, d)
        # Missingness in cols >=1 depends strongly on col 0 (observed).
        driver = x[:, 0:1]
        r = torch.ones(n, d, dtype=torch.bool)
        for j in range(1, d):
            # Higher col 0 → more likely missing
            prob_missing = torch.sigmoid(2.5 * driver.squeeze(-1))
            u = rng.rand(n)
            r[:, j] = u >= prob_missing
        return x, r


class _MislabeledMCAR(Generator):
    """Produces MAR patterns but claims to be MCAR — should fail validation."""

    def __init__(self, generator_id: int = 2):
        super().__init__(
            generator_id=generator_id,
            name=f"mislabeled_{generator_id}",
            class_id=MCAR,  # Lie — pattern is actually MAR
            params=GeneratorParams(miss_rate=0.3),
        )

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rng.randn(n, d)
        driver = x[:, 0:1]
        r = torch.ones(n, d, dtype=torch.bool)
        for j in range(1, d):
            prob_missing = torch.sigmoid(3.0 * driver.squeeze(-1))
            u = rng.rand(n)
            r[:, j] = u >= prob_missing
        return x, r


class _AllObserved(Generator):
    """Never introduces missingness — every sample is skipped by Little's test."""

    def __init__(self, generator_id: int = 3):
        super().__init__(
            generator_id=generator_id,
            name="all_observed",
            class_id=MCAR,
            params=GeneratorParams(miss_rate=0.0),
        )

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rng.randn(n, d)
        r = torch.ones(n, d, dtype=torch.bool)
        return x, r


# =============================================================================
# Normal cases
# =============================================================================


def test_true_mcar_passes_validation():
    """A real MCAR generator should reject at roughly alpha."""
    gen = _TrueMCAR(generator_id=0, miss_rate=0.2)
    result = validate_generator(
        gen, seed=7, n_datasets=40, n_rows=200, n_cols=5, alpha=0.05
    )
    assert isinstance(result, ValidationResult)
    assert result.class_id == MCAR
    assert result.n_datasets == 40
    # A few samples may be skipped (e.g. pystatistics solver warnings on
    # borderline cases); most must be usable.
    assert result.skipped_count < 10
    # Reject rate should be comfortably below alpha + tolerance (= 0.15)
    assert result.reject_rate <= 0.15
    assert result.verdict == VERDICT_PASS


def test_true_mar_passes_validation():
    """A real MAR generator should reject on nearly every sample."""
    gen = _TrueMAR(generator_id=1)
    result = validate_generator(
        gen, seed=11, n_datasets=25, n_rows=200, n_cols=5, alpha=0.05
    )
    assert result.class_id == MAR
    assert result.reject_rate >= 0.80
    assert result.verdict == VERDICT_PASS


def test_mnar_verdict_is_indeterminate():
    """MNAR generators are not adjudicated by Little's test (underpowered)."""

    class _TrueMNAR(Generator):
        def __init__(self):
            super().__init__(
                generator_id=99,
                name="true_mnar",
                class_id=MNAR,
                params=GeneratorParams(miss_rate=0.3),
            )

        def sample(self, rng, n, d):
            x = rng.randn(n, d)
            # Self-censor: drop values above 0.5 in every column
            r = x < 0.5
            return x, r

    result = validate_generator(
        _TrueMNAR(), seed=5, n_datasets=15, n_rows=150, n_cols=4, alpha=0.05
    )
    assert result.class_id == MNAR
    assert result.verdict == VERDICT_INDETERMINATE
    # Expected is NaN for MNAR
    assert result.expected_reject_rate != result.expected_reject_rate


# =============================================================================
# Edge cases
# =============================================================================


def test_all_observed_samples_are_skipped():
    """Generators with zero missingness yield skipped_count == n_datasets."""
    gen = _AllObserved()
    result = validate_generator(
        gen, seed=3, n_datasets=15, n_rows=100, n_cols=4, alpha=0.05
    )
    assert result.skipped_count == 15
    # reject_rate is NaN when everything was skipped; MCAR-declared
    # generator with no usable data must fail.
    assert result.verdict == VERDICT_FAIL


def test_tensor_to_nan_array_preserves_observed_values():
    """Observed cells survive; missing cells become NaN."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    r = torch.tensor([[True, False, True], [False, True, True]])
    arr = _tensor_to_nan_array(x, r)
    assert arr.dtype == np.float64
    assert arr.shape == (2, 3)
    assert arr[0, 0] == 1.0 and np.isnan(arr[0, 1]) and arr[0, 2] == 3.0
    assert np.isnan(arr[1, 0]) and arr[1, 1] == 5.0 and arr[1, 2] == 6.0


def test_reporting_helpers():
    """format_results_table and summarize produce sane output."""
    gen_mcar = _TrueMCAR(generator_id=0)
    gen_mar = _TrueMAR(generator_id=1)
    results = [
        validate_generator(gen_mcar, seed=1, n_datasets=15, n_rows=120, n_cols=4),
        validate_generator(gen_mar, seed=2, n_datasets=15, n_rows=120, n_cols=4),
    ]
    table = format_results_table(results)
    assert "true_mcar" in table
    assert "true_mar" in table
    n_pass, n_fail, n_indet = summarize(results)
    assert n_pass + n_fail + n_indet == 2

    assert format_results_table([]) == "(no results)"


# =============================================================================
# Failure cases
# =============================================================================


def test_mislabeled_mcar_fails_validation():
    """A generator labeled MCAR but emitting MAR patterns must fail."""
    gen = _MislabeledMCAR()
    result = validate_generator(
        gen, seed=13, n_datasets=25, n_rows=200, n_cols=5, alpha=0.05
    )
    # Reject-rate for MAR-pattern data should be high — far above MCAR ceiling.
    assert result.reject_rate > 0.5
    assert result.verdict == VERDICT_FAIL


def test_n_datasets_too_small_rejected():
    gen = _TrueMCAR()
    with pytest.raises(ValueError, match="n_datasets"):
        validate_generator(gen, seed=0, n_datasets=5)


def test_n_rows_not_exceeding_n_cols_rejected():
    gen = _TrueMCAR()
    with pytest.raises(ValueError, match="n_rows"):
        validate_generator(gen, seed=0, n_datasets=10, n_rows=5, n_cols=5)


def test_non_bool_r_rejected():
    x = torch.randn(10, 3)
    r_float = torch.ones(10, 3)  # wrong dtype
    with pytest.raises(ValueError, match="bool"):
        _tensor_to_nan_array(x, r_float)


def test_shape_mismatch_rejected():
    x = torch.randn(10, 3)
    r = torch.ones(10, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="Shape mismatch"):
        _tensor_to_nan_array(x, r)
