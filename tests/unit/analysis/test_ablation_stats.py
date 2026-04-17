"""
Tests for lacuna.analysis.ablation_stats

Covers:
    Normal cases:
        - Clear positive shift yields small p-values, positive dz, CI excluding 0.
        - Clear negative shift yields small p-values, negative dz, CI excluding 0.
        - No shift (iid noise) yields non-significant p-values, CI around 0.
    Edge cases:
        - Zero-variance delta: paired_comparison returns NaN dz but valid p.
        - Bootstrap is deterministic under fixed seed.
        - Permutation p-value obeys (1 + count) / (1 + n_perm) floor.
        - One-sided alternatives behave correctly.
    Failure cases:
        - Length mismatch raises ValueError.
        - Non-1D input raises.
        - Non-finite input raises.
        - n_boot / n_perm below minimum raises.
        - Bad alternative string raises.
"""

import numpy as np
import pytest

from lacuna.analysis.ablation_stats import (
    PairedComparison,
    BootstrapDeltaCI,
    PairedPermutation,
    paired_comparison,
    bootstrap_delta_ci,
    paired_permutation,
    format_ablation_row,
)


# =============================================================================
# Normal cases — known-answer properties
# =============================================================================


def test_positive_shift_detected():
    """Variant consistently higher than baseline → small p, positive delta."""
    rng = np.random.default_rng(0)
    baseline = rng.standard_normal(30)
    variant = baseline + 0.5 + 0.05 * rng.standard_normal(30)
    cmp = paired_comparison(baseline, variant)
    assert isinstance(cmp, PairedComparison)
    assert cmp.n == 30
    assert cmp.mean_delta == pytest.approx(0.5, abs=0.05)
    assert cmp.t_p_value < 1e-6
    assert cmp.wilcoxon_p_value < 1e-4
    assert cmp.cohen_dz > 1.0  # large but not infinite

    ci = bootstrap_delta_ci(baseline, variant, n_boot=500, seed=42)
    assert ci.ci_low > 0  # CI excludes zero and is on the positive side
    assert ci.ci_high > ci.ci_low
    assert ci.ci_low < ci.mean_delta < ci.ci_high

    perm = paired_permutation(baseline, variant, n_perm=999, seed=42)
    assert perm.p_value < 0.01


def test_negative_shift_detected():
    """Variant consistently lower than baseline → negative delta, significant."""
    rng = np.random.default_rng(1)
    baseline = rng.standard_normal(25)
    variant = baseline - 0.3 + 0.05 * rng.standard_normal(25)
    cmp = paired_comparison(baseline, variant)
    assert cmp.mean_delta < 0
    assert cmp.t_p_value < 0.01
    assert cmp.cohen_dz < 0

    ci = bootstrap_delta_ci(baseline, variant, n_boot=500, seed=7)
    assert ci.ci_high < 0

    perm = paired_permutation(baseline, variant, n_perm=999, seed=7)
    assert perm.p_value < 0.05


def test_no_shift_yields_nonsignificant():
    """Identical distributions → p near 1, CI covers 0."""
    rng = np.random.default_rng(2)
    baseline = rng.standard_normal(40)
    variant = rng.standard_normal(40)  # independent but same distribution
    cmp = paired_comparison(baseline, variant)
    assert cmp.t_p_value > 0.05
    assert cmp.wilcoxon_p_value > 0.05

    ci = bootstrap_delta_ci(baseline, variant, n_boot=500, seed=11)
    assert ci.ci_low < 0 < ci.ci_high

    perm = paired_permutation(baseline, variant, n_perm=999, seed=11)
    assert perm.p_value > 0.05


# =============================================================================
# Edge cases
# =============================================================================


def test_zero_variance_delta_gives_nan_dz():
    """Constant delta (same shift each pair) gives NaN dz but valid p."""
    baseline = np.arange(10, dtype=np.float64)
    variant = baseline.copy()  # delta is all zeros -> std=0
    cmp = paired_comparison(baseline, variant)
    assert cmp.mean_delta == 0.0
    assert np.isnan(cmp.cohen_dz)


def test_bootstrap_deterministic_with_seed():
    """Same inputs + same seed → identical CI."""
    rng = np.random.default_rng(3)
    b = rng.standard_normal(20)
    v = b + 0.2
    ci1 = bootstrap_delta_ci(b, v, n_boot=500, seed=99)
    ci2 = bootstrap_delta_ci(b, v, n_boot=500, seed=99)
    assert ci1.ci_low == ci2.ci_low
    assert ci1.ci_high == ci2.ci_high


def test_permutation_floor_and_field_invariants():
    """Permutation p-value is in [1/(1+R), 1] and fields populated."""
    rng = np.random.default_rng(4)
    b = rng.standard_normal(15)
    v = b + 2.0  # enormous effect → count likely 0 → p = 1/(1+R)
    perm = paired_permutation(b, v, n_perm=499, seed=5)
    assert isinstance(perm, PairedPermutation)
    assert perm.n_perm == 499
    assert perm.alternative == "two.sided"
    assert perm.p_value >= 1.0 / (1 + 499)  # floor
    assert perm.p_value == pytest.approx(1.0 / (1 + 499), abs=1e-12)


def test_permutation_one_sided_greater():
    """Large positive delta: 'greater' p small; 'less' p near 1."""
    rng = np.random.default_rng(5)
    b = rng.standard_normal(20)
    v = b + 0.5
    p_greater = paired_permutation(b, v, n_perm=999, alternative="greater", seed=1).p_value
    p_less = paired_permutation(b, v, n_perm=999, alternative="less", seed=1).p_value
    assert p_greater < 0.05
    assert p_less > 0.9


def test_format_ablation_row_runs():
    rng = np.random.default_rng(6)
    b = rng.standard_normal(12)
    v = b + 0.1
    cmp = paired_comparison(b, v)
    ci = bootstrap_delta_ci(b, v, n_boot=300, seed=0)
    perm = paired_permutation(b, v, n_perm=399, seed=0)
    row = format_ablation_row(
        name="disable_littles", comparison=cmp, ci=ci, permutation=perm
    )
    assert "disable_littles" in row
    assert "CI:" in row
    assert "n=12" in row


# =============================================================================
# Failure cases
# =============================================================================


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        paired_comparison(np.zeros(10), np.zeros(9))


def test_non_1d_raises():
    with pytest.raises(ValueError, match="1-D"):
        paired_comparison(np.zeros((3, 3)), np.zeros((3, 3)))


def test_non_finite_raises():
    with pytest.raises(ValueError, match="finite"):
        paired_comparison(
            np.array([1.0, 2.0, np.nan]), np.array([1.0, 2.0, 3.0])
        )


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="paired observations"):
        paired_comparison(np.array([1.0]), np.array([2.0]))


def test_n_boot_below_minimum_raises():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_delta_ci(np.zeros(10), np.ones(10), n_boot=50)


def test_n_perm_below_minimum_raises():
    with pytest.raises(ValueError, match="n_perm"):
        paired_permutation(np.zeros(10), np.ones(10), n_perm=50)


def test_bad_conf_level_raises():
    with pytest.raises(ValueError, match="conf_level"):
        bootstrap_delta_ci(np.zeros(10), np.ones(10), n_boot=500, conf_level=1.5)


def test_bad_alternative_raises():
    with pytest.raises(ValueError, match="alternative"):
        paired_permutation(
            np.zeros(10), np.ones(10), n_perm=500, alternative="nope"
        )
