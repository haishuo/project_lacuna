"""Tests for lacuna.priors.metadata_prior — the ADR-0008 metadata prior channel."""

import numpy as np
import pytest

from lacuna.priors.metadata_prior import (
    MCAR, MAR, MNAR, N_CLASSES, SEMANTIC_PRIOR_SPEC,
    reliability_to_strength, semantic_prior_alpha, combine_prior_likelihood,
    prior_mean, channel_disagreement, aggregate_column_priors, semantic_tier,
    gated_reliability, gated_semantic_prior_alpha, _GATE_FLOOR,
)


# ---------------------------------------------------------------------------
# Normal cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,favored", [
    ("lab_lod", MNAR), ("sensitive_disclosure", MNAR), ("skip_gated", MAR),
    ("planned_random", MCAR), ("administrative", MCAR), ("demographic_core", MAR),
    ("routine_measure", MAR),
])
def test_each_class_favors_correct_mechanism(cls, favored):
    a = semantic_prior_alpha(cls)
    assert a.shape == (N_CLASSES,)
    assert int(np.argmax(prior_mean(a))) == favored, f"{cls} should favour {favored}"


def test_indeterminate_is_flat():
    a = semantic_prior_alpha("indeterminate")
    assert np.allclose(a, np.ones(N_CLASSES))
    assert np.allclose(prior_mean(a), np.full(N_CLASSES, 1 / N_CLASSES))


def test_reliability_to_strength_hits_target():
    for r in (0.4, 0.55, 0.65, 0.70):
        k = reliability_to_strength(r)
        favored_prob = (1 + k) / (N_CLASSES + k)
        assert abs(favored_prob - r) < 1e-9


def test_prior_strength_scales_with_tier():
    """Gut/moderate/weak priors stay overridable (<=0.71); fact-tier priors are strong (>gut, <0.95);
    flat is uniform. The fact tier must be strictly stronger than the gut tier (Stage-P3 refinement)."""
    for cls, (favored, _r, tier) in SEMANTIC_PRIOR_SPEC.items():
        p = float(prior_mean(semantic_prior_alpha(cls)).max())
        assert semantic_tier(cls) == tier
        if tier in ("gut", "moderate", "weak"):
            assert p <= 0.71, f"{cls} ({tier}) should be overridable, got {p:.3f}"
        elif tier == "fact":
            assert 0.71 < p < 0.95, f"{cls} (fact) should be strong but not absolute, got {p:.3f}"
        else:  # flat
            assert abs(p - 1.0 / N_CLASSES) < 1e-9
    fact_p = prior_mean(semantic_prior_alpha("planned_random")).max()
    gut_p = prior_mean(semantic_prior_alpha("sensitive_disclosure")).max()
    assert fact_p > gut_p, "fact-tier prior must be stronger than the gut-tier prior"


def test_unknown_class_tier_raises():
    with pytest.raises(ValueError, match="unknown semantic class"):
        semantic_tier("nope")


# ---------------------------------------------------------------------------
# Combination — graceful degradation, the lean, and the override
# ---------------------------------------------------------------------------

def test_flat_prior_is_a_noop():
    like = np.array([1.0, 3.0, 6.0])  # data evidence leaning MNAR
    post = combine_prior_likelihood(semantic_prior_alpha("indeterminate"), like)
    assert np.allclose(post, like)


def test_prior_leans_when_likelihood_is_flat():
    """On the non-identifiable axis (flat likelihood), the prior decides — the whole point."""
    flat_like = np.ones(N_CLASSES)
    post = combine_prior_likelihood(semantic_prior_alpha("sensitive_disclosure"), flat_like)
    assert int(np.argmax(prior_mean(post))) == MNAR


def test_strong_likelihood_overrides_modest_prior():
    """A confident, contrary data signal pulls the posterior off the prior (the override property)."""
    prior = semantic_prior_alpha("sensitive_disclosure")     # leans MNAR (~0.65)
    strong_mcar_like = np.array([1.0 + 12.0, 1.0, 1.0])      # data: strongly MCAR
    post = combine_prior_likelihood(prior, strong_mcar_like)
    assert int(np.argmax(prior_mean(post))) == MCAR


def test_combine_is_evidence_sum():
    p = np.array([1.0, 1.0, 4.0])
    q = np.array([1.0, 3.0, 1.0])
    assert np.allclose(combine_prior_likelihood(p, q), np.array([1.0, 3.0, 4.0]))


def test_channel_disagreement():
    flat = np.ones(N_CLASSES)
    assert channel_disagreement(flat, flat) == 0.0
    d = channel_disagreement(semantic_prior_alpha("sensitive_disclosure"),  # MNAR
                             semantic_prior_alpha("planned_random"))         # MCAR
    assert d > 0.3


# ---------------------------------------------------------------------------
# By-cell aggregation of per-column priors -> dataset prior
# ---------------------------------------------------------------------------

def test_aggregate_all_same_preserves_strength():
    """If every column carries the same prior, the dataset prior equals it (agreement keeps strength)."""
    a = semantic_prior_alpha("sensitive_disclosure")
    cols = np.stack([a, a, a])
    agg = aggregate_column_priors(cols, weights=np.array([3.0, 5.0, 2.0]))
    assert np.allclose(agg, a)


def test_aggregate_weights_toward_heavy_column():
    """A column with most of the missing cells dominates the dataset prior's mean."""
    mnar = semantic_prior_alpha("sensitive_disclosure")   # leans MNAR
    mcar = semantic_prior_alpha("planned_random")         # leans MCAR
    heavy_mnar = aggregate_column_priors(np.stack([mnar, mcar]), weights=np.array([9.0, 1.0]))
    assert int(np.argmax(prior_mean(heavy_mnar))) == MNAR
    heavy_mcar = aggregate_column_priors(np.stack([mnar, mcar]), weights=np.array([1.0, 9.0]))
    assert int(np.argmax(prior_mean(heavy_mcar))) == MCAR


def test_aggregate_flat_columns_stay_flat():
    flat = np.ones((4, N_CLASSES))
    agg = aggregate_column_priors(flat, weights=np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(agg, np.ones(N_CLASSES))


def test_aggregate_bad_shape_raises():
    with pytest.raises(ValueError, match=r"\[n, 3\]"):
        aggregate_column_priors(np.ones((3, 2)), weights=np.ones(3))


def test_aggregate_zero_total_weight_raises():
    with pytest.raises(ValueError, match="sum to > 0"):
        aggregate_column_priors(np.ones((2, N_CLASSES)), weights=np.zeros(2))


# ---------------------------------------------------------------------------
# Confidence gate on the fact tier (Stage-P3 lead 1)
# ---------------------------------------------------------------------------

def test_gated_fact_tier_scales_with_confidence():
    full = SEMANTIC_PRIOR_SPEC["planned_random"][1]   # 0.85
    assert abs(gated_reliability("planned_random", 1.0) - full) < 1e-9
    assert abs(gated_reliability("planned_random", 0.0) - _GATE_FLOOR) < 1e-9
    assert abs(gated_reliability("planned_random", 0.5) - (_GATE_FLOOR + (full - _GATE_FLOOR) * 0.5)) < 1e-9
    # monotonic increasing in confidence
    vals = [gated_reliability("lab_lod", c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("cls", ["sensitive_disclosure", "demographic_core", "administrative", "indeterminate"])
def test_gated_nonfact_ignores_confidence(cls):
    r = SEMANTIC_PRIOR_SPEC[cls][1]
    assert gated_reliability(cls, 0.0) == r and gated_reliability(cls, 1.0) == r
    # and the gated alpha equals the ungated alpha regardless of confidence
    assert np.allclose(gated_semantic_prior_alpha(cls, 0.1), semantic_prior_alpha(cls))


def test_gated_fact_alpha_full_conf_equals_ungated():
    assert np.allclose(gated_semantic_prior_alpha("planned_random", 1.0),
                       semantic_prior_alpha("planned_random"))


def test_low_confidence_fact_prior_is_more_overridable():
    """A contrary likelihood overrides a LOW-confidence fact prior but not a HIGH-confidence one."""
    contrary = np.array([1.0, 1.0 + 9.0, 1.0])  # data leans MAR
    hi = combine_prior_likelihood(gated_semantic_prior_alpha("planned_random", 1.0), contrary)  # MCAR prior
    lo = combine_prior_likelihood(gated_semantic_prior_alpha("planned_random", 0.0), contrary)
    assert int(np.argmax(prior_mean(hi))) == MCAR    # confident fact resists the contrary data
    assert int(np.argmax(prior_mean(lo))) == MAR     # unconfident fact is overridden by the data


def test_gated_bad_confidence_raises():
    with pytest.raises(ValueError, match="confidence must be in"):
        gated_reliability("planned_random", 1.5)


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_unknown_semantic_class_raises():
    with pytest.raises(ValueError, match="unknown semantic class"):
        semantic_prior_alpha("not_a_class")


@pytest.mark.parametrize("bad_r", [0.2, 1.0, 1.5, -0.1])
def test_reliability_out_of_range_raises(bad_r):
    with pytest.raises(ValueError, match="target reliability"):
        reliability_to_strength(bad_r)


def test_combine_bad_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        combine_prior_likelihood(np.ones(2), np.ones(N_CLASSES))


def test_combine_subunit_alpha_raises():
    with pytest.raises(ValueError, match=">= 1"):
        combine_prior_likelihood(np.array([0.5, 1.0, 1.0]), np.ones(N_CLASSES))
