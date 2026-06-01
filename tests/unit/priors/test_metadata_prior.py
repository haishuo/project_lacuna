"""Tests for lacuna.priors.metadata_prior — the ADR-0008 metadata prior channel."""

import numpy as np
import pytest

from lacuna.priors.metadata_prior import (
    MCAR, MAR, MNAR, N_CLASSES, SEMANTIC_PRIOR_SPEC,
    reliability_to_strength, semantic_prior_alpha, combine_prior_likelihood,
    prior_mean, channel_disagreement,
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


def test_prior_strength_bounded_for_overridability():
    """Every favoured prior probability stays <= ~0.70 so the data likelihood can override it."""
    for cls, (favored, _) in SEMANTIC_PRIOR_SPEC.items():
        p = prior_mean(semantic_prior_alpha(cls))
        assert p.max() <= 0.71, f"{cls} prior too strong ({p.max():.3f}) — not overridable"


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
