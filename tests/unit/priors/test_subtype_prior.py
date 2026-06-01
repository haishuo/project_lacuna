"""Tests for lacuna.priors.subtype_prior — the metadata->subtype Dirichlet prior channel."""

import numpy as np
import pytest

from lacuna.priors.dirichlet_evidence import combine_evidence, evidence_mean
from lacuna.priors.subtype_ontology import (
    N_SUBTYPES, THRESHOLD_MNAR, DETECTION_MNAR, SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB,
)
from lacuna.priors.subtype_prior import (
    SEMANTIC_SUBTYPE_SPEC, _GATE_FLOOR, subtype_prior_alpha, subtype_tier,
    gated_reliability, gated_subtype_prior_alpha, selective_subtype_decision,
)


# --- the semantic -> subtype map ---------------------------------------------------------------

@pytest.mark.parametrize("cls,favored", [
    ("lab_lod", DETECTION_MNAR),
    ("skip_gated", MAR_SUB),
    ("planned_random", MCAR_SUB),
    ("sensitive_disclosure", SELF_CENSORING_MNAR),
    ("administrative", MCAR_SUB),
    ("demographic_core", MAR_SUB),
    ("routine_measure", MAR_SUB),
])
def test_each_class_favors_correct_subtype(cls, favored):
    a = subtype_prior_alpha(cls)
    assert a.shape == (N_SUBTYPES,)
    assert int(np.argmax(evidence_mean(a))) == favored


def test_indeterminate_is_flat():
    a = subtype_prior_alpha("indeterminate")
    assert np.allclose(a, np.ones(N_SUBTYPES))
    assert np.allclose(evidence_mean(a), np.full(N_SUBTYPES, 1.0 / N_SUBTYPES))


def test_no_semantic_class_favors_threshold():
    """threshold_mnar is a DATA-only subtype; metadata never supplies it (the honest seam)."""
    for cls, (favored, _r, _t) in SEMANTIC_SUBTYPE_SPEC.items():
        assert favored != THRESHOLD_MNAR, f"{cls} should not favour threshold (data-only)"


def test_prior_strength_scales_with_tier():
    flat = 1.0 / N_SUBTYPES
    for cls, (_favored, _r, tier) in SEMANTIC_SUBTYPE_SPEC.items():
        p = float(evidence_mean(subtype_prior_alpha(cls)).max())
        assert subtype_tier(cls) == tier
        if tier in ("gut", "moderate", "weak"):
            assert p <= 0.66, f"{cls} ({tier}) should be overridable, got {p:.3f}"
        elif tier == "fact":
            assert 0.66 < p < 0.95, f"{cls} (fact) should be strong but not absolute, got {p:.3f}"
        else:  # flat
            assert abs(p - flat) < 1e-9
    fact_p = evidence_mean(subtype_prior_alpha("planned_random")).max()
    gut_p = evidence_mean(subtype_prior_alpha("sensitive_disclosure")).max()
    assert fact_p > gut_p, "fact-tier prior must be stronger than the gut-tier prior"


# --- combination: graceful degradation, the lean, the override ---------------------------------

def test_flat_prior_is_a_noop():
    like = np.array([1.0, 5.0, 1.0, 1.0, 1.0])  # data leans threshold's neighbour
    assert np.allclose(combine_evidence(subtype_prior_alpha("indeterminate"), like), like)


def test_prior_leans_when_likelihood_flat():
    """Where the data is silent (flat likelihood), the prior decides the subtype — the whole point."""
    post = combine_evidence(subtype_prior_alpha("sensitive_disclosure"), np.ones(N_SUBTYPES))
    assert int(np.argmax(evidence_mean(post))) == SELF_CENSORING_MNAR


def test_strong_likelihood_overrides_modest_prior():
    """A confident threshold detection pulls the posterior off a self-censoring (gut) prior."""
    prior = subtype_prior_alpha("sensitive_disclosure")          # leans self_censoring (~0.65)
    strong_threshold = np.array([1.0 + 20.0, 1.0, 1.0, 1.0, 1.0])  # data: strongly threshold
    post = combine_evidence(prior, strong_threshold)
    assert int(np.argmax(evidence_mean(post))) == THRESHOLD_MNAR


def test_fact_prior_resists_modest_contrary_data():
    """A by-design fact (planned_random -> mcar) the data cannot see resists a modest contrary read."""
    prior = subtype_prior_alpha("planned_random")                # strong mcar fact (~0.85)
    modest_contrary = np.array([1.0, 1.0, 3.0, 1.0, 1.0])        # mild self_censoring lean
    post = combine_evidence(prior, modest_contrary)
    assert int(np.argmax(evidence_mean(post))) == MCAR_SUB


# --- confidence gate (Stage P4 transferred) ----------------------------------------------------

def test_gated_fact_scales_with_confidence():
    full = SEMANTIC_SUBTYPE_SPEC["planned_random"][1]
    assert abs(gated_reliability("planned_random", 1.0) - full) < 1e-9
    assert abs(gated_reliability("planned_random", 0.0) - _GATE_FLOOR) < 1e-9
    vals = [gated_reliability("lab_lod", c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("cls", ["sensitive_disclosure", "administrative", "demographic_core", "indeterminate"])
def test_gated_nonfact_ignores_confidence(cls):
    r = SEMANTIC_SUBTYPE_SPEC[cls][1]
    assert gated_reliability(cls, 0.0) == r and gated_reliability(cls, 1.0) == r
    assert np.allclose(gated_subtype_prior_alpha(cls, 0.1), subtype_prior_alpha(cls))


def test_gated_fact_full_conf_equals_ungated():
    assert np.allclose(gated_subtype_prior_alpha("lab_lod", 1.0), subtype_prior_alpha("lab_lod"))


def test_low_confidence_fact_is_more_overridable():
    contrary = np.array([1.0, 1.0, 1.0, 1.0, 1.0 + 9.0])  # data leans mar
    hi = combine_evidence(gated_subtype_prior_alpha("planned_random", 1.0), contrary)  # mcar fact
    lo = combine_evidence(gated_subtype_prior_alpha("planned_random", 0.0), contrary)
    assert int(np.argmax(evidence_mean(hi))) == MCAR_SUB   # confident fact resists
    assert int(np.argmax(evidence_mean(lo))) == MAR_SUB    # unconfident fact is overridden


# --- selective subtype decision (Stage P6 transferred) -----------------------------------------

def test_selective_commits_when_confident():
    alpha = np.array([1.0, 1.0, 1.0, 1.0, 16.0])  # peaked on mar, p_max = 16/20 = 0.8
    cls, p = selective_subtype_decision(alpha, commit_threshold=0.7)
    assert cls == MAR_SUB and p > 0.7


def test_selective_abstains_when_flat():
    cls, p = selective_subtype_decision(np.ones(N_SUBTYPES), commit_threshold=0.5)
    assert cls is None and abs(p - 1.0 / N_SUBTYPES) < 1e-9


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.1])
def test_selective_bad_threshold_raises(bad):
    with pytest.raises(ValueError, match="commit_threshold"):
        selective_subtype_decision(np.ones(N_SUBTYPES), bad)


# --- failure cases -----------------------------------------------------------------------------

def test_unknown_semantic_class_raises():
    with pytest.raises(ValueError, match="unknown semantic class"):
        subtype_prior_alpha("not_a_class")


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="unknown semantic class"):
        subtype_tier("nope")


def test_gated_bad_confidence_raises():
    with pytest.raises(ValueError, match="confidence must be in"):
        gated_reliability("planned_random", 1.5)
