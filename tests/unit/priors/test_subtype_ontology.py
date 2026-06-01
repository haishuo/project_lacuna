"""Tests for lacuna.priors.subtype_ontology — the O/L label spaces and their maps."""

import numpy as np
import pytest

from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.mnar_column_pool import MNAR_SUBTYPES
from lacuna.priors.subtype_ontology import (
    SUBTYPES, N_SUBTYPES, THRESHOLD_MNAR, DETECTION_MNAR, SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB,
    DETECTABLE_SUBTYPES, SUBTYPE_PARENT_MECHANISM,
    LIKELIHOOD_LABELS, N_LIKELIHOOD_LABELS, LIKE_THRESHOLD, LIKE_DETECTION, LIKE_INDETERMINATE,
    THRESHOLD_SUBTYPE_NAMES, DETECTION_SUBTYPE_NAMES,
    realised_subtype_to_ontology, realised_subtype_to_likelihood_label, embed_likelihood_evidence,
)


# --- ontology shape / consistency --------------------------------------------------------------

def test_ontology_is_five_class_with_two_detectable():
    assert N_SUBTYPES == 5 and len(SUBTYPES) == 5
    assert DETECTABLE_SUBTYPES == (THRESHOLD_MNAR, DETECTION_MNAR)
    assert N_LIKELIHOOD_LABELS == 3 and len(LIKELIHOOD_LABELS) == 3


def test_parent_mechanism_map_is_correct():
    assert SUBTYPE_PARENT_MECHANISM[THRESHOLD_MNAR] == MNAR
    assert SUBTYPE_PARENT_MECHANISM[DETECTION_MNAR] == MNAR
    assert SUBTYPE_PARENT_MECHANISM[SELF_CENSORING_MNAR] == MNAR
    assert SUBTYPE_PARENT_MECHANISM[MCAR_SUB] == MCAR
    assert SUBTYPE_PARENT_MECHANISM[MAR_SUB] == MAR


def test_loud_families_are_real_pool_subtypes():
    """Guard against a stale enumeration: every loud name must exist in the diverse MNAR pool."""
    assert THRESHOLD_SUBTYPE_NAMES <= set(MNAR_SUBTYPES)
    assert DETECTION_SUBTYPE_NAMES <= set(MNAR_SUBTYPES)
    assert THRESHOLD_SUBTYPE_NAMES.isdisjoint(DETECTION_SUBTYPE_NAMES)


# --- realised_subtype_to_ontology --------------------------------------------------------------

@pytest.mark.parametrize("subtype", sorted(THRESHOLD_SUBTYPE_NAMES))
def test_mnar_threshold_maps_to_threshold(subtype):
    assert realised_subtype_to_ontology(MNAR, subtype) == THRESHOLD_MNAR


@pytest.mark.parametrize("subtype", sorted(DETECTION_SUBTYPE_NAMES))
def test_mnar_detection_maps_to_detection(subtype):
    assert realised_subtype_to_ontology(MNAR, subtype) == DETECTION_MNAR


@pytest.mark.parametrize("subtype", ["self_censoring", "under_report", "gaming", "volunteer",
                                     "symptom_triggered", "privacy", "demographic_dependent"])
def test_quiet_mnar_maps_to_self_censoring_catchall(subtype):
    assert realised_subtype_to_ontology(MNAR, subtype) == SELF_CENSORING_MNAR


def test_mcar_and_mar_map_by_parent_mechanism_ignoring_subtype():
    assert realised_subtype_to_ontology(MCAR, "anything") == MCAR_SUB
    assert realised_subtype_to_ontology(MAR, "mar_probit") == MAR_SUB
    assert realised_subtype_to_ontology(MAR, "logistic") == MAR_SUB


def test_to_ontology_invalid_mechanism_raises():
    with pytest.raises(ValueError, match="mechanism must be MCAR"):
        realised_subtype_to_ontology(-1, "self_censoring")  # OBSERVED sentinel must not be mapped


# --- realised_subtype_to_likelihood_label ------------------------------------------------------

@pytest.mark.parametrize("subtype", sorted(THRESHOLD_SUBTYPE_NAMES))
def test_threshold_label(subtype):
    assert realised_subtype_to_likelihood_label(MNAR, subtype) == LIKE_THRESHOLD


@pytest.mark.parametrize("subtype", sorted(DETECTION_SUBTYPE_NAMES))
def test_detection_label(subtype):
    assert realised_subtype_to_likelihood_label(MNAR, subtype) == LIKE_DETECTION


@pytest.mark.parametrize("mech,subtype", [
    (MNAR, "self_censoring"), (MNAR, "under_report"), (MNAR, "gaming"),
    (MAR, "mar_logistic"), (MAR, "logistic"), (MCAR, "mcar"),
])
def test_non_loud_folds_to_indeterminate_reject(mech, subtype):
    assert realised_subtype_to_likelihood_label(mech, subtype) == LIKE_INDETERMINATE


def test_to_label_invalid_mechanism_raises():
    with pytest.raises(ValueError, match="mechanism must be MCAR"):
        realised_subtype_to_likelihood_label(99, "self_censoring")


def test_loud_subtypes_are_detectable_and_label_matches_ontology():
    """A loud subtype's L label index coincides with its detectable O index (0/1 aligned)."""
    for s in THRESHOLD_SUBTYPE_NAMES | DETECTION_SUBTYPE_NAMES:
        o = realised_subtype_to_ontology(MNAR, s)
        ell = realised_subtype_to_likelihood_label(MNAR, s)
        assert o in DETECTABLE_SUBTYPES and o == ell


# --- embed_likelihood_evidence -----------------------------------------------------------------

def test_embed_threshold_prob_lands_on_threshold_class():
    p = np.array([1.0, 0.0, 0.0])  # confident threshold detection
    alpha = embed_likelihood_evidence(p, kappa=10.0)
    assert alpha.shape == (N_SUBTYPES,)
    assert np.isclose(alpha[THRESHOLD_MNAR], 1.0 + 10.0)
    assert np.allclose(np.delete(alpha, THRESHOLD_MNAR), 1.0)


def test_embed_detection_prob_lands_on_detection_class():
    alpha = embed_likelihood_evidence(np.array([0.0, 1.0, 0.0]), kappa=4.0)
    assert np.isclose(alpha[DETECTION_MNAR], 1.0 + 4.0)
    assert np.allclose(np.delete(alpha, DETECTION_MNAR), 1.0)


def test_embed_reject_prob_is_dropped_to_flat():
    """All-reject readout -> flat evidence over O (defers entirely to the prior)."""
    alpha = embed_likelihood_evidence(np.array([0.0, 0.0, 1.0]), kappa=10.0)
    assert np.allclose(alpha, np.ones(N_SUBTYPES))


def test_embed_mixed_and_kappa_scaling():
    p = np.array([0.5, 0.2, 0.3])  # 0.3 reject mass dropped
    alpha = embed_likelihood_evidence(p, kappa=20.0)
    assert np.isclose(alpha[THRESHOLD_MNAR], 1.0 + 20.0 * 0.5)
    assert np.isclose(alpha[DETECTION_MNAR], 1.0 + 20.0 * 0.2)
    assert np.allclose(alpha[[SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB]], 1.0)


def test_embed_batched():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    alpha = embed_likelihood_evidence(p, kappa=5.0)
    assert alpha.shape == (2, N_SUBTYPES)
    assert np.isclose(alpha[0, THRESHOLD_MNAR], 6.0)
    assert np.allclose(alpha[1], 1.0)


def test_embed_zero_kappa_is_flat():
    alpha = embed_likelihood_evidence(np.array([0.9, 0.1, 0.0]), kappa=0.0)
    assert np.allclose(alpha, np.ones(N_SUBTYPES))


def test_embed_bad_last_dim_raises():
    with pytest.raises(ValueError, match="last dim must be 3"):
        embed_likelihood_evidence(np.array([0.5, 0.5]), kappa=1.0)


def test_embed_negative_prob_raises():
    with pytest.raises(ValueError, match="finite and >= 0"):
        embed_likelihood_evidence(np.array([-0.1, 0.6, 0.5]), kappa=1.0)


def test_embed_negative_kappa_raises():
    with pytest.raises(ValueError, match="kappa must be finite and >= 0"):
        embed_likelihood_evidence(np.array([0.5, 0.5, 0.0]), kappa=-1.0)
