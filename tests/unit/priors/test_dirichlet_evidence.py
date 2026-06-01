"""Tests for lacuna.priors.dirichlet_evidence — the generic K-class Dirichlet evidence ops."""

import numpy as np
import pytest

from lacuna.priors.dirichlet_evidence import (
    reliability_to_strength, combine_evidence, aggregate_evidence, evidence_mean,
    channel_disagreement, argmax_decision,
)


# --- reliability_to_strength -------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 5, 8])
@pytest.mark.parametrize("r", [0.45, 0.6, 0.85])
def test_reliability_hits_target_for_any_k(k, r):
    if r < 1.0 / k:
        pytest.skip("r below the 1/K floor for this K")
    kappa = reliability_to_strength(r, k)
    favored_prob = (1 + kappa) / (k + kappa)
    assert abs(favored_prob - r) < 1e-9


def test_reliability_floor_is_zero_strength():
    assert reliability_to_strength(1.0 / 5, 5) == 0.0


@pytest.mark.parametrize("bad_r", [0.1, 1.0, 1.5, -0.1])
def test_reliability_out_of_range_raises(bad_r):
    with pytest.raises(ValueError, match="target reliability"):
        reliability_to_strength(bad_r, 5)


def test_reliability_bad_n_classes_raises():
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        reliability_to_strength(0.5, 1)


# --- combine_evidence --------------------------------------------------------------------------

def test_combine_is_evidence_sum():
    p = np.array([1.0, 1.0, 4.0, 1.0, 1.0])
    q = np.array([1.0, 3.0, 1.0, 1.0, 1.0])
    assert np.allclose(combine_evidence(p, q), np.array([1.0, 3.0, 4.0, 1.0, 1.0]))


def test_flat_prior_is_a_noop():
    like = np.array([1.0, 3.0, 6.0, 1.0, 1.0])
    assert np.allclose(combine_evidence(np.ones(5), like), like)


def test_strong_likelihood_overrides_modest_prior():
    prior = np.array([1.0, 1.0, 3.0, 1.0, 1.0])          # modest lean on class 2
    strong_like = np.array([1.0, 1.0, 1.0, 13.0, 1.0])   # data strongly on class 3
    post = combine_evidence(prior, strong_like)
    assert int(np.argmax(evidence_mean(post))) == 3


def test_combine_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share shape"):
        combine_evidence(np.ones(3), np.ones(5))


def test_combine_subunit_raises():
    with pytest.raises(ValueError, match=">= 1"):
        combine_evidence(np.array([0.5, 1.0, 1.0]), np.ones(3))


def test_combine_too_short_raises():
    with pytest.raises(ValueError, match="length K>=2"):
        combine_evidence(np.array([1.0]), np.array([1.0]))


# --- aggregate_evidence ------------------------------------------------------------------------

def test_aggregate_all_same_preserves_strength():
    a = np.array([1.0, 1.0, 4.0, 1.0, 1.0])
    cols = np.stack([a, a, a])
    agg = aggregate_evidence(cols, weights=np.array([3.0, 5.0, 2.0]))
    assert np.allclose(agg, a)


def test_aggregate_weights_toward_heavy_column():
    c2 = np.array([1.0, 1.0, 9.0, 1.0, 1.0])   # leans class 2
    c3 = np.array([1.0, 1.0, 1.0, 9.0, 1.0])   # leans class 3
    heavy2 = aggregate_evidence(np.stack([c2, c3]), weights=np.array([9.0, 1.0]))
    assert int(np.argmax(evidence_mean(heavy2))) == 2
    heavy3 = aggregate_evidence(np.stack([c2, c3]), weights=np.array([1.0, 9.0]))
    assert int(np.argmax(evidence_mean(heavy3))) == 3


def test_aggregate_flat_stays_flat():
    flat = np.ones((4, 5))
    assert np.allclose(aggregate_evidence(flat, weights=np.array([1.0, 2.0, 3.0, 4.0])), np.ones(5))


def test_aggregate_bad_shape_raises():
    with pytest.raises(ValueError, match=r"\[n_cols, K>=2\]"):
        aggregate_evidence(np.ones((3,)), weights=np.ones(3))


def test_aggregate_zero_total_weight_raises():
    with pytest.raises(ValueError, match="sum to > 0"):
        aggregate_evidence(np.ones((2, 5)), weights=np.zeros(2))


def test_aggregate_subunit_raises():
    with pytest.raises(ValueError, match=">= 1"):
        aggregate_evidence(np.array([[0.5, 1.0, 1.0]]), weights=np.ones(1))


# --- evidence_mean / channel_disagreement ------------------------------------------------------

def test_evidence_mean_normalises():
    m = evidence_mean(np.array([1.0, 1.0, 2.0, 1.0, 1.0]))
    assert abs(m.sum() - 1.0) < 1e-12 and np.argmax(m) == 2


def test_channel_disagreement_zero_when_equal_and_positive_when_opposed():
    flat = np.ones(5)
    assert channel_disagreement(flat, flat) == 0.0
    a = np.array([1.0, 1.0, 9.0, 1.0, 1.0])   # class 2
    b = np.array([1.0, 1.0, 1.0, 9.0, 1.0])   # class 3
    assert channel_disagreement(a, b) > 0.3


# --- argmax_decision ---------------------------------------------------------------------------

def test_argmax_commits_when_confident():
    alpha = np.array([1.0, 1.0, 9.0, 1.0, 1.0])  # p_max = 9/13 ~ 0.69
    cls, p = argmax_decision(alpha, commit_threshold=0.6)
    assert cls == 2 and p > 0.6


def test_argmax_abstains_when_uncertain():
    cls, p = argmax_decision(np.ones(5), commit_threshold=0.5)
    assert cls is None and abs(p - 0.2) < 1e-9


def test_argmax_threshold_boundary():
    alpha = np.array([1.0, 1.0, 9.0, 1.0, 1.0])  # p_max = 9/13
    pmax = 9.0 / 13.0
    assert argmax_decision(alpha, pmax)[0] == 2          # >= commits
    assert argmax_decision(alpha, pmax + 0.01)[0] is None  # just above abstains


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.1])
def test_argmax_bad_threshold_raises(bad):
    with pytest.raises(ValueError, match="commit_threshold"):
        argmax_decision(np.ones(5), bad)
