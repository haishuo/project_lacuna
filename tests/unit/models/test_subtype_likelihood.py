"""Tests for lacuna.models.subtype_likelihood — the deployable-feature subtype detector."""

import numpy as np
import pytest

from lacuna.models.column_deployable_features import N_DEPLOYABLE_FEATURES
from lacuna.priors.subtype_ontology import N_LIKELIHOOD_LABELS
from lacuna.models.subtype_likelihood import SubtypeLikelihoodDetector


def _synthetic(n_per, rng):
    """Three separable classes: class 0 spikes feature 1, class 1 spikes feature 2, class 2 is flat."""
    X, y = [], []
    for cls, spike in [(0, 1), (1, 2), (2, None)]:
        f = rng.normal(0.0, 0.3, size=(n_per, N_DEPLOYABLE_FEATURES))
        if spike is not None:
            f[:, spike] += 3.0
        X.append(f); y += [cls] * n_per
    X = np.concatenate(X); y = np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_fits_and_separates_clean_classes():
    rng = np.random.default_rng(0)
    Xtr, ytr = _synthetic(300, rng)
    Xte, yte = _synthetic(150, rng)
    det = SubtypeLikelihoodDetector(seed=0).fit(Xtr, ytr)
    proba = det.predict_proba(Xte)
    assert proba.shape == (len(yte), N_LIKELIHOOD_LABELS)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-6)
    acc = (proba.argmax(1) == yte).mean()
    assert acc > 0.85, f"separable classes should be learnable, got acc {acc:.3f}"


def test_deterministic_given_seed():
    rng = np.random.default_rng(1)
    Xtr, ytr = _synthetic(200, rng)
    Xte, _ = _synthetic(100, rng)
    a = SubtypeLikelihoodDetector(seed=7).fit(Xtr, ytr).predict_proba(Xte)
    b = SubtypeLikelihoodDetector(seed=7).fit(Xtr, ytr).predict_proba(Xte)
    assert np.array_equal(a, b)


def test_absent_class_column_is_zero():
    rng = np.random.default_rng(2)
    X, y = _synthetic(200, rng)
    keep = y != 1                       # train with only classes {0, 2}
    det = SubtypeLikelihoodDetector(seed=0).fit(X[keep], y[keep])
    proba = det.predict_proba(X[:50])
    assert np.allclose(proba[:, 1], 0.0)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-6)


# --- failure cases -----------------------------------------------------------------------------

def test_custom_n_features_subset():
    """The detector can take a feature SUBSET (the subtype layer drops missing_rate -> 4 features)."""
    rng = np.random.default_rng(5)
    Xtr, ytr = _synthetic(200, rng)
    Xte, yte = _synthetic(100, rng)
    det = SubtypeLikelihoodDetector(seed=0, n_features=N_DEPLOYABLE_FEATURES - 1).fit(Xtr[:, 1:], ytr)
    proba = det.predict_proba(Xte[:, 1:])
    assert proba.shape == (len(yte), N_LIKELIHOOD_LABELS)
    with pytest.raises(ValueError, match=r"\[N, 4\]"):   # full 5-wide input is now the wrong shape
        det.predict_proba(Xte)


def test_bad_n_features_raises():
    with pytest.raises(ValueError, match="n_features must be in"):
        SubtypeLikelihoodDetector(seed=0, n_features=1)
    with pytest.raises(ValueError, match="n_features must be in"):
        SubtypeLikelihoodDetector(seed=0, n_features=N_DEPLOYABLE_FEATURES + 1)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="before fit"):
        SubtypeLikelihoodDetector(seed=0).predict_proba(np.zeros((3, N_DEPLOYABLE_FEATURES)))


def test_bad_feature_dim_raises():
    det = SubtypeLikelihoodDetector(seed=0)
    with pytest.raises(ValueError, match=r"\[N, 5\]"):
        det.fit(np.zeros((10, 3)), np.zeros(10, dtype=int))


def test_nonfinite_features_raise():
    rng = np.random.default_rng(3)
    X, y = _synthetic(50, rng)
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        SubtypeLikelihoodDetector(seed=0).fit(X, y)


def test_bad_labels_raise():
    rng = np.random.default_rng(4)
    X, y = _synthetic(50, rng)
    y[0] = 5
    with pytest.raises(ValueError, match="labels must be in"):
        SubtypeLikelihoodDetector(seed=0).fit(X, y)


def test_single_class_raises():
    X = np.zeros((20, N_DEPLOYABLE_FEATURES))
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
        SubtypeLikelihoodDetector(seed=0).fit(X, np.zeros(20, dtype=int))


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError, match="labels must be"):
        SubtypeLikelihoodDetector(seed=0).fit(np.zeros((10, N_DEPLOYABLE_FEATURES)), np.zeros(8, dtype=int))
