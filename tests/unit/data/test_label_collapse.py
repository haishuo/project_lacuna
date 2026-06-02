"""Tests for lacuna.data.label_collapse — answer-sheet -> composition + collapsed dataset label."""

import numpy as np
import pytest

from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.mixed_missingness import OBSERVED
from lacuna.data.label_collapse import composition_vector, collapse_label


# --- composition_vector --------------------------------------------------------------------------

def test_composition_by_missing_cell_weight():
    # 1 MCAR col @0.1, 1 MAR col @0.2, 1 MNAR col @0.5, 1 observed; n=100
    # missing cells: MCAR 10, MAR 20, MNAR 50 -> total 80 -> [0.125, 0.25, 0.625]
    comp = composition_vector((MCAR, MAR, MNAR, OBSERVED), (0.1, 0.2, 0.5, 0.0), n_rows=100)
    assert np.allclose(comp, [10 / 80, 20 / 80, 50 / 80])
    assert abs(comp.sum() - 1.0) < 1e-9


def test_observed_columns_contribute_nothing():
    comp = composition_vector((OBSERVED, MNAR), (0.0, 0.3), n_rows=50)
    assert np.allclose(comp, [0.0, 0.0, 1.0])


def test_composition_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        composition_vector((MCAR, MAR), (0.1,), n_rows=10)


def test_composition_zero_missing_raises():
    with pytest.raises(ValueError, match="no missing cells"):
        composition_vector((OBSERVED, OBSERVED), (0.0, 0.0), n_rows=10)


def test_composition_bad_nrows_raises():
    with pytest.raises(ValueError, match="n_rows must be"):
        composition_vector((MNAR,), (0.3,), n_rows=0)


# --- collapse_label ------------------------------------------------------------------------------

def test_dominant_is_argmax():
    assert collapse_label(np.array([0.1, 0.3, 0.6]), "dominant") == MNAR
    assert collapse_label(np.array([0.6, 0.3, 0.1]), "dominant") == MCAR


def test_primary_returns_intended_mechanism_regardless_of_composition():
    # composition is MNAR-dominant, but primary says MAR -> label MAR
    assert collapse_label(np.array([0.1, 0.2, 0.7]), "primary", primary=MAR) == MAR


def test_thresholded_flags_mnar_above_threshold():
    assert collapse_label(np.array([0.4, 0.3, 0.3]), "thresholded", mnar_threshold=0.25) == MNAR
    # below threshold -> argmax over {MCAR, MAR}
    assert collapse_label(np.array([0.5, 0.4, 0.1]), "thresholded", mnar_threshold=0.25) == MCAR
    assert collapse_label(np.array([0.4, 0.5, 0.1]), "thresholded", mnar_threshold=0.25) == MAR


def test_thresholded_boundary_is_inclusive():
    assert collapse_label(np.array([0.3, 0.3, 0.4]), "thresholded", mnar_threshold=0.4) == MNAR


# --- failures ------------------------------------------------------------------------------------

def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown collapse rule"):
        collapse_label(np.array([0.3, 0.3, 0.4]), "average")


def test_primary_without_primary_raises():
    with pytest.raises(ValueError, match="requires primary"):
        collapse_label(np.array([0.3, 0.3, 0.4]), "primary")


def test_bad_threshold_raises():
    with pytest.raises(ValueError, match="mnar_threshold"):
        collapse_label(np.array([0.3, 0.3, 0.4]), "thresholded", mnar_threshold=1.5)


def test_bad_composition_raises():
    with pytest.raises(ValueError, match="finite non-negative"):
        collapse_label(np.array([0.5, 0.5]), "dominant")
