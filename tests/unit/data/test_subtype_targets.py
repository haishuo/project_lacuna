"""Tests for lacuna.data.subtype_targets — MixedBatch -> per-column subtype targets."""

import torch
import pytest

from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.mixed_batch import MixedBatch
from lacuna.data.subtype_targets import subtype_targets
from lacuna.priors.subtype_ontology import (
    THRESHOLD_MNAR, DETECTION_MNAR, SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB,
    LIKE_THRESHOLD, LIKE_DETECTION, LIKE_INDETERMINATE,
)


def _mb(labels, sup, mnar_subtypes, mar_subtypes):
    """Construct a minimal MixedBatch (only the fields subtype_targets reads are populated)."""
    return MixedBatch(
        batch=None, labels=torch.tensor(labels, dtype=torch.long),
        supervision_mask=torch.tensor(sup, dtype=torch.bool),
        compositions=(), complete_values=None,
        mnar_subtypes=mnar_subtypes, mar_subtypes=mar_subtypes,
    )


def test_targets_map_mechanism_and_subtype():
    mb = _mb(
        labels=[[MNAR, MNAR, MCAR, 0], [MAR, MNAR, 0, 0]],
        sup=[[True, True, True, False], [True, True, False, False]],
        mnar_subtypes=({0: "threshold_left", 1: "detection_lower"}, {1: "self_censoring"}),
        mar_subtypes=({}, {0: "mar_probit"}),
    )
    like, onto, sup = subtype_targets(mb)

    # item 0: loud threshold, loud detection, MCAR(reject/mcar)
    assert like[0, 0] == LIKE_THRESHOLD and onto[0, 0] == THRESHOLD_MNAR
    assert like[0, 1] == LIKE_DETECTION and onto[0, 1] == DETECTION_MNAR
    assert like[0, 2] == LIKE_INDETERMINATE and onto[0, 2] == MCAR_SUB
    # item 1: MAR(reject/mar), quiet MNAR(reject/self_censoring)
    assert like[1, 0] == LIKE_INDETERMINATE and onto[1, 0] == MAR_SUB
    assert like[1, 1] == LIKE_INDETERMINATE and onto[1, 1] == SELF_CENSORING_MNAR
    # supervision echoed; unsupervised positions are zero placeholders
    assert sup.tolist() == [[True, True, True, False], [True, True, False, False]]
    assert like[0, 3] == 0 and onto[1, 3] == 0


def test_all_observed_item_has_no_supervision():
    mb = _mb(labels=[[0, 0]], sup=[[False, False]], mnar_subtypes=({},), mar_subtypes=({},))
    like, onto, sup = subtype_targets(mb)
    assert not sup.any() and like.sum() == 0 and onto.sum() == 0


def test_missing_mnar_subtype_raises():
    mb = _mb(labels=[[MNAR]], sup=[[True]], mnar_subtypes=({},), mar_subtypes=({},))
    with pytest.raises(ValueError, match="no recorded subtype"):
        subtype_targets(mb)


def test_missing_mar_subtype_raises():
    mb = _mb(labels=[[MAR]], sup=[[True]], mnar_subtypes=({},), mar_subtypes=({},))
    with pytest.raises(ValueError, match="no recorded subtype"):
        subtype_targets(mb)
