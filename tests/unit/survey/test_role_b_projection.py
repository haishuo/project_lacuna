"""Tests for lacuna.survey.role_b_projection — role-A → role-B complete-case projection.

Covers sentinel recoding, complete-case projection, continuous-target filtering, the full
project_source provenance (incl. the binding projected_from_naturally_missing flag), and fail-loud
contracts. Pure NumPy; deterministic.
"""

import numpy as np
import pytest

from lacuna.survey.role_b_projection import (
    DEFAULT_SENTINELS,
    PROJECTION_FLAG,
    complete_case_project,
    continuous_targets,
    project_source,
    recode_sentinels,
    to_raw_dataset,
)


def test_recode_sentinels_to_nan():
    x = np.array([[1.0, 99.0], [2.0, 3.0], [77.0, 4.0]])
    out = recode_sentinels(x, [77, 99])
    assert np.isnan(out[0, 1]) and np.isnan(out[2, 0])
    assert out[1, 0] == 2.0 and out[1, 1] == 3.0


def test_recode_empty_sentinels_noop():
    x = np.array([[1.0, 2.0]])
    assert np.array_equal(recode_sentinels(x, []), x)


def test_complete_case_project_keeps_observed_cols_and_complete_rows():
    # col0 fully observed; col1 90% observed; col2 mostly missing
    x = np.array([[1.0, 1.0, np.nan], [2.0, 2.0, np.nan], [3.0, np.nan, np.nan],
                  [4.0, 4.0, np.nan], [5.0, 5.0, 5.0]])
    sub, names, rows = complete_case_project(x, ["a", "b", "c"], tau_col=0.7)
    assert names == ["a", "b"]            # c dropped (low observed)
    assert sub.shape[1] == 2
    assert 2 not in rows                  # row 2 (b missing) dropped
    assert not np.isnan(sub).any()


def test_complete_case_fail_loud():
    allmiss = np.full((5, 2), np.nan)
    with pytest.raises(ValueError):
        complete_case_project(allmiss, ["a", "b"], tau_col=0.95)
    with pytest.raises(ValueError):
        complete_case_project(np.zeros((3, 2)), ["a", "b"], tau_col=1.5)  # bad tau


def test_continuous_targets_by_cardinality():
    x = np.column_stack([np.arange(100.0), np.zeros(100), np.tile([0.0, 1.0], 50)])
    tgt = continuous_targets(x, min_card=30)
    assert tgt == [0]  # only the 100-distinct-value column


def test_project_source_full_provenance():
    rng = np.random.default_rng(0)
    cont = rng.normal(size=(200, 1)) * 10        # continuous target (high card)
    pred = rng.normal(size=(200, 1))
    sparse = np.full((200, 1), np.nan); sparse[:5] = 1.0
    x = np.hstack([cont, pred, sparse]).astype(float)
    x[0, 0] = 99.0  # a refusal sentinel in the target
    res = project_source(x, ["income", "age", "sparse"], source="ESS", name="ess_pooled",
                         domain="attitudes", source_block="ESS-R11", sentinels=[99], tau_col=0.9)
    p = res.provenance
    assert p["flag"] == PROJECTION_FLAG
    assert p["projection"] == "complete_case_from_naturally_missing"
    assert "sparse" not in res.feature_names      # dropped (low observed)
    assert p["sentinels_recoded"] == [99]
    assert p["n_continuous_targets"] >= 1
    assert p["survey_weights"] == "present_but_unused"
    assert res.data.dtype == np.float32
    assert len(res.targetable_idx) == p["n_continuous_targets"]


def test_project_source_no_continuous_target_fails():
    x = np.tile([0.0, 1.0], (100, 1))  # binary only → no continuous target
    with pytest.raises(ValueError):
        project_source(x, ["a", "b"], source="X", name="x", domain="d", source_block="b",
                       sentinels=[], tau_col=0.9, min_card=30)


def test_to_raw_dataset():
    rng = np.random.default_rng(1)
    x = np.hstack([rng.normal(size=(120, 1)) * 5, rng.normal(size=(120, 1))]).astype(float)
    res = project_source(x, ["t", "p"], source="NHANES", name="nhanes_demographics",
                         domain="demographics", source_block="NHANES-2017-18",
                         sentinels=DEFAULT_SENTINELS, tau_col=0.9, min_card=30)
    raw = to_raw_dataset(res)
    assert raw.name == "nhanes_demographics"
    assert raw.source == PROJECTION_FLAG
    assert raw.data.shape[1] == len(res.feature_names)
