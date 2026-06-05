"""Tests for the LOD example-source plumbing (LODSurveyExampleSource + make_lod_example)."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.answer_sheet import LOD_FAMILY
from lacuna.survey.batching import collate, make_lod_example
from lacuna.survey.example_source import LODSurveyExampleSource, SurveyExampleSource


class _Cfg:
    target_rate = 0.3
    max_rows = 128


def test_make_lod_example_subsamples_and_records_family():
    raw = create_default_catalog().load("survey_cps1985")
    ex = make_lod_example(raw, beta1=1.0, delta=2.0, target_rate=0.3, tau_quantile=0.70,
                          rng=RNGState(seed=1), max_rows=128)
    assert ex.observed.n == 128
    assert ex.answer_sheet.generator_family == LOD_FAMILY
    assert ex.answer_sheet.tau is not None


def test_lod_source_family_and_describe():
    raw = create_default_catalog().load("survey_cps1985")
    src = LODSurveyExampleSource([raw], tau_quantile=0.70)
    assert src.generator_family == LOD_FAMILY
    d = src.describe()
    assert d["idiom"] == "lod_top_coding" and d["tau_quantile"] == 0.70
    # own-value source still reports the own-value family
    assert SurveyExampleSource([raw]).generator_family != LOD_FAMILY


def test_lod_source_make_one_and_collate():
    raw = create_default_catalog().load("survey_cps1985")
    src = LODSurveyExampleSource([raw], tau_quantile=0.70)
    rng = RNGState(seed=2)
    exs = [src.make_one(_Cfg(), rng.spawn(), delta=d, beta1=1.0) for d in (0.0, 1.0, 2.5)]
    db = collate(exs, max_rows=128, max_cols=8)
    assert db.delta_bin.tolist() == [0, 3, 6]
    assert all(s.generator_family == LOD_FAMILY for s in db.sheets)


def test_lod_source_rejects_bad_inputs():
    raw = create_default_catalog().load("survey_cps1985")
    with pytest.raises(ValueError):
        LODSurveyExampleSource([], tau_quantile=0.70)
    with pytest.raises(ValueError):
        LODSurveyExampleSource([raw], tau_quantile=1.5)
