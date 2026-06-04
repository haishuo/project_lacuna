"""Tests for lacuna.survey.example_source — synthetic 2-col (rung 1) and survey sources."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.answer_sheet import GENERATOR_FAMILY
from lacuna.survey.delta_bins import NUM_BINS, assign_delta_bin
from lacuna.survey.example_source import (
    StratifiedRealXSource,
    SurveyExampleSource,
    SyntheticTwoColSource,
)
from lacuna.survey.leakage import assess_leakage, leakage_pass


class _Cfg:
    """Minimal stand-in for TrainConfig fields the sources read."""
    target_rate = 0.3
    max_rows = 800


# ---------- synthetic 2-col (rung 1) ----------

def test_synthetic_shapes_and_known_target():
    src = SyntheticTwoColSource(rho_grid=[0.0, 0.5])
    ex = src.make_one(_Cfg(), RNGState(seed=1), delta=0.75, beta1=1.0)
    assert ex.observed.n == 800 and ex.observed.d == 2
    s = ex.answer_sheet
    assert s.target_col_idx == 1 and s.predictor_col_idx == 0
    assert s.target_col_name == "target" and s.predictor_col_name == "predictor"
    assert s.generator_family == GENERATOR_FAMILY
    assert s.delta == 0.75 and s.delta_bin == assign_delta_bin(0.75)
    # only the target column (col 1) may be censored
    assert ex.observed.r[:, 0].all()
    assert not ex.observed.r[:, 1].all()


def test_synthetic_corr_matches_rho():
    src = SyntheticTwoColSource(rho_grid=[0.7])
    ex = src.make_one(_Cfg(), RNGState(seed=2), delta=1.0, beta1=0.5)
    assert ex.answer_sheet.corr_target_predictor == pytest.approx(0.7)


def test_synthetic_rate_matched_near_target():
    src = SyntheticTwoColSource(rho_grid=[0.3])
    ex = src.make_one(_Cfg(), RNGState(seed=3), delta=2.0, beta1=1.0)
    assert abs(ex.answer_sheet.realized_rate - 0.3) < 0.05


def test_synthetic_determinism():
    src = SyntheticTwoColSource(rho_grid=[0.0, 0.5, 0.9])
    a = src.make_one(_Cfg(), RNGState(seed=9), delta=1.25, beta1=1.0)
    b = src.make_one(_Cfg(), RNGState(seed=9), delta=1.25, beta1=1.0)
    assert torch.equal(a.observed.r, b.observed.r)
    assert a.answer_sheet == b.answer_sheet


def test_synthetic_num_bins_default():
    assert SyntheticTwoColSource(rho_grid=[0.0]).num_bins == NUM_BINS


def test_synthetic_describe():
    d = SyntheticTwoColSource(rho_grid=[0.0, 0.5]).describe()
    assert d["x_source"] == "synthetic_2col" and d["rho_grid"] == [0.0, 0.5]


def test_synthetic_no_rate_leakage_across_delta():
    """δ=0 (MAR) vs δ>0 realized rates match — the matched-rate solve holds on synthetic X too."""
    src = SyntheticTwoColSource(rho_grid=[0.0, 0.5])
    cfg = _Cfg()
    grid = [0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5]
    rng = RNGState(seed=5)
    sheets = []
    for i in range(140):
        d = grid[i % len(grid)]
        sheets.append(src.make_one(cfg, rng.spawn(), delta=d, beta1=1.0).answer_sheet)
    report = assess_leakage(sheets, RNGState(seed=6))
    assert abs(report.delta_rate_pearson) < 0.2
    assert leakage_pass(report)


# ---------- survey source (preserves P2.2 path) ----------

def test_survey_source_make_one():
    raw = create_default_catalog().load("survey_bfi")
    src = SurveyExampleSource([raw])

    class C:
        target_rate = 0.25
        max_rows = 128

    ex = src.make_one(C(), RNGState(seed=1), delta=1.0, beta1=1.0)
    assert ex.observed.n == 128
    assert ex.answer_sheet.source_name == "survey_bfi"


def test_survey_describe():
    raw = create_default_catalog().load("survey_bfi")
    assert SurveyExampleSource([raw]).describe()["x_source"] == "real_survey"


# ---------- stratified real-X source (Part B) ----------

def test_stratified_pins_target_and_balances():
    raw = create_default_catalog().load("survey_bfi")
    dbn = {"survey_bfi": raw}
    strata = [[("survey_bfi", 0), ("survey_bfi", 1)], [("survey_bfi", 5)]]
    src = StratifiedRealXSource(dbn, strata)

    class C:
        target_rate = 0.25
        max_rows = 128

    seen = set()
    rng = RNGState(seed=3)
    for _ in range(20):
        ex = src.make_one(C(), rng.spawn(), delta=1.0, beta1=1.0)
        seen.add(ex.answer_sheet.target_col_idx)
    assert seen.issubset({0, 1, 5})  # only pinned targets used
    assert src.describe()["x_source"] == "real_survey_r2_stratified"


def test_stratified_rejects_empty():
    raw = create_default_catalog().load("survey_bfi")
    with pytest.raises(ValueError):
        StratifiedRealXSource({"survey_bfi": raw}, [[]])


def test_stratified_rejects_unknown_dataset():
    raw = create_default_catalog().load("survey_bfi")
    with pytest.raises(ValueError):
        StratifiedRealXSource({"survey_bfi": raw}, [[("nope", 0)]])


# ---------- failure cases ----------

def test_synthetic_rejects_empty_grid():
    with pytest.raises(ValueError):
        SyntheticTwoColSource(rho_grid=[])


def test_synthetic_rejects_bad_rho():
    with pytest.raises(ValueError):
        SyntheticTwoColSource(rho_grid=[1.0])


def test_survey_rejects_empty_pool():
    with pytest.raises(ValueError):
        SurveyExampleSource([])
