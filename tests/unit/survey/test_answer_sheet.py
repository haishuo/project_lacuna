"""Tests for lacuna.survey.answer_sheet."""

import pytest

from lacuna.survey.answer_sheet import (
    GENERATOR_FAMILY,
    SCHEMA_VERSION,
    AnswerSheet,
)


def _sheet() -> AnswerSheet:
    return AnswerSheet(
        source_name="survey_demo",
        n=500,
        d=8,
        target_col_idx=3,
        target_col_name="income",
        predictor_col_idx=1,
        predictor_col_name="age",
        beta0=-1.234,
        beta1=0.5,
        delta=0.8,
        delta_bin=3,
        generator_family=GENERATOR_FAMILY,
        target_rate=0.3,
        realized_rate=0.31,
        corr_target_predictor=0.42,
        seed=99,
    )


# ---------- normal cases ----------

def test_to_dict_round_trip():
    s = _sheet()
    restored = AnswerSheet.from_dict(s.to_dict())
    assert restored == s


def test_to_dict_carries_schema_version():
    assert _sheet().to_dict()["schema_version"] == SCHEMA_VERSION


# ---------- failure cases ----------

def test_from_dict_rejects_non_dict():
    with pytest.raises(ValueError):
        AnswerSheet.from_dict([1, 2, 3])


def test_from_dict_rejects_schema_mismatch():
    payload = _sheet().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError):
        AnswerSheet.from_dict(payload)


def test_from_dict_rejects_missing_field():
    payload = _sheet().to_dict()
    del payload["delta"]
    with pytest.raises(ValueError):
        AnswerSheet.from_dict(payload)


def test_from_dict_rejects_unknown_family():
    payload = _sheet().to_dict()
    payload["generator_family"] = "skip_logic"
    with pytest.raises(ValueError):
        AnswerSheet.from_dict(payload)
