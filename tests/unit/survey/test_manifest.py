"""Tests for lacuna.survey.manifest."""

import json

import pytest

from lacuna.survey.answer_sheet import GENERATOR_FAMILY, SCHEMA_VERSION
from lacuna.survey.manifest import (
    REQUIRED_FIELDS,
    build_manifest,
    validate_manifest,
    write_manifest,
)


def _manifest(**overrides) -> dict:
    base = dict(
        run_id="p2p1-smoke-001",
        git_commit="abc1234",
        timestamp="2026-06-04T00:00:00Z",
        kind="smoke",
        delta_grid=[0.0, 0.25, 0.5, 1.0, 2.0],
        beta1_range=[0.0, 2.0],
        target_rate=0.3,
        dataset_pool=["survey_bfi", "survey_cps1985"],
        seed=42,
        answer_sheets_saved=True,
    )
    base.update(overrides)
    return build_manifest(**base)


# ---------- normal cases ----------

def test_build_and_validate_ok():
    m = _manifest()
    validate_manifest(m)  # no raise
    assert set(REQUIRED_FIELDS).issubset(m.keys())
    assert m["generator_family"] == GENERATOR_FAMILY
    assert m["answer_sheet_schema_version"] == SCHEMA_VERSION


def test_write_round_trips_through_disk(tmp_path):
    m = _manifest()
    path = write_manifest(tmp_path / "manifest.json", m)
    loaded = json.loads(path.read_text())
    validate_manifest(loaded)
    assert loaded["dataset_pool"] == ["survey_bfi", "survey_cps1985"]


# ---------- failure cases ----------

def test_build_rejects_bad_kind():
    with pytest.raises(ValueError):
        _manifest(kind="production")


def test_validate_rejects_missing_field():
    m = _manifest()
    del m["delta_grid"]
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_validate_rejects_null_field():
    m = _manifest()
    m["seed"] = None
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_validate_requires_mar_in_grid():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(delta_grid=[0.25, 0.5, 1.0]))


def test_validate_rejects_negative_delta_in_grid():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(delta_grid=[0.0, -0.5]))


def test_validate_rejects_bad_beta1_range():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(beta1_range=[2.0, 1.0]))
    with pytest.raises(ValueError):
        validate_manifest(_manifest(beta1_range=[-1.0, 1.0]))


def test_validate_rejects_bad_target_rate():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(target_rate=1.0))


def test_validate_rejects_empty_dataset_pool():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(dataset_pool=[]))


def test_validate_rejects_family_tamper():
    m = _manifest()
    m["generator_family"] = "skip_logic"
    with pytest.raises(ValueError):
        validate_manifest(m)
