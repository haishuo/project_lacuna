"""Tests for lacuna.feasibility.manifest (validity gate)."""

import json

import pytest

from lacuna.feasibility.manifest import (
    REQUIRED_FIELDS,
    build_manifest,
    validate_manifest,
    write_manifest,
)


def _oracle_manifest(**over):
    base = dict(
        run_id="r1",
        git_commit="abc123",
        timestamp="2026-06-02T00:00:00",
        arm="oracle",
        kind="main",
        grid={"delta": [0.0, 1.0], "beta1": [0.0], "target_rate": [0.3], "rho": [0.0], "n": [256]},
        xmodel={"type": "ConditionalGaussian", "exact_synthetic": True},
        wall_clock_seconds=12.3,
        metrics={"min_bayes_error": 0.41},
    )
    base.update(over)
    return build_manifest(**base)


def test_oracle_manifest_valid_with_nullable_model_fields():
    m = _oracle_manifest()
    validate_manifest(m)  # should not raise
    assert m["all_layers_trainable"] is None
    assert m["trainable_param_count"] is None
    assert set(REQUIRED_FIELDS).issubset(m.keys())


def test_missing_field_raises():
    m = _oracle_manifest()
    del m["metrics"]
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_null_required_nonnullable_raises():
    m = _oracle_manifest()
    m["wall_clock_seconds"] = None
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_main_with_checkpoint_loaded_forbidden():
    m = _oracle_manifest(checkpoint_loaded=True)
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_main_model_must_train_all_layers():
    m = build_manifest(
        run_id="m1", git_commit="abc", timestamp="t", arm="model", kind="main",
        grid={}, xmodel={}, wall_clock_seconds=100.0, metrics={"auc": 0.8},
        checkpoint_loaded=False, all_layers_trainable=False, trainable_param_count=900000,
        split_scheme="held-out-dataset", calibration={"ece": 0.05},
    )
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_model_main_valid():
    m = build_manifest(
        run_id="m2", git_commit="abc", timestamp="t", arm="model", kind="main",
        grid={}, xmodel={}, wall_clock_seconds=100.0, metrics={"auc": 0.8},
        checkpoint_loaded=False, all_layers_trainable=True, trainable_param_count=900000,
        split_scheme="held-out-dataset", calibration={"ece": 0.05},
    )
    validate_manifest(m)


def test_invalid_arm_or_kind_raises():
    with pytest.raises(ValueError):
        _oracle_manifest(arm="bogus")
    with pytest.raises(ValueError):
        _oracle_manifest(kind="bogus")


def test_write_round_trip(tmp_path):
    m = _oracle_manifest()
    p = write_manifest(tmp_path / "manifest.json", m)
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["run_id"] == "r1"
    assert loaded["arm"] == "oracle"
