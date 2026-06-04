"""Tests for lacuna.survey.run_manifest."""

import json

import pytest

from lacuna.survey.run_manifest import (
    REQUIRED_FIELDS,
    build_manifest,
    validate_manifest,
    write_manifest,
)


def _manifest(**overrides):
    base = dict(
        run_id="p2p2-smoke", git_commit="deadbeef", timestamp="2026-06-04T00:00:00Z",
        kind="main",
        delta_grid=[0.0, 0.4, 1.0, 2.5], beta1_range=[0.0, 2.0], target_rate=0.3, seed=1,
        model_arch={"hidden_dim": 64}, trainable_param_count=112585,
        checkpoint_loaded=False, all_layers_trainable=True, temperature=1.2,
        split_scheme={"scheme": "leave-datasets-out"},
        metrics={"best_val_rps": 0.1}, calibration={"ece_after": 0.05},
        leakage={"delta_rate_pearson": 0.01}, leakage_pass=True, wall_clock_seconds=12.3,
    )
    base.update(overrides)
    return build_manifest(**base)


# ---------- normal ----------

def test_build_validate_write(tmp_path):
    m = _manifest()
    validate_manifest(m)
    assert set(REQUIRED_FIELDS).issubset(m.keys())
    assert m["loss"] == "RPS"
    path = write_manifest(tmp_path / "run.json", m)
    validate_manifest(json.loads(path.read_text()))


# ---------- main-run guarantees ----------

def test_main_rejects_checkpoint_loaded():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(checkpoint_loaded=True))


def test_main_rejects_frozen_encoder():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(all_layers_trainable=False))


def test_main_rejects_leakage_fail():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(leakage_pass=False))


def test_ablation_allows_leakage_fail():
    # a non-main run may record leakage_pass=False without being rejected
    validate_manifest(_manifest(kind="ablation", leakage_pass=False))


# ---------- generic failures ----------

def test_rejects_missing_field():
    m = _manifest()
    del m["metrics"]
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_rejects_null_field():
    m = _manifest()
    m["temperature"] = None
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_rejects_grid_without_mar():
    with pytest.raises(ValueError):
        validate_manifest(_manifest(delta_grid=[0.4, 1.0]))


def test_rejects_bad_kind():
    with pytest.raises(ValueError):
        _manifest(kind="production")


def test_rejects_loss_tamper():
    m = _manifest()
    m["loss"] = "cross_entropy"
    with pytest.raises(ValueError):
        validate_manifest(m)
