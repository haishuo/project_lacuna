"""Fast smoke test for lacuna.survey.train — the full P2.2 pipeline on a tiny config."""

import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.run_manifest import validate_manifest
from lacuna.survey.train import TrainConfig, train_delta_prior


def _pools():
    cat = create_default_catalog()
    train = [cat.load("survey_bfi"), cat.load("survey_cps1985")]
    val = [cat.load("survey_psid1976")]
    test = [cat.load("survey_cars93")]
    return train, val, test


def _tiny_cfg():
    return TrainConfig(
        delta_grid=[0.0, 0.5, 1.0, 2.5], beta1_range=(0.0, 1.0), target_rate=0.3,
        max_rows=64, max_cols=32, batch_size=4, train_batches_per_epoch=3,
        max_epochs=2, patience=2, val_size=12, test_size=12,
        hidden_dim=32, evidence_dim=16, n_layers=1, n_heads=2,
    )


def test_train_smoke_runs_and_validates():
    train, val, test = _pools()
    out = train_delta_prior(
        train, val, test, _tiny_cfg(), RNGState(seed=0),
        kind="smoke", run_id="smoke-1", git_commit="abc", timestamp="2026-06-04T00:00:00Z",
    )
    # model built, manifest valid
    validate_manifest(out["manifest"])
    m = out["manifest"]
    assert m["trainable_param_count"] > 0
    assert m["checkpoint_loaded"] is False
    assert m["all_layers_trainable"] is True
    assert m["loss"] == "RPS"
    # metrics present and finite
    res = out["results"]
    assert torch.isfinite(torch.tensor(res["best_val_rps"]))
    assert "test_after_temperature" in res
    assert "rps" in res["test_after_temperature"]
    # leakage assessed
    assert "leakage" in m and "delta_rate_pearson" in m["leakage"]


def test_train_smoke_deterministic_data_and_structure():
    """The injected-RNG contract: construction + seeded DATA generation are bit-deterministic.

    Trained WEIGHTS are NOT asserted bit-equal — train-time dropout and nn.Embedding/scatter-add
    backward are torch-level non-determinisms (documented in train.py), outside the RNGState
    contract. We assert the parts that ARE contractual: param count, architecture, split scheme,
    and the seeded leakage corpus (pure data, no embedding backward).
    """
    train, val, test = _pools()
    cfg = _tiny_cfg()
    a = train_delta_prior(train, val, test, cfg, RNGState(seed=3),
                          kind="smoke", run_id="r", git_commit="c", timestamp="t")
    b = train_delta_prior(train, val, test, cfg, RNGState(seed=3),
                          kind="smoke", run_id="r", git_commit="c", timestamp="t")
    assert a["manifest"]["trainable_param_count"] == b["manifest"]["trainable_param_count"]
    assert a["manifest"]["model_arch"] == b["manifest"]["model_arch"]
    assert a["manifest"]["split_scheme"] == b["manifest"]["split_scheme"]
    # leakage is computed over the seeded val+test corpora -> bit-identical
    assert a["manifest"]["leakage"]["delta_rate_pearson"] == b["manifest"]["leakage"]["delta_rate_pearson"]
    assert a["manifest"]["leakage"]["per_bin"] == b["manifest"]["leakage"]["per_bin"]
