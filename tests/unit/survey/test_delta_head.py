"""Tests for lacuna.survey.delta_head — DeltaPriorModel + DeltaBinHead."""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.survey.batching import collate, make_example
from lacuna.survey.delta_bins import NUM_BINS
from lacuna.survey.delta_head import (
    DeltaBinHead,
    DeltaPriorModel,
    assert_fresh_and_trainable,
    count_parameters,
    create_delta_prior_model,
)


def _mini(rng_seed=1):
    return create_delta_prior_model(
        hidden_dim=64, evidence_dim=32, n_layers=2, n_heads=2, max_cols=32,
        num_bins=NUM_BINS, rng=RNGState(seed=rng_seed),
    )


def _batch(seed=7, deltas=(0.0, 0.5, 1.5, 2.5), max_rows=64):
    raw = create_default_catalog().load("survey_bfi")
    rng = RNGState(seed=seed)
    exs = [make_example(raw, beta1=1.0, delta=d, target_rate=0.25, rng=rng.spawn(),
                        max_rows=max_rows) for d in deltas]
    return collate(exs, max_rows=max_rows, max_cols=32)


# ---------- head ----------

def test_head_output_shape():
    head = DeltaBinHead(evidence_dim=32, num_bins=NUM_BINS)
    out = head(torch.randn(5, 32))
    assert out.shape == (5, NUM_BINS)


def test_head_rejects_bad_evidence_shape():
    head = DeltaBinHead(evidence_dim=32, num_bins=NUM_BINS)
    with pytest.raises(ValueError):
        head(torch.randn(5, 16))


def test_head_rejects_too_few_bins():
    with pytest.raises(ValueError):
        DeltaBinHead(evidence_dim=32, num_bins=1)


# ---------- model: no v1.0 heads present ----------

def test_model_has_no_v1_heads():
    model = _mini()
    assert not hasattr(model, "moe")
    assert not hasattr(model, "reconstruction")
    assert not hasattr(model, "decision_rule")
    assert not hasattr(model, "missingness_extractor")
    # only encoder + head are submodules
    child_names = {n for n, _ in model.named_children()}
    assert child_names == {"encoder", "head"}


def test_forward_returns_bin_logits():
    model = _mini()
    model.eval()
    batch = _batch()
    logits = model(batch.tokens)
    assert logits.shape == (4, NUM_BINS)
    assert torch.isfinite(logits).all()


def test_predict_proba_normalized():
    model = _mini()
    model.eval()
    probs = model.predict_proba(_batch().tokens)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


# ---------- from-scratch guarantees ----------

def test_all_layers_trainable():
    model = _mini()
    n = assert_fresh_and_trainable(model)
    assert n == count_parameters(model) > 0


def test_freezing_a_layer_trips_assert():
    model = _mini()
    next(model.encoder.parameters()).requires_grad_(False)
    with pytest.raises(ValueError):
        assert_fresh_and_trainable(model)


# ---------- determinism ----------

def test_init_determinism_same_seed_same_logits():
    a = _mini(rng_seed=5)
    b = _mini(rng_seed=5)
    a.eval(); b.eval()
    batch = _batch()
    assert torch.allclose(a(batch.tokens), b(batch.tokens), atol=1e-6)


def test_init_different_seed_differs():
    a = _mini(rng_seed=5)
    b = _mini(rng_seed=6)
    a.eval(); b.eval()
    batch = _batch()
    assert not torch.allclose(a(batch.tokens), b(batch.tokens), atol=1e-4)


def test_set_temperature_rejects_nonpositive():
    model = _mini()
    with pytest.raises(ValueError):
        model.set_temperature(0.0)
