"""Tests for lacuna.survey.conditioned_head — head-side target-conditioned δ-model (rung 3).

Covers the required rung-3 pre-run checks: target index flows out-of-band; pooled target shape;
build + backprop; no v1.0 heads; tokenization unchanged (4 channels); global vs conditioned
distinguishable.
"""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.tokenization import TOKEN_DIM
from lacuna.survey.batching import collate, make_example
from lacuna.survey.conditioned_head import (
    CONDITIONING_METHOD,
    TargetConditionedDeltaModel,
    create_target_conditioned_model,
)
from lacuna.survey.delta_bins import NUM_BINS


def _model(seed=1):
    return create_target_conditioned_model(
        hidden_dim=64, evidence_dim=32, n_layers=2, n_heads=2, max_cols=32,
        num_bins=NUM_BINS, rng=RNGState(seed=seed),
    )


def _batch(seed=7, deltas=(0.0, 0.5, 1.5, 2.5), max_rows=96):
    raw = create_default_catalog().load("survey_bfi")
    rng = RNGState(seed=seed)
    exs = [make_example(raw, beta1=1.0, delta=d, target_rate=0.25, rng=rng.spawn(),
                        max_rows=max_rows) for d in deltas]
    return collate(exs, max_rows=max_rows, max_cols=32)


# 1. target index flows through the batch out-of-band
def test_target_idx_in_batch():
    db = _batch()
    assert db.target_idx.shape == (4,)
    assert db.target_idx.dtype == torch.long
    # equals the answer sheets' target column, and 3-class fields stay unused
    assert db.target_idx.tolist() == [s.target_col_idx for s in db.sheets]
    assert db.tokens.generator_ids is None and db.tokens.class_ids is None


# 2. pooled target representation shape is correct
def test_pooled_target_shape():
    model = _model()
    db = _batch()
    enc = model.encoder(db.tokens.tokens, db.tokens.row_mask, db.tokens.col_mask,
                        return_intermediates=True)
    pooled = model._pool_target(enc["token_representations"], db.tokens.row_mask, db.target_idx)
    assert pooled.shape == (4, model.encoder.config.hidden_dim)
    assert torch.isfinite(pooled).all()


# 4. builds and backprops, returns bin logits
def test_forward_and_backprop():
    model = _model()
    model.train()
    db = _batch()
    logits = model(db.tokens, db.target_idx)
    assert logits.shape == (4, NUM_BINS)
    loss = torch.nn.functional.cross_entropy(logits, db.delta_bin)
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert gnorm > 0 and torch.isfinite(torch.tensor(gnorm))


# 6. no old v1.0 heads enter the graph
def test_no_v1_heads():
    model = _model()
    for h in ("moe", "reconstruction", "decision_rule", "missingness_extractor"):
        assert not hasattr(model, h)
    assert {n for n, _ in model.named_children()} == {"encoder", "head"}


# 5. tokenization unchanged (still 4 channels)
def test_tokenization_unchanged():
    assert TOKEN_DIM == 4
    db = _batch()
    assert db.tokens.tokens.shape[-1] == 4


# 3. global-only vs target-conditioned are distinguishable (head input dim differs)
def test_conditioned_head_wider_than_global():
    model = _model()
    # head input = evidence_dim + hidden_dim (conditioning concatenation)
    first_linear = model.head.net[0]
    assert first_linear.in_features == model.encoder.config.evidence_dim + model.encoder.config.hidden_dim
    assert CONDITIONING_METHOD == "head_side_target_token_pool"


# determinism + interface
def test_init_determinism():
    a, b = _model(seed=5), _model(seed=5)
    a.eval(); b.eval()
    db = _batch()
    assert torch.allclose(a(db.tokens, db.target_idx), b(db.tokens, db.target_idx), atol=1e-6)


def test_predict_proba_normalized():
    model = _model()
    model.eval()
    db = _batch()
    p = model.predict_proba(db.tokens, db.target_idx)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)


# failure cases
def test_missing_target_idx_raises():
    model = _model()
    db = _batch()
    with pytest.raises(ValueError):
        model(db.tokens, None)


def test_out_of_range_target_idx_raises():
    model = _model()
    db = _batch()
    bad = db.target_idx.clone()
    bad[0] = 999
    with pytest.raises(ValueError):
        model(db.tokens, bad)


def test_set_temperature_rejects_nonpositive():
    model = _model()
    with pytest.raises(ValueError):
        model.set_temperature(-1.0)


# ---- rep-ECDF distributional stream (Proposal B): enabled path ----

def _ecdf_model(seed=1, n_probes=4):
    return create_target_conditioned_model(
        hidden_dim=64, evidence_dim=32, n_layers=2, n_heads=2, max_cols=32,
        num_bins=NUM_BINS, rep_ecdf_pooling=True, n_shape_probes=n_probes,
        rng=RNGState(seed=seed),
    )


def test_default_path_unchanged_no_stream_child():
    """Disabled (default) ⇒ mean pool, no rep_pool child, head width = evidence + hidden."""
    model = _model()
    assert not model.rep_ecdf_pooling
    assert not hasattr(model, "rep_pool")
    assert {n for n, _ in model.named_children()} == {"encoder", "head"}
    assert model.head.net[0].in_features == (
        model.encoder.config.evidence_dim + model.encoder.config.hidden_dim
    )


def test_ecdf_head_width_and_child():
    """Enabled ⇒ head input = evidence + rep_pool.out_dim; rep_pool is a child module."""
    model = _ecdf_model(n_probes=4)
    assert model.rep_ecdf_pooling and hasattr(model, "rep_pool")
    assert "rep_pool" in {n for n, _ in model.named_children()}
    assert model.head.net[0].in_features == (
        model.encoder.config.evidence_dim + model.rep_pool.out_dim
    )


def test_ecdf_forward_and_backprop():
    model = _ecdf_model()
    model.train()
    db = _batch()
    logits = model(db.tokens, db.target_idx)
    assert logits.shape == (4, NUM_BINS)
    loss = torch.nn.functional.cross_entropy(logits, db.delta_bin)
    loss.backward()
    # gradient reaches the encoder THROUGH the quantile pooling (the whole point of the stream)
    enc_grad = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
    proj_grad = model.rep_pool.proj.weight.grad
    assert enc_grad > 0
    assert proj_grad is not None and float(proj_grad.abs().sum()) > 0


def test_ecdf_no_v1_heads():
    model = _ecdf_model()
    for h in ("moe", "reconstruction", "decision_rule", "missingness_extractor"):
        assert not hasattr(model, h)
    assert {n for n, _ in model.named_children()} == {"encoder", "head", "rep_pool"}
