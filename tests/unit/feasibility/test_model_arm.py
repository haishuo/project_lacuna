"""Tests for lacuna.feasibility.model_arm (binary H0-vs-H1 model arm on full Lacuna)."""

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR, MNAR
from lacuna.feasibility.delta_generator import SelfCensorParams
from lacuna.feasibility.model_arm import (
    binary_loss,
    binary_pred,
    binary_q,
    evaluate,
    lr_at,
    regime_pool,
    sample_observed_fixed,
    train_regime,
    _ece,
)


def test_lr_schedule_warmup_then_cosine():
    base, lr_min, warm, total = 1e-4, 1e-5, 300, 3000
    assert lr_at(0, base, warm, total, lr_min) < base            # warmup starts low
    assert abs(lr_at(warm - 1, base, warm, total, lr_min) - base) < 1e-9  # peaks at base end of warmup
    # cosine: monotone non-increasing after warmup, ending near lr_min
    after = [lr_at(s, base, warm, total, lr_min) for s in range(warm, total + 1, 100)]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:]))
    assert abs(after[-1] - lr_min) < 1e-6
    # legacy constant: warmup 0, lr_min==base
    assert lr_at(500, base, 0, total, base) == base


# ---------- constraint #4: MCAR must not affect the binary objective ----------

def test_binary_q_and_pred_independent_of_mcar_logit():
    """Varying the MCAR logit (fixed MAR/MNAR logits) leaves q and the prediction unchanged."""
    z_mar, z_mnar = 0.7, -0.3
    qs, preds = [], []
    for z_mcar in [-5.0, -1.0, 0.0, 2.0, 8.0]:
        # float64 to prove the EXACT mathematical cancellation (formula independent of MCAR)
        logits = torch.tensor([[z_mcar, z_mar, z_mnar]], dtype=torch.float64)  # MCAR, MAR, MNAR
        p = torch.softmax(logits, dim=1)
        qs.append(binary_q(p).item())
        preds.append(binary_pred(p).item())
    assert max(qs) - min(qs) < 1e-10         # q exactly invariant to MCAR (float64)
    assert len(set(preds)) == 1              # prediction invariant to MCAR
    # and q matches the closed-form 2-way softmax of (MAR, MNAR)
    expected = torch.softmax(torch.tensor([z_mar, z_mnar], dtype=torch.float64), 0)[1].item()
    assert abs(qs[0] - expected) < 1e-10


def test_binary_loss_and_pred_basic():
    # confident-correct MNAR
    p = torch.tensor([[0.01, 0.01, 0.98]])
    y = torch.tensor([1])
    assert binary_pred(p).item() == 1
    assert binary_loss(p, y).item() < 0.1
    # confident-wrong
    assert binary_loss(p, torch.tensor([0])).item() > 1.0


# ---------- data generation ----------

def test_sample_observed_only_target_censored():
    h = SelfCensorParams(beta0=-1.0, beta1=1.0, beta2=1.0)
    x, r, rate = sample_observed_fixed(0.5, h, n=4000, rng=RNGState(seed=0))
    assert x.shape == (4000, 2) and r.shape == (4000, 2)
    assert r[:, 0].all()             # predictor always observed
    assert not r[:, 1].all()         # target censored
    assert 0.0 < rate < 1.0


def test_regime_pool_balanced_and_rate_matched_when_h0_equals_h1():
    h = SelfCensorParams(beta0=-1.0, beta1=1.0, beta2=0.0)
    ds, labels, stats = regime_pool(0.6, h, h, n=1000, n_datasets=40, rng=RNGState(seed=1))
    assert len(ds) == 40
    assert labels.count(0) == labels.count(1) == 20      # balanced
    # identical mechanisms ⇒ class rates statistically equal
    assert abs(stats["rate_h0_mean"] - stats["rate_h1_mean"]) < 0.03


def test_ece_in_range():
    q = torch.rand(500)
    y = (torch.rand(500) < q).long()
    e = _ece(q, y)
    assert 0.0 <= e <= 1.0


# ---------- tiny end-to-end (CPU, structure only; not an accuracy claim) ----------

def test_train_regime_runs_and_reports(tmp_path):
    from lacuna.models import create_lacuna_model

    def factory():
        return create_lacuna_model(hidden_dim=32, evidence_dim=16, n_layers=1, n_heads=2,
                                   max_cols=2, dropout=0.0, mnar_variants=["self_censoring"])

    h1 = SelfCensorParams(beta0=-1.0, beta1=1.0, beta2=1.5)
    h0 = SelfCensorParams(beta0=-1.0, beta1=1.4, beta2=0.0)
    m = train_regime(
        "tiny", rho=0.5, delta=1.5, n=32, h1=h1, h0=h0, ceiling=0.3, ceiling_se=0.01,
        model_factory=factory, rng=RNGState(seed=2), device="cpu",
        n_train=16, n_val=8, n_test=16, batch_size=8, max_epochs=2, patience=1,
    )
    for k in ["model_error", "gap", "gap_se", "ceiling", "trainable_param_count",
              "checkpoint_loaded", "suspicious_negative_gap", "rate_h0_mean", "rate_h1_mean", "ece"]:
        assert k in m
    assert m["checkpoint_loaded"] is False
    assert m["trainable_param_count"] > 0
    assert 0.0 <= m["model_error"] <= 1.0
