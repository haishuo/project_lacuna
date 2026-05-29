"""Tests for lacuna.models.column_recon_features — per-column reconstruction-error features."""

from pathlib import Path

import pytest
import torch

from lacuna.models.reconstruction.heads_container import create_reconstruction_heads
from lacuna.models.column_recon_features import (
    per_column_natural_error,
    per_column_recon_features,
    load_reconstruction_heads_from_baseline,
)

_BASELINE_CKPT = "/mnt/artifacts/project_lacuna/runs/stage0_general_baseline/checkpoints/best_model.pt"


def _heads(hidden_dim=16):
    rh = create_reconstruction_heads(hidden_dim=hidden_dim, head_hidden_dim=8,
                                     n_head_layers=2, dropout=0.0,
                                     mnar_variants=["self_censoring"])
    return rh.eval()


def _batch(B=2, R=6, C=4, H=16, seed=0, obs_col0=False):
    g = torch.Generator().manual_seed(seed)
    token_repr = torch.randn(B, R, C, H, generator=g)
    is_obs = (torch.rand(B, R, C, generator=g) > 0.4).float()
    if obs_col0:
        is_obs[:, :, 0] = 1.0  # column 0 fully observed -> no natural-missing cells
    value = torch.randn(B, R, C, generator=g) * is_obs
    feat_id = (torch.arange(C).float() / 47.0).view(1, 1, C).expand(B, R, C)
    tokens = torch.stack([value, is_obs, torch.zeros(B, R, C), feat_id], dim=-1)
    row_mask = torch.ones(B, R, dtype=torch.bool)
    col_mask = torch.ones(B, C, dtype=torch.bool)
    original_values = torch.randn(B, R, C, generator=g)
    return token_repr, tokens, row_mask, col_mask, original_values


def test_shape_n_heads():
    rh = _heads()
    feats = per_column_recon_features(rh, *_batch())
    assert feats.shape == (2, 4, rh.n_heads)  # n_heads = 2 + 1 = 3


def test_deterministic():
    rh = _heads()
    b = _batch(seed=3)
    a = per_column_recon_features(rh, *b)
    c = per_column_recon_features(rh, *b)
    assert torch.equal(a, c)


def test_fully_observed_column_zero_feature():
    rh = _heads()
    feats = per_column_recon_features(rh, *_batch(obs_col0=True))
    # column 0 has no naturally-missing cells -> error 0 -> log1p(0) = 0 for all heads
    assert torch.allclose(feats[:, 0, :], torch.zeros_like(feats[:, 0, :]), atol=1e-6)


def test_features_nonnegative():
    rh = _heads()
    feats = per_column_recon_features(rh, *_batch(seed=5))
    assert torch.all(feats >= 0.0)  # log1p of nonnegative squared errors


def test_raw_error_matches_manual():
    """Per-column raw error must equal a manual masked-mean for one head."""
    rh = _heads()
    tr, tokens, rm, cm, orig = _batch(seed=7)
    raw = per_column_natural_error(rh, tr, tokens, rm, cm, orig)  # [B,C,n_heads]
    # recompute head 0 ("mcar") manually
    results = rh.forward(tr, tokens, rm, cm, original_values=orig, compute_natural_errors=False)
    pred0 = rh.get_predictions_dict(results)["mcar"]
    is_obs = tokens[..., 1] > 0.5
    nm = (rm.unsqueeze(-1) & cm.unsqueeze(1) & ~is_obs).float()
    manual = (((pred0 - orig) ** 2) * nm).sum(1) / nm.sum(1).clamp(min=1.0)
    assert torch.allclose(raw[:, :, 0], manual, atol=1e-5)


@pytest.mark.skipif(not Path(_BASELINE_CKPT).exists(), reason="baseline checkpoint not present")
def test_loads_from_baseline():
    rh = load_reconstruction_heads_from_baseline(_BASELINE_CKPT, hidden_dim=128, device="cpu")
    assert rh.n_heads == 3
    # runs on a 128-dim batch
    feats = per_column_recon_features(rh, *_batch(H=128, seed=1))
    assert feats.shape == (2, 4, 3)
