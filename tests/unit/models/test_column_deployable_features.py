"""Tests for lacuna.models.column_deployable_features — deployable per-column features."""

import torch

from lacuna.models.column_deployable_features import (
    per_column_deployable_features,
    N_DEPLOYABLE_FEATURES,
)


def _tokens(values, is_obs):
    """Build a [B,R,C,4] token tensor from values [B,R,C] and an observed mask [B,R,C] bool."""
    B, R, C = values.shape
    feat_id = (torch.arange(C).float() / 47.0).view(1, 1, C).expand(B, R, C)
    return torch.stack([values * is_obs.float(), is_obs.float(),
                        torch.zeros(B, R, C), feat_id], dim=-1)


def _rand(B=2, R=40, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(B, R, C, generator=g)
    is_obs = torch.rand(B, R, C, generator=g) > 0.3
    rm = torch.ones(B, R, dtype=torch.bool)
    cm = torch.ones(B, C, dtype=torch.bool)
    return _tokens(values, is_obs), rm, cm


def test_shape():
    t, rm, cm = _rand()
    f = per_column_deployable_features(t, rm, cm)
    assert f.shape == (2, 4, N_DEPLOYABLE_FEATURES)


def test_deterministic():
    t, rm, cm = _rand(seed=2)
    assert torch.equal(per_column_deployable_features(t, rm, cm),
                       per_column_deployable_features(t, rm, cm))


def test_scale_invariance():
    """All three features are scale-free → per-column scaling leaves them unchanged."""
    g = torch.Generator().manual_seed(3)
    values = torch.randn(2, 50, 4, generator=g)
    is_obs = torch.rand(2, 50, 4, generator=g) > 0.3
    rm = torch.ones(2, 50, dtype=torch.bool); cm = torch.ones(2, 4, dtype=torch.bool)
    f1 = per_column_deployable_features(_tokens(values, is_obs), rm, cm)
    values2 = values.clone(); values2[:, :, 1] *= 7.0  # scale one column
    f2 = per_column_deployable_features(_tokens(values2, is_obs), rm, cm)
    assert torch.allclose(f1, f2, atol=1e-4)


def test_missing_rate_correct():
    B, R, C = 1, 10, 2
    is_obs = torch.ones(B, R, C, dtype=torch.bool)
    is_obs[0, :4, 0] = False  # column 0: 4 of 10 missing
    values = torch.randn(B, R, C)
    rm = torch.ones(B, R, dtype=torch.bool); cm = torch.ones(B, C, dtype=torch.bool)
    f = per_column_deployable_features(_tokens(values, is_obs), rm, cm)
    assert abs(float(f[0, 0, 0]) - 0.4) < 1e-5  # missing_rate feature


def test_too_few_observed_zeroed():
    B, R, C = 1, 10, 2
    is_obs = torch.ones(B, R, C, dtype=torch.bool)
    is_obs[0, 1:, 1] = False  # column 1: only 1 observed
    values = torch.randn(B, R, C)
    rm = torch.ones(B, R, dtype=torch.bool); cm = torch.ones(B, C, dtype=torch.bool)
    f = per_column_deployable_features(_tokens(values, is_obs), rm, cm)
    assert torch.all(f[0, 1, :] == 0.0)


def test_padding_columns_zeroed():
    t, rm, cm = _rand(B=1, C=4)
    cm[:, 3:] = False
    f = per_column_deployable_features(t, rm, cm)
    assert torch.all(f[:, 3, :] == 0.0)


def test_skew_detects_asymmetry():
    """A skewed observed distribution has higher robust_skew than a symmetric one."""
    g = torch.Generator().manual_seed(5)
    B, R, C = 1, 200, 2
    sym = torch.randn(B, R, 1, generator=g)            # symmetric
    skewed = -torch.randn(B, R, 1, generator=g).abs()  # one-sided (skewed)
    values = torch.cat([sym, skewed], dim=2)
    is_obs = torch.ones(B, R, C, dtype=torch.bool)
    rm = torch.ones(B, R, dtype=torch.bool); cm = torch.ones(B, C, dtype=torch.bool)
    f = per_column_deployable_features(_tokens(values, is_obs), rm, cm)
    assert float(f[0, 1, 1]) > float(f[0, 0, 1])  # robust_skew: skewed col > symmetric col
