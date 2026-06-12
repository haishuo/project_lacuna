"""Tests for the T3 conditional-φ encoder (lacuna/survey/conditional_phi.py)."""

import pytest
import torch

from lacuna.survey.conditional_phi import (ConditionalPhi, ConditionalPhiConfig,
                                           _masked_group_stats)


def _batch(b=3, r=8, dp=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(b, r, dp, generator=g)
    pcol_mask = torch.ones(b, dp, dtype=torch.bool)
    if dp > 1:
        pcol_mask[:, -1] = False                     # one padded predictor column
    obs = (torch.rand(b, r, generator=g) > 0.4).float()
    tval = torch.randn(b, r, generator=g) * obs
    row_mask = torch.ones(b, r, dtype=torch.bool)
    row_mask[:, -1] = False                          # one padded row
    return P, pcol_mask, obs, tval, row_mask


# ---- normal cases ----
def test_forward_shape_and_finite():
    model = ConditionalPhi()
    out = model(*_batch())
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_deterministic():
    model = ConditionalPhi()
    model.eval()
    args = _batch()
    with torch.no_grad():
        assert torch.allclose(model(*args), model(*args))


def test_variable_width_shares_weights():
    """Different predictor counts produce valid output from one weight set."""
    model = ConditionalPhi()
    for dp in (1, 3, 7):
        out = model(*_batch(dp=dp))
        assert out.shape == (3,) and torch.isfinite(out).all()


def test_gradients_flow_to_all_blocks():
    model = ConditionalPhi()
    model(*_batch()).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ---- edge cases ----
def test_all_observed_empty_missing_group():
    """An example whose target is fully observed (empty missing group) pools to zeros there."""
    P, pcol_mask, obs, tval, row_mask = _batch()
    obs[0] = (row_mask[0]).float()                   # all valid rows observed
    out = ConditionalPhi()(P, pcol_mask, obs, tval, row_mask)
    assert torch.isfinite(out).all()


def test_all_missing_empty_observed_group():
    P, pcol_mask, obs, tval, row_mask = _batch()
    obs[0] = 0.0
    tval[0] = 0.0
    out = ConditionalPhi()(P, pcol_mask, obs, tval, row_mask)
    assert torch.isfinite(out).all()


def test_masked_group_stats_empty_is_zero():
    tokens = torch.randn(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    stats = _masked_group_stats(tokens, mask, ConditionalPhiConfig().quantile_levels)
    assert stats.shape == (2, 4 * (1 + 3))
    assert torch.count_nonzero(stats) == 0


def test_masked_group_stats_single_row_quantiles_equal_value():
    tokens = torch.randn(1, 6, 3)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[0, 2] = True                                # exactly one valid row
    stats = _masked_group_stats(tokens, mask, (0.25, 0.5, 0.75))
    # mean and every quantile must equal that single row's value
    for blk in range(4):
        assert torch.allclose(stats[0, blk * 3:(blk + 1) * 3], tokens[0, 2], atol=1e-5)


# ---- failure cases (Rule 1: fail loud) ----
def test_bad_P_rank_raises():
    with pytest.raises(ValueError, match="P must be"):
        ConditionalPhi()(torch.randn(3, 8), torch.ones(3, 4, dtype=torch.bool),
                         torch.zeros(3, 8), torch.zeros(3, 8), torch.ones(3, 8, dtype=torch.bool))


def test_shape_mismatch_raises():
    P, pcol_mask, obs, tval, row_mask = _batch()
    with pytest.raises(ValueError, match="obs must be"):
        ConditionalPhi()(P, pcol_mask, obs[:, :-1], tval, row_mask)


def test_row_with_no_valid_rows_raises():
    P, pcol_mask, obs, tval, row_mask = _batch()
    row_mask[1] = False
    with pytest.raises(ValueError, match=">= 1 valid row"):
        ConditionalPhi()(P, pcol_mask, obs, tval, row_mask)


def test_example_with_no_valid_predictor_raises():
    P, pcol_mask, obs, tval, row_mask = _batch()
    pcol_mask[2] = False
    with pytest.raises(ValueError, match=">= 1 valid predictor"):
        ConditionalPhi()(P, pcol_mask, obs, tval, row_mask)
