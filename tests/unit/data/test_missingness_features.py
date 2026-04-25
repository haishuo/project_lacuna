"""Tests for lacuna.data.missingness_features.

Covers:
    Normal cases:
        - Default config produces n_features = 4+3 = 7 scalars (ADR 0004:
          cached Little's slot is off by default).
        - extract_missingness_features returns [B, n_features] shape.
        - Opt-in cached slot: include_littles_approx=True adds 2 scalars.
        - Heuristic flag still works; same n_features as cached opt-in.
        - compute_littles_test_approx detects structured (non-MCAR)
          missingness: MAR-pattern data produces larger test_stat than
          uniformly MCAR data.
        - get_feature_names emits heuristic_littles_* names when enabled.
    Edge cases:
        - Both MCAR flags False → n_features stays at 7 (the default).
        - Heuristic path works with all columns observed (no missingness).
    Failure cases:
        - Enabling both include_littles_approx and include_heuristic_littles
          raises ValueError.
        - Heuristic path does NOT require cached little_mcar tensors.
"""

import pytest
import torch

from lacuna.data.missingness_features import (
    MissingnessFeatureConfig,
    compute_littles_test_approx,
    extract_missingness_features,
    get_feature_names,
)
from lacuna.data.tokenization import IDX_OBSERVED, IDX_VALUE, TOKEN_DIM


def _make_tokens(values: torch.Tensor, is_observed: torch.Tensor) -> torch.Tensor:
    """Pack values + observed mask into a TOKEN_DIM-width token tensor."""
    B, R, C = values.shape
    tokens = torch.zeros(B, R, C, TOKEN_DIM)
    tokens[..., IDX_VALUE] = values
    tokens[..., IDX_OBSERVED] = is_observed.float()
    return tokens


# =============================================================================
# Config validation
# =============================================================================


def test_default_config_n_features():
    # ADR 0004: default drops from 9 → 7 (cached Little's slot is off).
    cfg = MissingnessFeatureConfig()
    assert cfg.include_littles_approx is False
    assert cfg.n_features == 4 + 3


def test_cached_opt_in_adds_two_features():
    cfg = MissingnessFeatureConfig(include_littles_approx=True)
    assert cfg.n_features == 4 + 3 + 2


def test_heuristic_flag_adds_two_features():
    cfg = MissingnessFeatureConfig(
        include_littles_approx=False,
        include_heuristic_littles=True,
    )
    assert cfg.n_features == 4 + 3 + 2


def test_both_mcar_flags_false_matches_default():
    cfg = MissingnessFeatureConfig(
        include_littles_approx=False,
        include_heuristic_littles=False,
    )
    assert cfg.n_features == 4 + 3


def test_mutex_raises_when_both_true():
    with pytest.raises(ValueError, match="mutually exclusive"):
        MissingnessFeatureConfig(
            include_littles_approx=True,
            include_heuristic_littles=True,
        )


# =============================================================================
# Heuristic computation
# =============================================================================


def test_compute_littles_test_approx_shape():
    B, R, C = 3, 20, 5
    torch.manual_seed(0)
    values = torch.randn(B, R, C)
    is_observed = (torch.rand(B, R, C) > 0.3).float()
    row_mask = torch.ones(B, R, dtype=torch.bool)
    col_mask = torch.ones(B, C, dtype=torch.bool)
    out = compute_littles_test_approx(values, is_observed, row_mask, col_mask)
    assert out.shape == (B, 2)
    assert torch.all(out >= 0) and torch.all(out <= 10.0)


def test_heuristic_detects_structured_missingness():
    """Structured missingness (high-value rows missing more in cols 1..C-1)
    should give a larger test_stat than uniform MCAR. Averaged over many
    trials to be robust to sampling noise — the heuristic is noisy by
    design at small R."""
    torch.manual_seed(0)
    R, C = 400, 6
    row_mask = torch.ones(1, R, dtype=torch.bool)
    col_mask = torch.ones(1, C, dtype=torch.bool)

    mcar_stats, mar_stats = [], []
    for trial in range(10):
        values = torch.randn(R, C)
        # MCAR: independent 30% drop
        mcar_obs = (torch.rand(R, C) > 0.3).float()
        # MAR: drop prob in cols 1..C-1 is a strong function of col 0
        col0 = values[:, 0:1]  # [R, 1]
        drop_prob = torch.sigmoid(3.0 * col0).expand(R, C).clone()
        drop_prob[:, 0] = 0.0  # keep col 0 fully observed so split is informative
        mar_obs = (torch.rand(R, C) > drop_prob).float()

        out_mcar = compute_littles_test_approx(
            values.unsqueeze(0), mcar_obs.unsqueeze(0), row_mask, col_mask
        )
        out_mar = compute_littles_test_approx(
            values.unsqueeze(0), mar_obs.unsqueeze(0), row_mask, col_mask
        )
        mcar_stats.append(out_mcar[0, 0].item())
        mar_stats.append(out_mar[0, 0].item())

    mean_mcar = sum(mcar_stats) / len(mcar_stats)
    mean_mar = sum(mar_stats) / len(mar_stats)
    assert mean_mar > mean_mcar, (
        f"MAR mean stat {mean_mar:.4f} should exceed MCAR {mean_mcar:.4f}"
    )


def test_heuristic_no_missingness_is_finite():
    B, R, C = 2, 10, 3
    values = torch.randn(B, R, C)
    is_observed = torch.ones(B, R, C)
    row_mask = torch.ones(B, R, dtype=torch.bool)
    col_mask = torch.ones(B, C, dtype=torch.bool)
    out = compute_littles_test_approx(values, is_observed, row_mask, col_mask)
    assert torch.isfinite(out).all()
    # With no missingness, low/high partitions degenerate and SMD should be ~0.
    assert out[:, 0].max().item() < 1.0


# =============================================================================
# Extractor dispatch
# =============================================================================


def test_extract_with_heuristic_does_not_need_cached_scalars():
    cfg = MissingnessFeatureConfig(
        include_littles_approx=False,
        include_heuristic_littles=True,
    )
    B, R, C = 2, 8, 4
    torch.manual_seed(1)
    values = torch.randn(B, R, C)
    is_observed = (torch.rand(B, R, C) > 0.3).float()
    tokens = _make_tokens(values, is_observed)
    row_mask = torch.ones(B, R, dtype=torch.bool)
    col_mask = torch.ones(B, C, dtype=torch.bool)
    # Must succeed WITHOUT little_mcar_stat/pvalue being passed.
    feats = extract_missingness_features(tokens, row_mask, col_mask, cfg)
    assert feats.shape == (B, cfg.n_features)
    assert torch.isfinite(feats).all()


def test_extract_with_cached_still_requires_tensors():
    # Opt-in cached slot still requires the loader-attached tensors.
    cfg = MissingnessFeatureConfig(include_littles_approx=True)
    B, R, C = 2, 8, 4
    tokens = torch.zeros(B, R, C, TOKEN_DIM)
    row_mask = torch.ones(B, R, dtype=torch.bool)
    col_mask = torch.ones(B, C, dtype=torch.bool)
    with pytest.raises(ValueError, match="little_mcar_stat"):
        extract_missingness_features(tokens, row_mask, col_mask, cfg)


def test_feature_names_includes_heuristic_labels():
    cfg = MissingnessFeatureConfig(
        include_littles_approx=False,
        include_heuristic_littles=True,
    )
    names = get_feature_names(cfg)
    assert "heuristic_littles_stat" in names
    assert "heuristic_littles_sig_proxy" in names
    assert "littles_stat" not in names
    assert len(names) == cfg.n_features
