"""
lacuna.models.column_deployable_features

Deployable (no-oracle) per-column features for the column head (Stage 3, ADR-0006).

Stage 2b showed the per-column MNAR signal exists in reconstruction error against the TRUE
missing values — but that target is an oracle (unavailable at real inference). This module
computes per-column features from ONLY the observed values and the missingness pattern, so
they ARE deployable. The bet: MNAR self-censoring truncates a column's observed distribution
(the extreme values are the ones that go missing), leaving a distributional footprint in what
remains. We capture it with SCALE-FREE statistics (values reach the model at raw catalog scale,
so scale-dependent features would be uninformative across datasets):

  - missing_rate:    fraction missing among valid rows.
  - robust_skew:     |mean - median| / std of observed values (truncation skews this).
  - excess_kurtosis: m4 / var^2 - 3 of observed values (truncation reduces tail weight).

Deterministic. Returns [B, C, N_DEPLOYABLE_FEATURES]; padding / too-few-observed columns are 0.
"""

import torch

from lacuna.data.tokenization import IDX_VALUE, IDX_OBSERVED

N_DEPLOYABLE_FEATURES = 3  # [missing_rate, robust_skew, excess_kurtosis]


def per_column_deployable_features(
    tokens: torch.Tensor,    # [B, R, C, TOKEN_DIM]
    row_mask: torch.Tensor,  # [B, R] bool
    col_mask: torch.Tensor,  # [B, C] bool
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-column scale-free distributional features from observed values only → [B, C, 3]."""
    B, R, C, _ = tokens.shape
    vals = tokens[..., IDX_VALUE]                                  # [B, R, C]
    is_obs = tokens[..., IDX_OBSERVED] > 0.5
    valid = row_mask.unsqueeze(-1) & col_mask.unsqueeze(1)         # [B, R, C]
    obs = (is_obs & valid).to(vals.dtype)                         # [B, R, C]

    n_obs = obs.sum(dim=1)                                        # [B, C]
    n_valid = valid.to(vals.dtype).sum(dim=1).clamp(min=1.0)      # [B, C]
    miss_rate = 1.0 - n_obs / n_valid                            # [B, C]

    nden = n_obs.clamp(min=1.0)
    mean = (vals * obs).sum(dim=1) / nden                        # [B, C]
    diff = (vals - mean.unsqueeze(1)) * obs                      # [B, R, C]
    var = (diff ** 2).sum(dim=1) / nden                          # [B, C]
    std = var.sqrt().clamp(min=eps)

    # Median of observed values per column via nanquantile (mask non-observed to NaN).
    masked = torch.where(obs > 0, vals, torch.full_like(vals, float("nan")))
    masked_flat = masked.permute(0, 2, 1).reshape(B * C, R)      # [B*C, R]
    median = torch.nanquantile(masked_flat, 0.5, dim=1).reshape(B, C)
    median = torch.where(torch.isnan(median), mean, median)

    robust_skew = (mean - median).abs() / std                    # [B, C], scale-free
    m4 = (diff ** 4).sum(dim=1) / nden
    excess_kurt = m4 / (var ** 2).clamp(min=eps) - 3.0           # [B, C], scale-free

    feats = torch.stack([miss_rate, robust_skew, excess_kurt], dim=-1)  # [B, C, 3]

    # Columns with <2 observed values have undefined shape stats → zero them.
    too_few = (n_obs < 2).unsqueeze(-1)
    feats = torch.where(too_few, torch.zeros_like(feats), feats)
    feats = torch.where(torch.isnan(feats) | torch.isinf(feats), torch.zeros_like(feats), feats)
    feats = feats.clamp(min=-10.0, max=10.0)
    # Zero padding columns.
    feats = feats * col_mask.to(feats.dtype).unsqueeze(-1)
    return feats
