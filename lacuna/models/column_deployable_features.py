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
  - signed_skew:     m3 / std^3 (DIRECTION of truncation — self-censoring is one-sided).
  - smd_to_others:   mean over other columns k of |E[X_k | this col missing] - E[X_k | observed]|
                     / std(X_k) — the MAR axis (does this column's missingness track other
                     observed values?), with a minimum-group-size guard.

Deterministic. Returns [B, C, N_DEPLOYABLE_FEATURES]; padding / too-few-observed columns are 0.
"""

import torch

from lacuna.data.tokenization import IDX_VALUE, IDX_OBSERVED

N_DEPLOYABLE_FEATURES = 5  # [missing_rate, robust_skew, excess_kurtosis, signed_skew, smd_to_others]
_MIN_GROUP = 5.0  # minimum per-group sample size for the SMD-to-others feature


def per_column_deployable_features(
    tokens: torch.Tensor,    # [B, R, C, TOKEN_DIM]
    row_mask: torch.Tensor,  # [B, R] bool
    col_mask: torch.Tensor,  # [B, C] bool
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-column scale-free distributional features from observed values only → [B, C, 5].

    The 5 features are [missing_rate, robust_skew, excess_kurtosis, signed_skew, smd_to_others]
    (see module docstring); `N_DEPLOYABLE_FEATURES` is the source of truth for the count.
    """
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
    m3 = (diff ** 3).sum(dim=1) / nden
    signed_skew = (m3 / (std ** 3).clamp(min=eps)).clamp(min=-10.0, max=10.0)  # [B, C], scale-free

    # SMD-to-others (MAR axis): does THIS column's missingness shift OTHER columns' observed means?
    miss = (valid & ~is_obs).to(vals.dtype)                      # [B, R, C]
    weighted = vals * obs                                        # observed X (0 elsewhere)
    numer_miss = torch.einsum("brj,brk->bjk", miss, weighted)    # sum X_k over rows where j missing & k obs
    denom_miss = torch.einsum("brj,brk->bjk", miss, obs)         # count thereof
    numer_obs = torch.einsum("brj,brk->bjk", obs, weighted)      # j observed & k observed
    denom_obs = torch.einsum("brj,brk->bjk", obs, obs)
    mean_k_miss = numer_miss / denom_miss.clamp(min=1.0)
    mean_k_obs = numer_obs / denom_obs.clamp(min=1.0)
    smd_jk = (mean_k_miss - mean_k_obs).abs() / std.unsqueeze(1).clamp(min=eps)  # [B, j, k]
    eye = torch.eye(C, dtype=torch.bool, device=vals.device).unsqueeze(0)
    valid_pair = ((denom_miss >= _MIN_GROUP) & (denom_obs >= _MIN_GROUP) & ~eye
                  & col_mask.unsqueeze(1) & col_mask.unsqueeze(2))
    smd_jk = torch.where(valid_pair, smd_jk, torch.zeros_like(smd_jk))
    smd_jk = torch.where(torch.isnan(smd_jk) | torch.isinf(smd_jk), torch.zeros_like(smd_jk), smd_jk)
    n_pairs = valid_pair.to(vals.dtype).sum(dim=2).clamp(min=1.0)  # [B, j]
    smd_to_others = (smd_jk.sum(dim=2) / n_pairs).clamp(min=0.0, max=10.0)  # [B, C]

    feats = torch.stack(
        [miss_rate, robust_skew, excess_kurt, signed_skew, smd_to_others], dim=-1
    )  # [B, C, 5]

    # Columns with <2 observed values have undefined shape stats → zero them.
    too_few = (n_obs < 2).unsqueeze(-1)
    feats = torch.where(too_few, torch.zeros_like(feats), feats)
    feats = torch.where(torch.isnan(feats) | torch.isinf(feats), torch.zeros_like(feats), feats)
    feats = feats.clamp(min=-10.0, max=10.0)
    # Zero padding columns.
    feats = feats * col_mask.to(feats.dtype).unsqueeze(-1)
    return feats
