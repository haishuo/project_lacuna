"""
lacuna.data.missingness_features

Explicit missingness pattern features for mechanism classification.

The Problem:
    Reconstruction errors alone cannot distinguish MCAR from MAR effectively
    because cross-attention helps prediction under BOTH mechanisms (just less
    so under MCAR). The gap between MCAR and MAR reconstruction ratios is tiny.

The Solution:
    Add explicit statistical features that capture the STRUCTURE of missingness.

Features Extracted by default (2 groups, 7 scalars total — ADR 0004):
    - Missing rate statistics (mean, variance, range, max across columns): 4 scalars
    - Cross-column missingness correlations (mean, max, fraction high-corr): 3 scalars

Optional cached Little's MCAR slot (off by default — ADR 0004):
    `include_littles_approx=True` re-enables a 2-scalar slot filled with a
    precomputed (chi-squared statistic, p-value) pair read from the
    offline cache (`lacuna.data.littles_cache`). The n=30 ablation
    (docs/experiments/2026-04-25-canonical-n30.md) found that this slot
    *hurts* mechanism classification; it remains available for research-
    mode experiments that reproduce the bakeoff but is not part of the
    default training path.

Feature-group selection history:
    An earlier version also computed point-biserial correlations (missingness
    vs. observed row-mean) and distributional statistics (skewness, kurtosis);
    the n=5 ablation on 2026-04-17 found both non-contributory and they were
    removed 2026-04-18 (ADR 0001). The same ablation found the "Little's
    slot" contributory under the median-split SMD heuristic (mean Δ ≈ −6%,
    95% CI [−0.17, −0.01]); the heuristic was upgraded to the real Little's
    test via the offline cache on 2026-04-18 (ADR 0002).

Usage:
    >>> from lacuna.data.missingness_features import extract_missingness_features
    >>> features = extract_missingness_features(tokens, row_mask, col_mask)
    >>> # features: [B, n_features] tensor ready for MoE gating
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from lacuna.data.tokenization import IDX_OBSERVED, IDX_VALUE


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MissingnessFeatureConfig:
    """Configuration for missingness feature extraction.

    Two feature groups are enabled by default after ADR 0004; the cached
    Little's MCAR slot is off by default (it hurt accuracy at n=30). See
    module docstring for the feature groups removed in ADR 0001.
    """

    # Which feature groups to include
    include_missing_rate_stats: bool = True      # Mean, var, range of missing rates
    include_cross_column_corr: bool = True       # Correlation: missingness across columns
    include_value_conditional: bool = True       # SMD + shape-shift (added 2026-04-26)
    include_littles_approx: bool = False         # Cached Little's MCAR (off by default — ADR 0004)
    include_heuristic_littles: bool = False      # Median-split SMD heuristic (computed from tokens)

    # Numerical stability
    eps: float = 1e-8

    def __post_init__(self):
        if self.include_littles_approx and self.include_heuristic_littles:
            raise ValueError(
                "include_littles_approx and include_heuristic_littles are "
                "mutually exclusive: both fill the same 2-scalar MCAR slot. "
                "Set exactly one to True (or both False to disable the slot)."
            )

    @property
    def n_features(self) -> int:
        """Total number of features extracted. Default: 10 (4 + 3 + 3)."""
        n = 0
        if self.include_missing_rate_stats:
            n += 4  # mean, var, range, max
        if self.include_cross_column_corr:
            n += 3  # mean, max of cross-column missingness correlation
        if self.include_value_conditional:
            n += 3  # smd_mean, smd_max, shape_shift
        if self.include_littles_approx:
            n += 2  # cached chi-squared statistic, p-value
        if self.include_heuristic_littles:
            n += 2  # heuristic test_stat, significance_proxy
        return n


# Default configuration
DEFAULT_CONFIG = MissingnessFeatureConfig()


# =============================================================================
# Core Feature Extraction Functions
# =============================================================================

def compute_missing_rate_stats(
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute statistics of missing rates across columns.
    
    MCAR Signature: Low variance in missing rates (uniform randomness)
    MAR/MNAR Signature: High variance (some columns more affected)
    
    Returns:
        features: [B, 4] tensor with [mean_rate, var_rate, range_rate, max_rate]
    """
    B, max_rows, max_cols = is_observed.shape
    device = is_observed.device
    
    # Expand masks for broadcasting
    # row_mask: [B, max_rows] -> [B, max_rows, 1]
    # col_mask: [B, max_cols] -> [B, 1, max_cols]
    row_mask_exp = row_mask.unsqueeze(-1).float()
    col_mask_exp = col_mask.unsqueeze(1).float()
    
    # Valid cell mask: [B, max_rows, max_cols]
    valid_mask = row_mask_exp * col_mask_exp
    
    # Compute missing rate per column
    # Sum of missing indicators per column / number of valid rows
    is_missing = 1.0 - is_observed.float()
    
    # Per-column: sum missing, sum valid rows
    missing_per_col = (is_missing * valid_mask).sum(dim=1)  # [B, max_cols]
    valid_rows_per_col = valid_mask.sum(dim=1)  # [B, max_cols]
    
    # Missing rate per column (avoid division by zero)
    missing_rate = missing_per_col / valid_rows_per_col.clamp(min=1)  # [B, max_cols]
    
    # Mask out invalid columns
    col_mask_float = col_mask.float()
    n_valid_cols = col_mask_float.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    
    # Compute statistics over valid columns only
    # Mean missing rate
    mean_rate = (missing_rate * col_mask_float).sum(dim=1) / n_valid_cols.squeeze(-1)
    
    # Variance of missing rates
    rate_diff = (missing_rate - mean_rate.unsqueeze(-1)) * col_mask_float
    var_rate = (rate_diff ** 2).sum(dim=1) / n_valid_cols.squeeze(-1).clamp(min=1)
    
    # Range of missing rates (max - min over valid columns)
    # Set invalid columns to extreme values for min/max computation
    rate_for_max = missing_rate.clone()
    rate_for_max[~col_mask] = -float('inf')
    max_rate = rate_for_max.max(dim=1).values
    
    rate_for_min = missing_rate.clone()
    rate_for_min[~col_mask] = float('inf')
    min_rate = rate_for_min.min(dim=1).values
    
    range_rate = max_rate - min_rate
    
    # Handle edge cases (single column or all same rate)
    range_rate = torch.where(
        torch.isinf(range_rate) | torch.isnan(range_rate),
        torch.zeros_like(range_rate),
        range_rate
    )
    max_rate = torch.where(
        torch.isinf(max_rate),
        mean_rate,
        max_rate
    )
    
    # Stack features: [B, 4]
    features = torch.stack([mean_rate, var_rate, range_rate, max_rate], dim=-1)
    
    # Clamp to reasonable range
    features = features.clamp(min=0.0, max=1.0)
    
    return features


def compute_cross_column_missingness_correlation(
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute correlations between missingness indicators across columns.
    
    MCAR Signature: Low cross-column correlation (independent missingness)
    MAR Signature: High cross-column correlation (shared predictor drives missingness)
    MNAR Signature: Variable
    
    Returns:
        features: [B, 3] tensor with [mean_cross_corr, max_cross_corr, n_high_corr_pairs]
    """
    B, max_rows, max_cols = is_observed.shape
    device = is_observed.device
    
    # Valid mask
    row_mask_exp = row_mask.unsqueeze(-1).float()
    valid_rows = row_mask_exp  # [B, max_rows, 1]
    
    # Missingness indicators
    is_missing = (1.0 - is_observed.float()) * valid_rows  # [B, max_rows, max_cols]
    
    # Number of valid rows
    n_rows = row_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    
    # Compute correlation matrix between columns
    # Center the missingness indicators
    miss_mean = is_missing.sum(dim=1, keepdim=True) / n_rows.unsqueeze(-1)  # [B, 1, max_cols]
    miss_centered = (is_missing - miss_mean) * valid_rows  # [B, max_rows, max_cols]
    
    # Covariance matrix: [B, max_cols, max_cols]
    # cov[i,j] = sum_rows(miss_centered[:, i] * miss_centered[:, j])
    cov_matrix = torch.bmm(
        miss_centered.transpose(1, 2),  # [B, max_cols, max_rows]
        miss_centered                    # [B, max_rows, max_cols]
    )  # [B, max_cols, max_cols]
    
    # Standard deviations
    variances = cov_matrix.diagonal(dim1=1, dim2=2)  # [B, max_cols]
    stds = variances.sqrt().clamp(min=eps)  # [B, max_cols]
    
    # Correlation matrix
    # corr[i,j] = cov[i,j] / (std[i] * std[j])
    std_outer = stds.unsqueeze(-1) * stds.unsqueeze(-2)  # [B, max_cols, max_cols]
    corr_matrix = cov_matrix / std_outer.clamp(min=eps)  # [B, max_cols, max_cols]
    
    # Mask out invalid columns and diagonal
    col_mask_2d = col_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)  # [B, max_cols, max_cols]
    diag_mask = ~torch.eye(max_cols, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pairs = col_mask_2d & diag_mask
    
    # Extract off-diagonal correlations
    corr_matrix = corr_matrix * valid_pairs.float()
    corr_matrix = torch.where(
        torch.isnan(corr_matrix) | torch.isinf(corr_matrix),
        torch.zeros_like(corr_matrix),
        corr_matrix
    )
    
    # Number of valid pairs
    n_pairs = valid_pairs.float().sum(dim=(1, 2)).clamp(min=1)  # [B]
    
    # Mean absolute cross-correlation
    mean_cross_corr = corr_matrix.abs().sum(dim=(1, 2)) / n_pairs
    
    # Max absolute cross-correlation
    corr_for_max = corr_matrix.abs().clone()
    corr_for_max[~valid_pairs] = -float('inf')
    max_cross_corr = corr_for_max.view(B, -1).max(dim=1).values
    max_cross_corr = torch.where(
        torch.isinf(max_cross_corr),
        torch.zeros_like(max_cross_corr),
        max_cross_corr
    )
    
    # Count of high-correlation pairs (|corr| > 0.3)
    high_corr_mask = (corr_matrix.abs() > 0.3) & valid_pairs
    n_high_corr = high_corr_mask.float().sum(dim=(1, 2)) / n_pairs  # Normalized
    
    # Stack features: [B, 3]
    features = torch.stack([mean_cross_corr, max_cross_corr, n_high_corr], dim=-1)
    
    # Clamp to valid range
    features = features.clamp(min=0.0, max=1.0)
    
    return features


def compute_value_conditional_features(
    values: torch.Tensor,       # [B, max_rows, max_cols]
    is_observed: torch.Tensor,  # [B, max_rows, max_cols] (float, 1.0=observed)
    row_mask: torch.Tensor,     # [B, max_rows] (bool)
    col_mask: torch.Tensor,     # [B, max_cols] (bool)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Three scalars per batch sample combining values with missingness:

      smd_mean: avg standardised mean difference of observed X_k between
                rows where X_j is missing vs observed, over all valid
                (j, k) pairs. Near-zero under MCAR; positive under MAR
                (missingness in j is predictable from values of k);
                positive under MNAR-with-correlated-cols (correlation
                propagates the censoring signal). MAR / not-MCAR.

      smd_max:  max over valid (j, k) pairs. High concentration on a
                single (j, k) pair suggests a specific predictor drives
                missingness — characteristic of MAR (one observed
                covariate explains the dropout). Broader pattern with
                lower max suggests correlation-driven MNAR.

      shape_shift: avg robust-skew (|mean − median| / std) of observed
                   values for columns WITH missingness, minus the same
                   averaged over columns WITHOUT missingness. Under
                   MNAR-self-censoring, partially-missing columns have
                   truncation-distorted observed shape; fully-observed
                   columns retain natural shape ⇒ positive shift. Under
                   MAR/MCAR, missingness doesn't directly selection-bias
                   the observed values of the affected column ⇒ shift
                   ≈ 0.

    All three are batched; computation is O(B · R · C²) which is fine
    for our regime (R ≤ 128, C ≤ 48).

    Returns:
        features: [B, 3] tensor [smd_mean, smd_max, shape_shift].
    """
    B, max_rows, max_cols = values.shape
    device = values.device

    is_observed_f = is_observed.float()
    is_missing_f = 1.0 - is_observed_f
    row_mask_f = row_mask.float().unsqueeze(-1)  # [B, R, 1]
    col_mask_f = col_mask.float().unsqueeze(1)    # [B, 1, C]
    valid_cell = row_mask_f * col_mask_f          # [B, R, C]

    # is_obs_valid[b, i, k] = 1 iff row i, col k observed AND row+col valid
    is_obs_valid = is_observed_f * valid_cell    # [B, R, C]
    # j_missing_row[b, i, j] = 1 iff col j missing in row i AND row+col valid
    j_missing_row = is_missing_f * valid_cell    # [B, R, C]
    j_observed_row = is_obs_valid                 # alias for readability

    # Weighted values (zero where invalid or missing)
    weighted_values = values * is_obs_valid       # [B, R, C]

    # For each (b, j, k):
    #   numer_miss[b, j, k] = sum_i [j missing AND k observed AND valid] * X[k]
    #   denom_miss[b, j, k] = sum_i [j missing AND k observed AND valid]
    numer_miss = torch.einsum('brj,brk->bjk', j_missing_row, weighted_values)
    denom_miss = torch.einsum('brj,brk->bjk', j_missing_row, is_obs_valid)
    numer_obs = torch.einsum('brj,brk->bjk', j_observed_row, weighted_values)
    denom_obs = torch.einsum('brj,brk->bjk', j_observed_row, is_obs_valid)

    mean_k_given_j_miss = numer_miss / denom_miss.clamp(min=1)   # [B, j, k]
    mean_k_given_j_obs = numer_obs / denom_obs.clamp(min=1)      # [B, j, k]

    # Marginal std of X_k (over valid observed cells in col k)
    n_obs_k = is_obs_valid.sum(dim=1)                            # [B, C]
    sum_k = (values * is_obs_valid).sum(dim=1)                   # [B, C]
    mean_k = sum_k / n_obs_k.clamp(min=1)                        # [B, C]
    diff_k = (values - mean_k.unsqueeze(1)) * is_obs_valid       # [B, R, C]
    var_k = (diff_k ** 2).sum(dim=1) / n_obs_k.clamp(min=1)
    std_k = var_k.sqrt().clamp(min=eps)                          # [B, C]

    smd = (mean_k_given_j_miss - mean_k_given_j_obs).abs() / std_k.unsqueeze(1).clamp(min=eps)
    smd = torch.where(torch.isnan(smd) | torch.isinf(smd), torch.zeros_like(smd), smd)

    # Pair validity + sample-size guard. Without a minimum per-group
    # sample size, |SMD| can spike to 3+ on real datasets with very
    # low overall missingness (e.g. psych::bfi at 0.6 % miss-rate, where
    # the "j-missing" group has 4-10 rows per column). Those spikes are
    # sampling noise, not signal. We require ≥ MIN_GROUP_N samples in
    # both the missing and observed groups, AND shrink each pair's
    # contribution by an effective-sample-size factor — so a pair with
    # exactly MIN_GROUP_N samples in the smaller group contributes ~50%,
    # asymptoting to 100 % as group sizes grow. The shrinkage is
    # `n_eff / (n_eff + MIN_GROUP_N)` where n_eff is the harmonic mean
    # of the two group sizes.
    MIN_GROUP_N = 5.0
    diag_mask = ~torch.eye(max_cols, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pair = (
        (denom_miss >= MIN_GROUP_N)
        & (denom_obs >= MIN_GROUP_N)
        & (n_obs_k.unsqueeze(1) >= 2)
        & diag_mask
        & col_mask.unsqueeze(2)
        & col_mask.unsqueeze(1)
    )
    # Effective-sample-size shrinkage: harmonic mean / (harmonic mean + MIN_GROUP_N)
    n_eff = 2.0 / (1.0 / denom_miss.clamp(min=1) + 1.0 / denom_obs.clamp(min=1))
    shrink = n_eff / (n_eff + MIN_GROUP_N)
    smd_shrunk = smd * shrink

    smd_masked = torch.where(valid_pair, smd_shrunk, torch.zeros_like(smd_shrunk))
    n_pairs = valid_pair.float().sum(dim=(1, 2)).clamp(min=1)
    smd_mean = smd_masked.sum(dim=(1, 2)) / n_pairs

    # Max — also use shrunk values so a 1-sample pair can't spike the max.
    smd_for_max = torch.where(valid_pair, smd_shrunk, torch.full_like(smd_shrunk, -1.0))
    smd_max = smd_for_max.view(B, -1).max(dim=1).values
    smd_max = torch.where(smd_max < 0, torch.zeros_like(smd_max), smd_max)

    # === shape_shift ===
    # For each col k: |mean_k - median_k| / std_k computed over observed cells.
    # Median requires sorting masked values; do it per-batch per-col.
    # Use a robust, batch-friendly approximation: weighted median via
    # the mean of (values where R=1) sorted; we approximate the median
    # by the 0.5 quantile after masking. PyTorch torch.quantile doesn't
    # support per-element masking directly, so build per-col masked
    # tensors and compute quantile column-wise.

    # We'll compute median by setting masked-out values to NaN and
    # using nanquantile. That is per-column independent, so we reshape
    # to [B*C, R].
    masked_vals = torch.where(is_obs_valid > 0, values, torch.full_like(values, float("nan")))
    masked_vals_flat = masked_vals.permute(0, 2, 1).reshape(B * max_cols, max_rows)
    median_k_flat = torch.nanquantile(masked_vals_flat, 0.5, dim=1)
    median_k = median_k_flat.reshape(B, max_cols)
    median_k = torch.where(torch.isnan(median_k), torch.zeros_like(median_k), median_k)

    skew_k = (mean_k - median_k).abs() / std_k.clamp(min=eps)    # [B, C]

    # Which cols have missingness? (within col_mask)
    has_miss_col = ((is_missing_f * valid_cell).sum(dim=1) > 0)  # [B, C]
    has_miss_col = has_miss_col & col_mask
    has_obs_col_no_miss = (~has_miss_col) & col_mask              # [B, C]

    n_miss_cols = has_miss_col.float().sum(dim=1).clamp(min=1)
    n_clean_cols = has_obs_col_no_miss.float().sum(dim=1).clamp(min=1)

    avg_skew_miss = (skew_k * has_miss_col.float()).sum(dim=1) / n_miss_cols
    avg_skew_clean = (skew_k * has_obs_col_no_miss.float()).sum(dim=1) / n_clean_cols
    shape_shift = avg_skew_miss - avg_skew_clean

    # Sanity: clamp to a reasonable range. SMD can exceed 10 only on
    # degenerate cells; cap to prevent gradient explosion downstream.
    smd_mean = smd_mean.clamp(min=0.0, max=10.0)
    smd_max = smd_max.clamp(min=0.0, max=10.0)
    shape_shift = shape_shift.clamp(min=-5.0, max=5.0)
    shape_shift = torch.where(torch.isnan(shape_shift) | torch.isinf(shape_shift),
                              torch.zeros_like(shape_shift), shape_shift)

    return torch.stack([smd_mean, smd_max, shape_shift], dim=-1)


def compute_littles_test_approx(
    values: torch.Tensor,       # [B, max_rows, max_cols]
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """Median-split standardised-mean-difference heuristic (pre-ADR-0002 slot).

    Partition rows into "low missingness" and "high missingness" groups at
    the per-sample mean missing-count, compute the standardised mean
    difference per column between groups, and aggregate. Emits 2 scalars
    matching the layout of the cached-Little's path:
        [test_stat, sig_proxy]
    where test_stat is mean squared SMD across valid columns (chi-squared-
    like) and sig_proxy is the fraction of columns with |SMD| > 0.5.

    Large test_stat ⇒ evidence against MCAR; small ⇒ consistent with MCAR.
    See ADR 0001 and `mcar-alternatives-bakeoff` in PLANNED.md for context.

    Returns:
        [B, 2] tensor.
    """
    B, max_rows, max_cols = values.shape

    row_mask_exp = row_mask.unsqueeze(-1).float()
    col_mask_exp = col_mask.unsqueeze(1).float()
    valid_mask = row_mask_exp * col_mask_exp

    is_missing = 1.0 - is_observed.float()
    missing_per_row = (is_missing * valid_mask).sum(dim=2)  # [B, max_rows]

    row_mask_f = row_mask.float()
    mean_missing = (missing_per_row * row_mask_f).sum(dim=1) / row_mask_f.sum(dim=1).clamp(min=1)

    low_miss_rows = (missing_per_row <= mean_missing.unsqueeze(-1)) & row_mask
    high_miss_rows = (missing_per_row > mean_missing.unsqueeze(-1)) & row_mask

    low_mask = low_miss_rows.unsqueeze(-1).float() * col_mask_exp * is_observed.float()
    high_mask = high_miss_rows.unsqueeze(-1).float() * col_mask_exp * is_observed.float()

    low_count = low_mask.sum(dim=1).clamp(min=1)
    high_count = high_mask.sum(dim=1).clamp(min=1)
    low_mean = (values * low_mask).sum(dim=1) / low_count
    high_mean = (values * high_mask).sum(dim=1) / high_count

    pooled_var = (
        ((values - low_mean.unsqueeze(1)) ** 2 * low_mask).sum(dim=1)
        + ((values - high_mean.unsqueeze(1)) ** 2 * high_mask).sum(dim=1)
    ) / (low_count + high_count - 2).clamp(min=1)
    pooled_std = pooled_var.sqrt().clamp(min=eps)

    col_mask_f = col_mask.float()
    mean_diff = (high_mean - low_mean).abs() / pooled_std
    mean_diff = mean_diff * col_mask_f
    mean_diff = torch.where(
        torch.isnan(mean_diff) | torch.isinf(mean_diff),
        torch.zeros_like(mean_diff), mean_diff,
    )

    n_valid_cols = col_mask_f.sum(dim=1).clamp(min=1)
    test_stat = (mean_diff ** 2).sum(dim=1) / n_valid_cols
    sig_proxy = (mean_diff > 0.5).float().sum(dim=1) / n_valid_cols

    features = torch.stack([test_stat, sig_proxy], dim=-1)
    features = features.clamp(min=0.0, max=10.0)
    return features


# =============================================================================
# Main Feature Extraction Function
# =============================================================================

def extract_missingness_features(
    tokens: torch.Tensor,       # [B, max_rows, max_cols, TOKEN_DIM]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    config: Optional[MissingnessFeatureConfig] = None,
    *,
    little_mcar_stat: Optional[torch.Tensor] = None,    # [B]
    little_mcar_pvalue: Optional[torch.Tensor] = None,  # [B]
) -> torch.Tensor:
    """
    Extract all missingness pattern features from tokenized batch.

    This is the main entry point for feature extraction. Call this from
    the model's forward pass to get features for the MoE gating network.

    Args:
        tokens: Tokenized batch from data loader.
        row_mask: Boolean mask for valid rows.
        col_mask: Boolean mask for valid columns.
        config: Feature extraction configuration.
        little_mcar_stat: [B] precomputed Little's chi-squared statistic,
            one scalar per sample, from the offline cache (see
            `lacuna.data.littles_cache`). REQUIRED when
            config.include_littles_approx is True.
        little_mcar_pvalue: [B] precomputed Little's p-value, matched to
            `little_mcar_stat`. REQUIRED when
            config.include_littles_approx is True.

    Returns:
        features: [B, n_features] tensor of missingness pattern features.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Extract values and observation mask from tokens
    is_observed = tokens[..., IDX_OBSERVED]  # [B, max_rows, max_cols]
    is_observed_bool = is_observed > 0.5
    values = tokens[..., IDX_VALUE]  # [B, max_rows, max_cols]

    feature_list = []

    # 1. Missing rate statistics
    if config.include_missing_rate_stats:
        rate_features = compute_missing_rate_stats(
            is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(rate_features)

    # 2. Cross-column missingness correlations
    if config.include_cross_column_corr:
        cc_features = compute_cross_column_missingness_correlation(
            is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(cc_features)

    # 3. Value-conditional features (SMD + shape-shift). Added 2026-04-26
    # to give the gate signal that distinguishes MAR from MNAR — pattern
    # features alone cannot disambiguate when MNAR-with-correlated-cols
    # produces the same per-col-rate / cross-col-corr fingerprint as MAR
    # driven by an observed predictor.
    if config.include_value_conditional:
        vc_features = compute_value_conditional_features(
            values, is_observed.float(), row_mask, col_mask, config.eps
        )
        feature_list.append(vc_features)

    # 4. Little's MCAR test result, read from the offline cache. The
    # real chi-squared + p-value replace the earlier median-split
    # standardised-mean-difference heuristic that occupied this slot
    # (removed 2026-04-18 — see docs/decisions/0002). Values are
    # delivered to this function as [B] tensors from the data loader,
    # NOT computed from `tokens`, because the EM-based test is too
    # slow to run per-batch.
    if config.include_littles_approx:
        if little_mcar_stat is None or little_mcar_pvalue is None:
            raise ValueError(
                "include_littles_approx=True but little_mcar_stat/pvalue "
                "not supplied. The data loader must attach cached Little's "
                "scalars — build the cache via scripts/build_littles_cache.py."
            )
        B = tokens.shape[0]
        if little_mcar_stat.shape != (B,) or little_mcar_pvalue.shape != (B,):
            raise ValueError(
                f"little_mcar_stat/pvalue must be [B={B}] tensors; got "
                f"{tuple(little_mcar_stat.shape)} / "
                f"{tuple(little_mcar_pvalue.shape)}"
            )
        stat = little_mcar_stat.to(dtype=torch.float32, device=tokens.device)
        pval = little_mcar_pvalue.to(dtype=torch.float32, device=tokens.device)
        feature_list.append(torch.stack([stat, pval], dim=-1))

    # 4. Median-split SMD heuristic — pre-ADR-0002 version of the MCAR
    # slot. Revived 2026-04-20 for the `mcar-alternatives-bakeoff` Stage 1
    # (see PLANNED.md §3). Mutually exclusive with include_littles_approx.
    if config.include_heuristic_littles:
        heur_features = compute_littles_test_approx(
            values, is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(heur_features)

    # Concatenate all features
    if feature_list:
        features = torch.cat(feature_list, dim=-1)
    else:
        # Return empty tensor if no features configured
        B = tokens.shape[0]
        features = torch.empty(B, 0, device=tokens.device)
    
    # Final cleanup: replace any remaining NaN/Inf and clamp to reasonable range
    features = torch.where(
        torch.isnan(features) | torch.isinf(features),
        torch.zeros_like(features),
        features
    )
    
    # Clamp to prevent extreme values that cause NaN in downstream computations
    features = features.clamp(min=-10.0, max=10.0)
    
    return features


# =============================================================================
# Feature Extractor Module (for integration into model)
# =============================================================================

class MissingnessFeatureExtractor(torch.nn.Module):
    """
    PyTorch module wrapper for missingness feature extraction.

    This can be added to the model architecture and will be included
    in the forward pass.

    Usage:
        >>> extractor = MissingnessFeatureExtractor()
        >>> features = extractor(
        ...     tokens, row_mask, col_mask,
        ...     little_mcar_stat=batch.little_mcar_stat,
        ...     little_mcar_pvalue=batch.little_mcar_pvalue,
        ... )
        >>> # Concatenate with evidence for MoE gating
        >>> gate_input = torch.cat([evidence, recon_errors, features], dim=-1)
    """

    def __init__(self, config: Optional[MissingnessFeatureConfig] = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

    @property
    def n_features(self) -> int:
        """Number of features extracted."""
        return self.config.n_features

    def forward(
        self,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        little_mcar_stat: Optional[torch.Tensor] = None,
        little_mcar_pvalue: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract missingness features from batch.

        `little_mcar_stat` and `little_mcar_pvalue` are REQUIRED whenever
        `config.include_littles_approx` is True. They come from the data
        loader, which attaches them from the offline Little's MCAR cache.
        """
        return extract_missingness_features(
            tokens,
            row_mask,
            col_mask,
            self.config,
            little_mcar_stat=little_mcar_stat,
            little_mcar_pvalue=little_mcar_pvalue,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def get_feature_names(config: Optional[MissingnessFeatureConfig] = None) -> list:
    """
    Get human-readable names for each feature.
    
    Useful for debugging and analysis.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    names = []
    
    if config.include_missing_rate_stats:
        names.extend([
            "miss_rate_mean",
            "miss_rate_var",
            "miss_rate_range",
            "miss_rate_max",
        ])

    if config.include_cross_column_corr:
        names.extend([
            "cross_col_corr_mean",
            "cross_col_corr_max",
            "cross_col_high_corr_frac",
        ])

    if config.include_value_conditional:
        names.extend([
            "smd_mean",
            "smd_max",
            "shape_shift",
        ])

    if config.include_littles_approx:
        names.extend([
            "littles_stat",
            "littles_sig_proxy",
        ])

    if config.include_heuristic_littles:
        names.extend([
            "heuristic_littles_stat",
            "heuristic_littles_sig_proxy",
        ])

    return names