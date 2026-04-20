"""
lacuna.data.missingness_features

Explicit missingness pattern features for mechanism classification.

The Problem:
    Reconstruction errors alone cannot distinguish MCAR from MAR effectively
    because cross-attention helps prediction under BOTH mechanisms (just less
    so under MCAR). The gap between MCAR and MAR reconstruction ratios is tiny.

The Solution:
    Add explicit statistical features that capture the STRUCTURE of missingness.

Features Extracted (3 groups, 9 scalars total):
    - Missing rate statistics (mean, variance, range, max across columns): 4 scalars
    - Cross-column missingness correlations (mean, max, fraction high-corr): 3 scalars
    - Little's MCAR test (chi-squared statistic, p-value): 2 scalars

Little's MCAR feature — cached, not computed in the forward pass:
    The real `pystatistics.mvnmle.little_mcar_test` is too slow to run per
    batch (seconds per call, EM-based). Instead, we precompute it once per
    (raw_dataset, generator_id) pair and cache the result. The data loader
    attaches the cached (statistic, p_value) tensors to every TokenBatch;
    this extractor reads them and slots them into the feature vector.
    See `lacuna.data.littles_cache` and `scripts/build_littles_cache.py`.

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

from lacuna.data.tokenization import IDX_OBSERVED


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MissingnessFeatureConfig:
    """Configuration for missingness feature extraction.

    The 3 surviving feature groups after the 2026-04-18 ablation. See module
    docstring for the two groups that were removed and why.
    """

    # Which feature groups to include
    include_missing_rate_stats: bool = True      # Mean, var, range of missing rates
    include_cross_column_corr: bool = True       # Correlation: missingness across columns
    include_littles_approx: bool = True          # Median-split standardised mean difference

    # Numerical stability
    eps: float = 1e-8

    @property
    def n_features(self) -> int:
        """Total number of features extracted."""
        n = 0
        if self.include_missing_rate_stats:
            n += 4  # mean, var, range, max
        if self.include_cross_column_corr:
            n += 3  # mean, max of cross-column missingness correlation
        if self.include_littles_approx:
            n += 2  # approximate chi-squared statistic, p-value proxy
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

    # 3. Little's MCAR test result, read from the offline cache. The
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

    if config.include_littles_approx:
        names.extend([
            "littles_stat",
            "littles_sig_proxy",
        ])

    return names