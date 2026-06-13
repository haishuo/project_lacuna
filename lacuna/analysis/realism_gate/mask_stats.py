"""
lacuna.analysis.realism_gate.mask_stats

Descriptive statistics of a binary missingness mask.

One job: turn a binary mask matrix ``M`` (1 = missing, 0 = observed) into the
numeric summaries the realism gate consumes:

- ``row_features``         : per-row feature vectors for the C2ST classifier
                            (the raw mask columns + two row-summary features),
                            so the classifier's coefficients localise *which*
                            columns are unrealistic (Lopez-Paz & Oquab 2016).
- ``column_rates``         : per-column missing rate (attribute fidelity, Dankar).
- ``comissingness_vector`` : pairwise co-missingness correlations (bivariate
                            fidelity, Dankar).
- ``run_length_hist``      : within-row consecutive-missing run-length histogram
                            (population fidelity, Dankar).
- ``row_count_hist``       : per-row total-missing-count histogram (population
                            fidelity, Dankar).

Contract: ``M`` is an integer/bool array of shape [n, d] containing only 0/1.
Fail loud (Coding Bible §1/§2) on wrong shape, dtype, or out-of-range values.
Deterministic; no RNG, no hidden state.
"""

from typing import Tuple

import numpy as np


def check_mask(M: np.ndarray, *, name: str = "M") -> np.ndarray:
    """Validate a binary mask and return it as a contiguous uint8 array.

    Args:
        M: candidate mask, shape [n, d], values in {0, 1}.
        name: label used in error messages.

    Returns:
        uint8 copy of M with shape [n, d].

    Raises:
        ValueError: on non-2D shape, empty array, or values outside {0, 1}.
    """
    arr = np.asarray(M)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [n, d], got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN; mask must be a 0/1 indicator")
    u = np.unique(arr)
    if not np.all(np.isin(u, (0, 1))):
        raise ValueError(f"{name} must contain only 0/1, got distinct values {u[:8]}")
    return arr.astype(np.uint8, copy=True)


def longest_run_per_row(M: np.ndarray) -> np.ndarray:
    """Length of the longest run of consecutive missing cells within each row.

    Columns are treated as ordered (their order in M). Returns an int array [n].
    """
    M = check_mask(M)
    n, d = M.shape
    longest = np.zeros(n, dtype=np.int64)
    cur = np.zeros(n, dtype=np.int64)
    for j in range(d):
        col = M[:, j].astype(np.int64)
        cur = (cur + col) * col  # reset to 0 where observed, else increment
        longest = np.maximum(longest, cur)
    return longest


def row_features(M: np.ndarray) -> np.ndarray:
    """Per-row feature vectors for the C2ST classifier.

    Each row contributes: the d raw mask indicators (so per-column importance
    localises unrealistic columns), plus the row's total missing count and its
    longest consecutive-missing run (cheap structure signals).

    Returns:
        float64 array of shape [n, d + 2].
    """
    M = check_mask(M)
    n, d = M.shape
    row_count = M.sum(axis=1, keepdims=True).astype(np.float64)
    longest = longest_run_per_row(M).reshape(n, 1).astype(np.float64)
    return np.hstack([M.astype(np.float64), row_count, longest])


def column_rates(M: np.ndarray) -> np.ndarray:
    """Per-column missing rate (attribute fidelity). Returns float64 [d]."""
    M = check_mask(M)
    return M.mean(axis=0).astype(np.float64)


def comissingness_vector(M: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise co-missingness correlations (bivariate fidelity).

    For each column pair (i < j) the Pearson correlation of their missing
    indicators. Columns with zero variance (always observed / always missing)
    yield correlation 0 with every other column (no co-variation to compare).

    Returns:
        float64 vector of length d*(d-1)/2 in row-major upper-triangular order.
    """
    M = check_mask(M).astype(np.float64)
    d = M.shape[1]
    if d < 2:
        raise ValueError(f"co-missingness needs d >= 2 columns, got d={d}")
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    nz = std > 0
    Z = np.zeros_like(M)
    Z[:, nz] = (M[:, nz] - mean[nz]) / std[nz]
    corr = (Z.T @ Z) / M.shape[0]
    iu = np.triu_indices(d, k=1)
    return corr[iu].astype(np.float64)


def run_length_hist(M: np.ndarray, *, max_len: int) -> np.ndarray:
    """Histogram of within-row consecutive-missing run lengths (population fidelity).

    Counts every maximal run of consecutive missing cells across all rows, binned
    by length 1..max_len (runs longer than max_len fall in the final bin). The
    histogram is normalised to sum to 1 over observed runs; if there are no runs
    (fully observed mask) a uniform-zero vector is returned.

    Args:
        M: binary mask [n, d].
        max_len: number of length bins (>= 1).

    Returns:
        float64 vector of length max_len.
    """
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")
    M = check_mask(M)
    n, d = M.shape
    counts = np.zeros(max_len, dtype=np.float64)
    for i in range(n):
        run = 0
        for j in range(d):
            if M[i, j]:
                run += 1
            elif run > 0:
                counts[min(run, max_len) - 1] += 1
                run = 0
        if run > 0:
            counts[min(run, max_len) - 1] += 1
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def row_count_hist(M: np.ndarray, *, n_bins: int) -> np.ndarray:
    """Histogram of per-row total missing counts (population fidelity).

    The per-row missing count ranges over 0..d; binned into ``n_bins`` equal-width
    bins over [0, d] and normalised to sum to 1.

    Args:
        M: binary mask [n, d].
        n_bins: number of bins (>= 1).

    Returns:
        float64 vector of length n_bins.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    M = check_mask(M)
    d = M.shape[1]
    per_row = M.sum(axis=1)
    hist, _ = np.histogram(per_row, bins=n_bins, range=(0, d))
    total = hist.sum()
    if total == 0:
        return hist.astype(np.float64)
    return hist.astype(np.float64) / total


def overall_rate(M: np.ndarray) -> float:
    """Overall fraction of missing cells. Returns float."""
    return float(check_mask(M).mean())


def feature_names(d: int) -> Tuple[str, ...]:
    """Names aligned with ``row_features`` columns (d mask cols + 2 summaries)."""
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    return tuple([f"col_{j}" for j in range(d)] + ["row_missing_count", "longest_run"])
