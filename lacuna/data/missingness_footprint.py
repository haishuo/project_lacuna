"""
lacuna.data.missingness_footprint

Observable missingness-footprint features (ADR-0007, Stage A — the realism harness).

Summarise a missingness MASK (+ the observed values) with a fixed vector of scale-free,
source-agnostic statistics computable from ONLY the mask and the observed values — no mechanism
label, no held-out truth. This is what lets a real survey mask and one of our synthetic masks be
compared on the same footing: if the generators are realistic, their footprint distribution should
be statistically indistinguishable from real survey data's (the realism check is a comparison of
these vectors, NOT a learned generator — see ADR-0007 commitment 5).

Convention (matches `ObservedDataset.r`): `r` is True where a cell is OBSERVED, False where missing.
Only `x[r]` is ever read, so it does not matter whether missing entries are encoded as 0 (synthetic
path) or NaN (real CSVs).

The 20 features, grouped:
  column rates    — per-column miss-rate distribution (level, spread, skew, extremes, how many
                    columns are fully observed / heavily missing).
  row patterns    — per-row missingness spread + complete / near-empty row fractions (the
                    unit-nonresponse signature).
  co-missingness  — pairwise missingness correlations (do columns go missing together → blocks /
                    modules).
  pattern struct  — distinct-pattern ratio + mass in the top patterns (skip-logic → few dominant
                    patterns; pure randomness → many).
  monotonicity    — fraction of rows whose missingness is a clean suffix under the best column order
                    (dropout / attrition signature).
  MAR axis        — coupling of each column's missingness with OTHER columns' observed values.
  MNAR axis       — shape (|skew|, excess kurtosis) of each column's observed values.

Deterministic (Coding Bible Rule 6): a pure summary, no RNG. Fails loud (Rule 1) on a structurally
invalid request (shape mismatch, n < 1, d < 1).

Contract:
    missingness_footprint(x, r, *, min_group=5) -> dict[str, float]
Keys are exactly `FOOTPRINT_FEATURES`, in that order. Undefined statistics (e.g. correlations on a
constant column, shape stats on < 4 observed values) are reported as 0.0, never NaN.
"""

from collections import Counter
from typing import Dict

import numpy as np

try:  # torch is the project's data primitive, but the function is pure-numpy internally.
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover - torch is always present in the lacuna env
    _TORCH = False


FOOTPRINT_FEATURES = (
    # column rates
    "col_rate_mean", "col_rate_sd", "col_rate_max", "col_rate_skew",
    "frac_cols_complete", "frac_cols_high_miss",
    # row patterns
    "row_missfrac_sd", "frac_rows_complete", "frac_rows_high_miss",
    # co-missingness
    "miss_corr_mean_abs", "miss_corr_max_abs", "frac_pairs_coupled",
    # pattern structure
    "distinct_pattern_ratio", "top1_pattern_frac", "top3_pattern_frac",
    # monotonicity
    "monotone_row_frac",
    # MAR axis (missingness vs other columns' observed values)
    "mar_coupling_mean", "mar_coupling_max",
    # MNAR axis (observed-value shape)
    "obs_abs_skew_mean", "obs_excess_kurt_mean",
)

_HIGH_MISS = 0.5      # a column/row above this miss fraction is "heavily missing"
_COUPLED = 0.5        # |corr| above this counts as a strongly-coupled pair


def _to_numpy(a) -> np.ndarray:
    if _TORCH and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _skew(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0.0
    s = v.std()
    if s == 0:
        return 0.0
    return float((((v - v.mean()) / s) ** 3).mean())


def _excess_kurt(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size < 4:
        return 0.0
    s = v.std()
    if s == 0:
        return 0.0
    return float((((v - v.mean()) / s) ** 4).mean() - 3.0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _column_rate_features(miss: np.ndarray) -> Dict[str, float]:
    m = miss.mean(axis=0)  # [d] per-column miss rate
    return {
        "col_rate_mean": float(m.mean()),
        "col_rate_sd": float(m.std()),
        "col_rate_max": float(m.max()),
        "col_rate_skew": _skew(m),
        "frac_cols_complete": float((m == 0.0).mean()),
        "frac_cols_high_miss": float((m > _HIGH_MISS).mean()),
    }


def _row_pattern_features(miss: np.ndarray) -> Dict[str, float]:
    f = miss.mean(axis=1)  # [n] per-row miss fraction
    return {
        "row_missfrac_sd": float(f.std()),
        "frac_rows_complete": float((f == 0.0).mean()),
        "frac_rows_high_miss": float((f > _HIGH_MISS).mean()),
    }


def _co_missingness_features(miss: np.ndarray) -> Dict[str, float]:
    n, d = miss.shape
    if d < 2 or n < 2:  # correlations are undefined with <2 columns or <2 rows
        return {"miss_corr_mean_abs": 0.0, "miss_corr_max_abs": 0.0, "frac_pairs_coupled": 0.0}
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(miss, rowvar=False)  # [d, d]; NaN where a column is constant
    iu = np.triu_indices(d, k=1)
    pair = np.abs(c[iu])
    pair = pair[np.isfinite(pair)]
    if pair.size == 0:
        return {"miss_corr_mean_abs": 0.0, "miss_corr_max_abs": 0.0, "frac_pairs_coupled": 0.0}
    return {
        "miss_corr_mean_abs": float(pair.mean()),
        "miss_corr_max_abs": float(pair.max()),
        "frac_pairs_coupled": float((pair > _COUPLED).mean()),
    }


def _pattern_structure_features(miss_bool: np.ndarray) -> Dict[str, float]:
    n = miss_bool.shape[0]
    _, counts = np.unique(miss_bool, axis=0, return_counts=True)
    cs = np.sort(counts)[::-1]
    return {
        "distinct_pattern_ratio": float(counts.size / n),
        "top1_pattern_frac": float(cs[0] / n),
        "top3_pattern_frac": float(cs[:3].sum() / n),
    }


def _monotonicity_feature(miss_bool: np.ndarray) -> Dict[str, float]:
    """Fraction of rows whose missingness is a clean suffix once columns are ordered most-observed
    first — i.e. the row observes a prefix and misses a suffix (monotone / dropout structure)."""
    d = miss_bool.shape[1]
    if d < 2:
        return {"monotone_row_frac": 1.0}  # a single column is trivially monotone
    rates = miss_bool.mean(axis=0)
    order = np.argsort(rates)                       # ascending miss rate = most-observed first
    seq = miss_bool[:, order].astype(np.int8)       # [n, d], 1 = missing
    nondec = (np.diff(seq, axis=1) >= 0).all(axis=1)  # observed(0)s then missing(1)s
    return {"monotone_row_frac": float(nondec.mean())}


def _coupling_features(x: np.ndarray, r: np.ndarray, miss: np.ndarray, min_group: int) -> Dict[str, float]:
    """MAR axis: per partially-missing column j, the strongest |corr| between its missingness
    indicator and any OTHER column k's observed values (over rows where k is observed)."""
    n, d = miss.shape
    per_col = []
    for j in range(d):
        mj = miss[:, j]
        if mj.std() == 0:          # fully observed or fully missing → nothing to couple
            continue
        best = 0.0
        for k in range(d):
            if k == j:
                continue
            obs_k = r[:, k]
            if obs_k.sum() < min_group:
                continue
            a = mj[obs_k]
            b = x[obs_k, k]
            fin = np.isfinite(b)
            if fin.sum() < min_group:
                continue
            best = max(best, abs(_safe_corr(a[fin], b[fin])))
        per_col.append(best)
    if not per_col:
        return {"mar_coupling_mean": 0.0, "mar_coupling_max": 0.0}
    return {"mar_coupling_mean": float(np.mean(per_col)), "mar_coupling_max": float(np.max(per_col))}


def _obs_shape_features(x: np.ndarray, r: np.ndarray) -> Dict[str, float]:
    """MNAR axis: mean |skew| and mean excess kurtosis of each column's observed values."""
    skews, kurts = [], []
    for j in range(x.shape[1]):
        v = x[r[:, j], j]
        v = v[np.isfinite(v)]
        if v.size < 4:
            continue
        skews.append(abs(_skew(v)))
        kurts.append(_excess_kurt(v))
    return {
        "obs_abs_skew_mean": float(np.mean(skews)) if skews else 0.0,
        "obs_excess_kurt_mean": float(np.mean(kurts)) if kurts else 0.0,
    }


def missingness_footprint(x, r, *, min_group: int = 5) -> Dict[str, float]:
    """Observable missingness-footprint feature vector for one dataset.

    Args:
        x: [n, d] values (torch.Tensor or np.ndarray). Only `x[r]` is read; missing entries may be
            anything (0 or NaN).
        r: [n, d] boolean mask, True where OBSERVED.
        min_group: minimum sample size for a correlation to be computed (else that term is 0).

    Returns:
        dict keyed by `FOOTPRINT_FEATURES` (that order), all finite floats.

    Raises:
        ValueError: on shape mismatch, n < 1, or d < 1.
    """
    x = _to_numpy(x).astype(float)
    r = _to_numpy(r).astype(bool)
    if x.shape != r.shape:
        raise ValueError(f"x and r must have the same shape, got {x.shape} vs {r.shape}")
    if x.ndim != 2:
        raise ValueError(f"x and r must be 2-D [n, d], got ndim={x.ndim}")
    n, d = x.shape
    if n < 1 or d < 1:
        raise ValueError(f"need n >= 1 and d >= 1, got n={n}, d={d}")

    miss_bool = ~r
    miss = miss_bool.astype(float)

    feats: Dict[str, float] = {}
    feats.update(_column_rate_features(miss))
    feats.update(_row_pattern_features(miss))
    feats.update(_co_missingness_features(miss))
    feats.update(_pattern_structure_features(miss_bool))
    feats.update(_monotonicity_feature(miss_bool))
    feats.update(_coupling_features(x, r, miss, min_group))
    feats.update(_obs_shape_features(x, r))

    # Guarantee the contract: exact key set, finite values, fixed order.
    out: Dict[str, float] = {}
    for name in FOOTPRINT_FEATURES:
        v = feats[name]
        out[name] = float(v) if np.isfinite(v) else 0.0
    return out
