"""
lacuna.training.composition_calibration

Calibrate the Stage-C composition posterior — the ADR-0007 HEADLINE (Stage D).

Lacuna's deliverable is a *calibrated* distribution over the missingness composition: when the
posterior says "p% sure the composition is in simplex-region Q", that should be right p% of the time.
Stage C produced a Dirichlet posterior whose point estimate recovers the identifiable composition but
whose stated confidence is not yet reliable (its query-ECE was ~0.07–0.09 and its vacuity stopped
tracking error). This module measures that reliability over simplex-region queries and recalibrates it.

Recalibration is **temperature scaling of the Dirichlet evidence**: `alpha_cal = 1 + (alpha-1)/tau`
(the evidential analogue of logit temperature scaling). `tau > 1` spreads the Dirichlet and pulls it
toward the uniform prior (less confident); `tau < 1` sharpens it. Like logit temperature — which
flattens the softmax toward uniform rather than preserving it — this does not preserve the mean; it
interpolates between the head's posterior (tau=1) and total ignorance (tau→∞), which is exactly the
knob that fixes over/under-confident region statements. The temperature is fit on a held-out
calibration split to minimise the query-ECE, then applied at test time — a single global scalar, the
standard low-variance post-hoc recalibrator.

Region probabilities use the EXACT marginal: a coordinate of `Dir(alpha)` is `Beta(alpha_c, alpha0-alpha_c)`,
so `P(f_c >= t) = 1 - I_t(alpha_c, alpha0-alpha_c)` (regularised incomplete beta) — no Monte Carlo.

Deterministic (Coding Bible Rule 6): pure numeric, no RNG. Fails loud (Rule 1) on contract violations.
"""

from typing import List, Tuple

import numpy as np
from scipy.special import betainc

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8)
DEFAULT_TEMP_GRID = tuple(float(x) for x in np.geomspace(0.25, 8.0, 45))


def _check(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"alpha must be [N, K>=2], got shape {a.shape}")
    if not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("alpha must be finite and strictly positive")
    return a


def apply_temperature(alpha: np.ndarray, tau: float) -> np.ndarray:
    """Temperature-scale the Dirichlet evidence: alpha_cal = 1 + (alpha-1)/tau (tau > 0).

    Interpolates toward the uniform prior as tau grows (tau>1 raises the can't-tell mass/vacuity and
    pulls the mean toward 1/K; tau<1 sharpens). Does NOT preserve the mean — see module docstring.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    a = _check(alpha)
    return 1.0 + (a - 1.0) / tau


def region_prob_ge(alpha: np.ndarray, c: int, t: float) -> np.ndarray:
    """P(f_c >= t) under Dir(alpha) [N], exact via the Beta(alpha_c, alpha0-alpha_c) marginal."""
    a = _check(alpha)
    if not 0 <= c < a.shape[1]:
        raise ValueError(f"class index c={c} out of range for K={a.shape[1]}")
    if not 0.0 < t < 1.0:
        raise ValueError(f"threshold t must be in (0, 1), got {t}")
    a0 = a.sum(axis=1)
    ac = a[:, c]
    return 1.0 - betainc(ac, a0 - ac, t)


def default_query_set(n_classes: int = 3,
                      thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS) -> List[Tuple[int, float]]:
    """The simplex-region query set: P(f_c >= t) for every class c and threshold t."""
    return [(c, t) for c in range(n_classes) for t in thresholds]


def collect_query_pairs(alpha: np.ndarray, realized: np.ndarray,
                        queries: List[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Pool (predicted region prob, realised 0/1 indicator) over all queries × items."""
    a = _check(alpha)
    r = np.asarray(realized, dtype=float)
    if r.shape != a.shape:
        raise ValueError(f"realized {r.shape} must match alpha {a.shape}")
    preds, inds = [], []
    for c, t in queries:
        preds.append(region_prob_ge(a, c, t))
        inds.append((r[:, c] >= t).astype(float))
    return np.concatenate(preds), np.concatenate(inds)


def expected_calibration_error(preds: np.ndarray, inds: np.ndarray, n_bins: int = 10) -> float:
    """ECE: |mean predicted prob − empirical frequency| averaged over equal-width bins, count-weighted."""
    preds = np.asarray(preds, dtype=float)
    inds = np.asarray(inds, dtype=float)
    if preds.shape != inds.shape or preds.ndim != 1:
        raise ValueError("preds and inds must be equal-length 1-D arrays")
    n = preds.size
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (preds >= lo) & (preds < hi if b < n_bins - 1 else preds <= hi)
        if m.any():
            ece += m.sum() / n * abs(preds[m].mean() - inds[m].mean())
    return float(ece)


def brier_score(preds: np.ndarray, inds: np.ndarray) -> float:
    """Mean squared error of region-probabilities vs 0/1 outcomes — a PROPER scoring rule.

    Lower is better. Unlike ECE (reliability only), Brier rewards *resolution* too, so a constant
    prior-only predictor — trivially reliable but uninformative — scores WORSE here than a posterior
    that is both calibrated and input-dependent. This is the bar the calibrated posterior must beat.
    """
    preds = np.asarray(preds, dtype=float)
    inds = np.asarray(inds, dtype=float)
    if preds.shape != inds.shape or preds.ndim != 1:
        raise ValueError("preds and inds must be equal-length 1-D arrays")
    return float(((preds - inds) ** 2).mean())


def reliability_curve(preds: np.ndarray, inds: np.ndarray,
                      n_bins: int = 10) -> List[Tuple[float, float, int]]:
    """Per-bin (mean predicted prob, empirical frequency, count) for plotting a reliability diagram."""
    preds = np.asarray(preds, dtype=float)
    inds = np.asarray(inds, dtype=float)
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (preds >= lo) & (preds < hi if b < n_bins - 1 else preds <= hi)
        if m.any():
            out.append((float(preds[m].mean()), float(inds[m].mean()), int(m.sum())))
    return out


def query_ece(alpha: np.ndarray, realized: np.ndarray, queries: List[Tuple[int, float]],
              n_bins: int = 10) -> float:
    """Convenience: ECE over a query set for one (alpha, realized) corpus."""
    preds, inds = collect_query_pairs(alpha, realized, queries)
    return expected_calibration_error(preds, inds, n_bins=n_bins)


def fit_temperature(alpha_cal: np.ndarray, realized_cal: np.ndarray,
                    queries: List[Tuple[int, float]],
                    grid: Tuple[float, ...] = DEFAULT_TEMP_GRID, n_bins: int = 10) -> float:
    """Pick the temperature that minimises query-ECE on the calibration split (grid search).

    Returns the best tau (a scalar). Grid search (not gradient) because tau is 1-D and ECE is
    piecewise-constant — robust and reproducible.
    """
    a = _check(alpha_cal)
    best_tau, best_ece = 1.0, float("inf")
    for tau in grid:
        e = query_ece(apply_temperature(a, tau), realized_cal, queries, n_bins=n_bins)
        if e < best_ece:
            best_ece, best_tau = e, float(tau)
    return best_tau
