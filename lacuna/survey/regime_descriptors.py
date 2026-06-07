"""
lacuna.survey.regime_descriptors

Footprint-geometry / distribution-regime descriptors for a single observed column (D2 of the SCF
consolidation memo: `CONSOLIDATION-MEMO-SCF-wealth-conditional-scaling.md`).

ONE job: turn a 1-D array of observed values into a small, **scale-invariant** descriptor of the
distribution SHAPE the φ-spine actually sees. The φ-spine standardizes each column within-observed
(z-score) before pooling order statistics, so raw scale / dynamic range is largely normalized away;
what SURVIVES standardization — and therefore defines the footprint geometry — is the scale-invariant
SHAPE (skewness / tail heaviness) and the ordinal-vs-continuous axis (cardinality).

Primary descriptors (scale- and location-invariant):
  • L-skewness τ3 = L3/L2  ∈ [-1, 1]   (robust skew; stable under heavy tails, unlike moment-skew)
  • L-kurtosis τ4 = L4/L2              (robust tail heaviness; bounded, unlike moment-kurtosis which a
                                        single wealth outlier can send to ~10^4)
  • log10(cardinality)                 (the continuous ↔ ordinal/Likert axis)
  • has_negative                       (support sign — net worth carries negatives; most survey $ do not)

Secondary (reported for context, NOT used as the primary regime coordinate because φ standardizes):
  • moment skewness / excess kurtosis, raw log dynamic range, p99/p50 tail ratio.

Pure NumPy; deterministic; no RNG; no I/O. Fail loud (Rule 1) on empty / constant / too-few-rows input.
L-moments use the standard probability-weighted-moment (PWM) estimator (Hosking 1990).
"""

from typing import Dict, Sequence

import numpy as np

MIN_ROWS = 4          # L4 needs the 4th PWM b3 ⇒ at least 4 distinct order positions
LOW_CARD_MAX = 30     # < 30 distinct values ⇒ ordinal/Likert/binary regime (mirrors continuous-target gate)


def _clean(v: Sequence[float]) -> np.ndarray:
    """Observed, finite values as float64. Fail loud on empty / too-few / constant."""
    a = np.asarray(v, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < MIN_ROWS:
        raise ValueError(f"need >= {MIN_ROWS} finite values for regime descriptors, got {a.size}")
    if np.unique(a).size < 2:
        raise ValueError("column is constant; regime descriptors are undefined")
    return a


def probability_weighted_moments(v: Sequence[float]) -> tuple:
    """First four PWMs (b0..b3) via the unbiased plotting-position estimator on sorted values."""
    x = np.sort(_clean(v))
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)            # 1-indexed order
    b0 = x.mean()
    # b_r = (1/n) Σ_i [ C(i-1, r) / C(n-1, r) ] x_(i)
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    return float(b0), float(b1), float(b2), float(b3)


def l_moments(v: Sequence[float]) -> Dict[str, float]:
    """L1..L4 and the L-moment ratios L-CV / τ3 (L-skew) / τ4 (L-kurt)."""
    b0, b1, b2, b3 = probability_weighted_moments(v)
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    if l2 <= 0:
        raise ValueError("L2 (L-scale) is non-positive; degenerate column")
    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4,
            "l_cv": l2 / l1 if l1 != 0 else float("nan"),
            "l_skew": l3 / l2, "l_kurt": l4 / l2}


def _moment_shape(a: np.ndarray) -> tuple:
    """Ordinary standardized skewness and EXCESS kurtosis (secondary; sensitive to heavy tails)."""
    m = a.mean()
    s = a.std()
    if s == 0:
        return 0.0, 0.0
    z = (a - m) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def shape_descriptors(v: Sequence[float]) -> Dict[str, float]:
    """Full regime descriptor record for one observed column.

    Primary (scale-invariant): l_skew, l_kurt, log10_card, has_neg.
    Secondary (context): skew, excess_kurt, log10_range, p99_over_p50, card, n_obs.
    """
    a = _clean(v)
    lm = l_moments(a)
    skew, exkurt = _moment_shape(a)
    card = int(np.unique(a).size)
    p1, p50, p99 = np.percentile(a, [1, 50, 99])
    rng = float(a.max() - a.min())
    return {
        # primary, scale-invariant
        "l_skew": float(lm["l_skew"]),
        "l_kurt": float(lm["l_kurt"]),
        "log10_card": float(np.log10(card)),
        "has_neg": float(bool((a < 0).any())),
        # secondary, context
        "skew": skew,
        "excess_kurt": exkurt,
        "log10_range": float(np.log10(rng)) if rng > 0 else float("nan"),
        "p99_over_p50": float(p99 / p50) if p50 != 0 else float("nan"),
        "card": card,
        "is_low_card": float(card < LOW_CARD_MAX),
        "n_obs": int(a.size),
    }


# The scale-invariant coordinates used to place a column in regime space (distance / coverage).
PRIMARY_KEYS = ("l_skew", "l_kurt", "log10_card", "has_neg")


def regime_vector(v: Sequence[float]) -> np.ndarray:
    """The 4-D scale-invariant regime coordinate of a column (order = PRIMARY_KEYS)."""
    d = shape_descriptors(v)
    return np.array([d[k] for k in PRIMARY_KEYS], dtype=np.float64)
