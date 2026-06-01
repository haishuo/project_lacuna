"""
lacuna.priors.dirichlet_evidence

Generic, ontology-agnostic Dirichlet evidence operations over an arbitrary K-class simplex.

The ADR-0008 prior x likelihood instrument combines two channels by EVIDENCE POOLING on a
Dirichlet: `alpha_post = 1 + (alpha_prior - 1) + (alpha_like - 1)` (the same conjugate operation the
composition deep-ensemble uses). The mechanism channel (`lacuna.priors.metadata_prior`) implements
this for the fixed 3-class (MCAR/MAR/MNAR) composition. The per-column SUBTYPE layer (ADR-0008
commitment 6) needs the IDENTICAL operations over a different, larger ontology (threshold / detection
/ self-censoring / ... + an abstain outcome). Rather than duplicate the math per ontology, this module
is the single generalized home for the K-class operations; the subtype modules build on it.

`metadata_prior` predates this module and keeps its own 3-class-PINNED copies (its public error
messages and shapes are a frozen contract guarded by 42 tests across the P0-P6 arc, so it is left
untouched here on purpose -- the Coding-Bible "don't perturb the frozen, attributable arc" rule wins
over de-duplicating ~10 lines of trivial numpy). This module is what NEW K-class code should import.

Determinism (Coding Bible Rule 6): pure, no RNG. Fails loud (Rule 1) on bad shapes, sub-unit /
non-finite evidence, or an out-of-range reliability / threshold -- never a silent default.
"""

from typing import Optional, Tuple

import numpy as np


def reliability_to_strength(r: float, n_classes: int) -> float:
    """Prior pseudo-count `kappa` on the favoured class so P(favoured) = r under Dir(1,...,1+kappa).

    For K classes, P(favoured) = (1 + kappa) / (K + kappa)  =>  kappa = (K*r - 1) / (1 - r).
    Valid for r in [1/K, 1): the floor 1/K is a flat prior (kappa = 0), and r -> 1 is an
    unattainable certainty (kappa -> inf).

    Raises:
        ValueError: if n_classes < 2 or r is not in [1/K, 1) (fail loud, not a silent clamp).
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    floor = 1.0 / n_classes
    if not floor <= r < 1.0:
        raise ValueError(f"target reliability must be in [1/{n_classes}, 1), got {r}")
    if abs(r - floor) < 1e-12:
        return 0.0
    return (n_classes * r - 1.0) / (1.0 - r)


def _check_vec(alpha: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if a.ndim != 1 or a.shape[0] < 2:
        raise ValueError(f"{name} must be a 1-D vector of length K>=2, got shape {a.shape}")
    if not np.isfinite(a).all() or (a < 1.0).any():
        raise ValueError(f"{name} must be finite and >= 1 (valid Dirichlet evidence)")
    return a


def combine_evidence(alpha_prior: np.ndarray, alpha_like: np.ndarray) -> np.ndarray:
    """Pool two Dirichlets by summing evidence: 1 + (a_prior-1) + (a_like-1), over K classes.

    A flat channel (alpha == 1) is a no-op (graceful degradation); a strong channel outweighs a
    modest one (the override property). Both inputs must share the same length K >= 2.
    """
    p = _check_vec(alpha_prior, "alpha_prior")
    q = _check_vec(alpha_like, "alpha_like")
    if p.shape != q.shape:
        raise ValueError(f"alphas must share shape, got {p.shape} and {q.shape}")
    return 1.0 + (p - 1.0) + (q - 1.0)


def aggregate_evidence(column_alphas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Aggregate per-column Dirichlets into ONE dataset Dirichlet, weighted by missing cells.

    The dataset composition uses the by-cell denominator (ADR-0007), so columns are weighted by how
    many missing cells they carry. Evidence is combined as a WEIGHTED AVERAGE (not a sum) so the
    dataset-level strength stays on the per-column scale (all-columns-agree preserves strength rather
    than inflating it), while the mean is the missing-cell-weighted mix of the columns' subtypes:

        alpha_dataset = 1 + sum_j (w_j / sum_w) * (alpha_col_j - 1)

    Args:
        column_alphas: [n_cols, K] per-column pseudo-counts (each >= 1, K >= 2).
        weights: [n_cols] non-negative missing-cell counts (or fractions); must sum to > 0.

    Raises:
        ValueError: on bad shapes, sub-unit / non-finite alphas, or non-positive total weight.
    """
    A = np.asarray(column_alphas, dtype=float)
    w = np.asarray(weights, dtype=float)
    if A.ndim != 2 or A.shape[1] < 2:
        raise ValueError(f"column_alphas must be [n_cols, K>=2], got shape {A.shape}")
    if w.shape != (A.shape[0],):
        raise ValueError(f"weights must be [{A.shape[0]}], got {w.shape}")
    if not np.isfinite(A).all() or (A < 1.0).any():
        raise ValueError("column_alphas must be finite and >= 1")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and sum to > 0")
    wn = w / w.sum()
    return 1.0 + (wn[:, None] * (A - 1.0)).sum(axis=0)


def evidence_mean(alpha: np.ndarray) -> np.ndarray:
    """Expected composition E[p] = alpha / alpha0 over K classes."""
    a = _check_vec(alpha, "alpha")
    return a / a.sum()


def channel_disagreement(alpha_a: np.ndarray, alpha_b: np.ndarray) -> float:
    """L1 distance between the two channels' mean compositions -- the size of their tension,
    reported as a first-class output (ADR-0008 commitment 1)."""
    return float(np.abs(evidence_mean(alpha_a) - evidence_mean(alpha_b)).sum())


def argmax_decision(alpha: np.ndarray, commit_threshold: float) -> Tuple[Optional[int], float]:
    """Commit to the argmax class only if its posterior probability clears `commit_threshold`.

    Returns ``(committed_class_index or None, p_max)``. If `p_max < commit_threshold` the caller
    ABSTAINS (class ``None``). The threshold is fit by a risk-coverage analysis so that, among the
    items on which it commits, the argmax is right at a target rate (Stage P6 machinery). This is the
    ontology-agnostic core; subtype-aware abstention (treating an explicit indeterminate argmax as an
    abstention too) layers on top in `lacuna.priors.subtype_prior`.
    """
    if not 0.0 < commit_threshold <= 1.0:
        raise ValueError(f"commit_threshold must be in (0, 1], got {commit_threshold}")
    m = evidence_mean(alpha)
    p_max = float(m.max())
    return (int(np.argmax(m)) if p_max >= commit_threshold else None), p_max
