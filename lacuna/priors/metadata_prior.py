"""
lacuna.priors.metadata_prior

The metadata-authored PRIOR channel of the ADR-0008 prior x likelihood instrument.

A column's metadata (name + codebook description + domain) carries a legitimate prior over its
missingness mechanism — "income" leans self-censoring MNAR, a detection-limit assay leans detection
MNAR, a skip-gated item is MAR-by-design, a randomized rotation is MCAR-by-design. This module turns a
column's SEMANTIC CLASS (authored upstream by a small local model — see scripts/metadata_prior_bakeoff.py
— or curated) into a Dirichlet PRIOR over the composition simplex `(MCAR, MAR, MNAR)`, and pools it with
the data-likelihood's Dirichlet evidence.

Why a Dirichlet pseudo-count vector. The composition head already emits a Dirichlet `alpha = softplus+1`
(evidence `alpha - 1 >= 0`), and a deep ensemble pools members by SUMMING evidence. The metadata prior is
just one more evidence source: `alpha_post = 1 + (alpha_prior - 1) + (alpha_like - 1)`. This makes the two
ADR-0008 channels combine by the same conjugate operation, and two properties fall out for free:
  - graceful degradation (ADR-0008 commitment 4): an uninformative column -> flat prior (alpha = 1) ->
    zero evidence -> `alpha_post == alpha_like` (pure data-driven), and
  - the OVERRIDE property (ADR-0008 commitment 1): a strong, contrary data likelihood adds more evidence
    than a modest prior, so the data can pull the posterior off the prior — the prior is a NUDGE, not a lock.

The prior STRENGTH is deliberately BOUNDED so the data can override it (commitment 1): the favored
mechanism's prior probability tops out around 0.70, not ~1.0. Strongly-grounded mechanisms (a real
detection limit, explicit skip-logic) do not need a near-certain prior — an agreeing data signal reinforces
them — whereas a mislabelled column must remain overridable. The strengths here are a documented,
conservative starting point (Stage P1); they are re-fit against real likelihoods and the override test in
Stage P2.

Determinism (Coding Bible Rule 6): pure, no RNG. Fails loud (Rule 1) on an unknown semantic class or an
out-of-range reliability/evidence.
"""

from typing import Dict, Optional, Tuple

import numpy as np

N_CLASSES = 3
MCAR, MAR, MNAR = 0, 1, 2
CLASS_NAMES = ("MCAR", "MAR", "MNAR")
_PRIOR_PROB_FLOOR = 1.0 / N_CLASSES  # a flat prior; nothing is less informative than this

# semantic class -> (favored mechanism index or None for flat, target prior probability for the favoured
# mechanism). The target reliability reflects how reliably the metadata implies the mechanism (the
# benchmark's grounding tiers): strong structural cues (real LOD flag, explicit skip-logic, randomized
# administration) lean most; consensus (sensitive items) less; demographic/routine only mildly; opaque
# columns not at all. Capped at 0.70 so the data likelihood can always override (ADR-0008 commitment 1).
SEMANTIC_PRIOR_SPEC: Dict[str, Tuple[Optional[int], float]] = {
    "lab_lod": (MNAR, 0.70),               # detection-limit MNAR (strong: real LOD flags)
    "skip_gated": (MAR, 0.70),             # MAR-by-design (strong: explicit skip-logic gate)
    "planned_random": (MCAR, 0.70),        # MCAR-by-design (strong: randomized administration)
    "sensitive_disclosure": (MNAR, 0.65),  # self-censoring MNAR (consensus, not certain)
    "administrative": (MCAR, 0.62),        # IDs/dates: value-independent (moderate)
    "demographic_core": (MAR, 0.52),       # near-complete predictors (weak lean)
    "routine_measure": (MAR, 0.52),        # routine clinical measures (weak lean)
    "indeterminate": (None, _PRIOR_PROB_FLOOR),  # opaque metadata -> flat -> pure data-driven
}


def reliability_to_strength(r: float) -> float:
    """Prior pseudo-count `kappa` on the favoured class so P(favoured) = r under Dir(1,1,1+kappa).

    P(favoured) = (1 + kappa) / (N_CLASSES + kappa)  =>  kappa = (N*r - 1) / (1 - r). r in [1/N, 1).
    """
    if not _PRIOR_PROB_FLOOR <= r < 1.0:
        raise ValueError(f"target reliability must be in [1/{N_CLASSES}, 1), got {r}")
    if abs(r - _PRIOR_PROB_FLOOR) < 1e-12:
        return 0.0
    return (N_CLASSES * r - 1.0) / (1.0 - r)


def semantic_prior_alpha(semantic_class: str) -> np.ndarray:
    """Dirichlet prior pseudo-counts `alpha_prior` [3] for a semantic class (all entries >= 1).

    Raises:
        ValueError: if `semantic_class` is not in SEMANTIC_PRIOR_SPEC (fail loud, not a silent flat prior).
    """
    if semantic_class not in SEMANTIC_PRIOR_SPEC:
        raise ValueError(f"unknown semantic class {semantic_class!r}; "
                         f"valid: {sorted(SEMANTIC_PRIOR_SPEC)}")
    favored, r = SEMANTIC_PRIOR_SPEC[semantic_class]
    alpha = np.ones(N_CLASSES, dtype=float)
    if favored is not None:
        alpha[favored] += reliability_to_strength(r)
    return alpha


def combine_prior_likelihood(alpha_prior: np.ndarray, alpha_like: np.ndarray) -> np.ndarray:
    """Pool the prior and likelihood Dirichlets by summing evidence: 1 + (a_prior-1) + (a_like-1).

    Conjugate / evidential pooling (the same op the composition deep-ensemble uses). A flat prior
    (alpha_prior == 1) is a no-op; a strong likelihood can outweigh a modest prior (the override).
    """
    p = np.asarray(alpha_prior, dtype=float)
    q = np.asarray(alpha_like, dtype=float)
    if p.shape != (N_CLASSES,) or q.shape != (N_CLASSES,):
        raise ValueError(f"alphas must be shape ({N_CLASSES},), got {p.shape} and {q.shape}")
    if not (np.isfinite(p).all() and np.isfinite(q).all()) or (p < 1.0).any() or (q < 1.0).any():
        raise ValueError("alphas must be finite and >= 1 (valid Dirichlet evidence)")
    return 1.0 + (p - 1.0) + (q - 1.0)


def prior_mean(alpha: np.ndarray) -> np.ndarray:
    """Expected composition E[p] = alpha / alpha0 [3]."""
    a = np.asarray(alpha, dtype=float)
    s = a.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError(f"alpha must have positive finite sum, got {a}")
    return a / s


def channel_disagreement(alpha_prior: np.ndarray, alpha_like: np.ndarray) -> float:
    """L1 distance between the prior-mean and likelihood-mean composition — the size of the
    prior-vs-data tension, reported as a first-class output (ADR-0008 commitment 1)."""
    return float(np.abs(prior_mean(alpha_prior) - prior_mean(alpha_like)).sum())
