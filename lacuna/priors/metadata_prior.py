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

The prior STRENGTH scales with the EPISTEMIC TIER of the semantic class (Stage P3 refinement):
  - FACT tier (planned_random, skip_gated, lab_lod): the metadata STATES a design/measurement fact the
    data channel cannot see (a rotated booklet IS MCAR-by-design; a skip-gate IS MAR; an LOD censors).
    These get a STRONG prior (favoured prob ~0.85) — strong enough to override a likelihood that is
    *misreading* the footprint (e.g. PISA rotation read as "structured"). This is not a spurious lookup;
    it is reading the codebook, and there is no legitimate data evidence against a stated design fact.
  - GUT tier (sensitive_disclosure): an honest domain HUNCH ("income is usually self-censored"). CAPPED
    (favoured prob <= 0.70) so a wrong hunch stays overridable by the data (ADR-0008 commitment 1; cf. the
    survey_chile income column whose nonresponse is consensus-MAR, not MNAR).
  - MODERATE/WEAK (administrative, demographic_core, routine_measure): mild value-independence or
    predictor leans, capped low. FLAT (indeterminate): no prior (graceful degradation).
The override-safety cap is therefore a GUT-tier rule, not a blanket one — the Stage-P3 finding that a
uniform 0.70 cap left the fact tier too weak to fix the documented PISA MCAR-by-design blind spot.

Determinism (Coding Bible Rule 6): pure, no RNG. Fails loud (Rule 1) on an unknown semantic class or an
out-of-range reliability/evidence.
"""

from typing import Dict, Optional, Tuple

import numpy as np

N_CLASSES = 3
MCAR, MAR, MNAR = 0, 1, 2
CLASS_NAMES = ("MCAR", "MAR", "MNAR")
_PRIOR_PROB_FLOOR = 1.0 / N_CLASSES  # a flat prior; nothing is less informative than this

# semantic class -> (favored mechanism index or None for flat, target prior probability, epistemic tier).
# Strength scales with tier (Stage P3): FACT tier states a design/measurement fact (strong, ~0.85, may
# override a misreading likelihood); GUT tier is an overridable hunch (capped <= 0.70); MODERATE/WEAK are
# mild leans; FLAT is no prior. See the module docstring for the rationale.
_FACT_R, _GUT_R = 0.85, 0.65
SEMANTIC_PRIOR_SPEC: Dict[str, Tuple[Optional[int], float, str]] = {
    "lab_lod": (MNAR, _FACT_R, "fact"),              # detection-limit MNAR (real LOD flag in the codebook)
    "skip_gated": (MAR, _FACT_R, "fact"),            # MAR-by-design (explicit skip-logic gate)
    "planned_random": (MCAR, _FACT_R, "fact"),       # MCAR-by-design (randomized/rotated administration)
    "sensitive_disclosure": (MNAR, _GUT_R, "gut"),   # self-censoring MNAR (a domain hunch, overridable)
    "administrative": (MCAR, 0.62, "moderate"),      # IDs/dates: value-independent
    "demographic_core": (MAR, 0.52, "weak"),         # near-complete predictors (mild lean)
    "routine_measure": (MAR, 0.52, "weak"),          # routine clinical measures (mild lean)
    "indeterminate": (None, _PRIOR_PROB_FLOOR, "flat"),  # opaque metadata -> flat -> pure data-driven
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
    favored, r, _tier = SEMANTIC_PRIOR_SPEC[semantic_class]
    alpha = np.ones(N_CLASSES, dtype=float)
    if favored is not None:
        alpha[favored] += reliability_to_strength(r)
    return alpha


def semantic_tier(semantic_class: str) -> str:
    """Epistemic tier of a semantic class: 'fact' / 'gut' / 'moderate' / 'weak' / 'flat'."""
    if semantic_class not in SEMANTIC_PRIOR_SPEC:
        raise ValueError(f"unknown semantic class {semantic_class!r}")
    return SEMANTIC_PRIOR_SPEC[semantic_class][2]


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


def aggregate_column_priors(column_alphas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Aggregate per-column Dirichlet priors into one DATASET prior, weighted by missing cells.

    The dataset composition uses the by-cell denominator (ADR-0007), so columns are weighted by how many
    missing cells they carry. Evidence is combined as a WEIGHTED AVERAGE (not a sum): the dataset prior
    strength stays on the per-column scale (so all-columns-agree preserves strength rather than inflating
    it), while the mean is the missing-cell-weighted mix of the columns' mechanisms.

        alpha_dataset = 1 + sum_j (w_j / sum_w) * (alpha_col_j - 1)

    Args:
        column_alphas: [n_cols, 3] per-column prior pseudo-counts (each >= 1).
        weights: [n_cols] non-negative per-column missing-cell counts (or fractions); must sum to > 0.

    Raises:
        ValueError: on bad shapes, sub-unit / non-finite alphas, or non-positive total weight.
    """
    A = np.asarray(column_alphas, dtype=float)
    w = np.asarray(weights, dtype=float)
    if A.ndim != 2 or A.shape[1] != N_CLASSES:
        raise ValueError(f"column_alphas must be [n, {N_CLASSES}], got {A.shape}")
    if w.shape != (A.shape[0],):
        raise ValueError(f"weights must be [{A.shape[0]}], got {w.shape}")
    if not np.isfinite(A).all() or (A < 1.0).any():
        raise ValueError("column_alphas must be finite and >= 1")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and sum to > 0")
    wn = w / w.sum()
    return 1.0 + (wn[:, None] * (A - 1.0)).sum(axis=0)


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
