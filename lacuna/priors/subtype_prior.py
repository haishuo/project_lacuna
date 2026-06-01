"""
lacuna.priors.subtype_prior

The metadata-authored PRIOR channel at SUBTYPE granularity (ADR-0008 commitment 6).

Where `lacuna.priors.metadata_prior` turns a column's semantic class into a Dirichlet prior over the
3-way MECHANISM composition (MCAR/MAR/MNAR), this module turns the SAME semantic class into a prior
over the finer SUBTYPE ontology `lacuna.priors.subtype_ontology.SUBTYPES` -- the granularity at which
property 3's literal output ("fairly certain 20% is THRESHOLD MNAR ... unable to determine for the
rest") lives. The semantic labels are authored upstream by the small local model (the Stage P0
bakeoff) or curated; this module is the frozen, auditable map from label to subtype prior.

The map (grounded in `scripts/metadata_prior/benchmark.json`'s `semantic_to_mechanism` table):
  - lab_lod              -> detection_mnar       (FACT: a real published LOD censors the assay)
  - skip_gated           -> mar                  (FACT: explicit skip-logic = MAR-by-design)
  - planned_random       -> mcar                 (FACT: randomized/rotated administration = MCAR-by-design)
  - sensitive_disclosure -> self_censoring_mnar  (GUT:  a domain hunch, overridable)
  - administrative       -> mcar                 (moderate: IDs/dates are value-independent)
  - demographic_core     -> mar                  (weak:  near-complete MAR predictors)
  - routine_measure      -> mar                  (weak:  measured-when-indicated)
  - indeterminate        -> flat                 (graceful degradation: opaque metadata)

`threshold_mnar` has NO semantic class that favours it -- a sharp value-threshold is a DATA
fingerprint (Stage 4), not something the codebook usually announces. The prior is therefore flat on
threshold and the DATA channel detects it; the prior's job is the OTHER subtypes (detection vs
self-censoring vs by-design), where the data is silent. This asymmetry is the honest seam at subtype
granularity: the data leads on the loud cliff, the prior leads on the semantic subtype.

Strength scales with EPISTEMIC TIER and is gated on classifier self-consistency confidence, exactly as
the mechanism prior (Stage P3 tiered fix + Stage P4 gate): FACT tier strong (favoured prob 0.85, may
override a misreading likelihood -- the by-design subtypes the data cannot see), GUT capped (0.65,
overridable), MODERATE/WEAK mild, FLAT a no-op. Reliability is computed for K = N_SUBTYPES classes.

Determinism (Coding Bible Rule 6): pure, no RNG. Fails loud (Rule 1) on an unknown semantic class or
out-of-range confidence -- never a silent flat prior.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from lacuna.priors.dirichlet_evidence import reliability_to_strength, argmax_decision
from lacuna.priors.subtype_ontology import (
    N_SUBTYPES, THRESHOLD_MNAR, DETECTION_MNAR, SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB,
)

_FLAT_R = 1.0 / N_SUBTYPES        # flat prior probability (nothing is less informative)
_FACT_R, _GUT_R = 0.85, 0.65      # tiered favoured-probabilities (Stage P3 refinement)
_GATE_FLOOR = 0.60                # de-rated fact-tier favoured prob at zero confidence (Stage P4)

# semantic class -> (favoured O subtype index or None for flat, target favoured-prob, epistemic tier).
SEMANTIC_SUBTYPE_SPEC: Dict[str, Tuple[Optional[int], float, str]] = {
    "lab_lod": (DETECTION_MNAR, _FACT_R, "fact"),
    "skip_gated": (MAR_SUB, _FACT_R, "fact"),
    "planned_random": (MCAR_SUB, _FACT_R, "fact"),
    "sensitive_disclosure": (SELF_CENSORING_MNAR, _GUT_R, "gut"),
    "administrative": (MCAR_SUB, 0.62, "moderate"),
    "demographic_core": (MAR_SUB, 0.52, "weak"),
    "routine_measure": (MAR_SUB, 0.52, "weak"),
    "indeterminate": (None, _FLAT_R, "flat"),
}


def _spec(semantic_class: str) -> Tuple[Optional[int], float, str]:
    if semantic_class not in SEMANTIC_SUBTYPE_SPEC:
        raise ValueError(f"unknown semantic class {semantic_class!r}; "
                         f"valid: {sorted(SEMANTIC_SUBTYPE_SPEC)}")
    return SEMANTIC_SUBTYPE_SPEC[semantic_class]


def subtype_tier(semantic_class: str) -> str:
    """Epistemic tier of a semantic class: 'fact' / 'gut' / 'moderate' / 'weak' / 'flat'."""
    return _spec(semantic_class)[2]


def subtype_prior_alpha(semantic_class: str) -> np.ndarray:
    """Dirichlet prior pseudo-counts `alpha_prior` [N_SUBTYPES] for a semantic class (all >= 1).

    Raises:
        ValueError: if `semantic_class` is unknown (fail loud, not a silent flat prior).
    """
    favored, r, _tier = _spec(semantic_class)
    alpha = np.ones(N_SUBTYPES, dtype=float)
    if favored is not None:
        alpha[favored] += reliability_to_strength(r, N_SUBTYPES)
    return alpha


def gated_reliability(semantic_class: str, confidence: float) -> float:
    """Effective favoured probability given classifier `confidence` in [0, 1].

    Fact tier: interpolate from `_GATE_FLOOR` (confidence 0) to full strength (confidence 1) -- an
    unstably-classified design fact is de-rated toward an overridable lean. All other tiers ignore
    confidence (already capped / weak / flat). Confidence is metadata-only (self-consistency over
    stochastic classifications), so this does NOT couple the prior to the data channel.
    """
    favored, r, tier = _spec(semantic_class)
    if tier != "fact":
        return r
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    return _GATE_FLOOR + (r - _GATE_FLOOR) * confidence


def gated_subtype_prior_alpha(semantic_class: str, confidence: float) -> np.ndarray:
    """Confidence-gated prior pseudo-counts [N_SUBTYPES]: fact-tier strength scaled by confidence.

    Identical to `subtype_prior_alpha` for non-fact tiers (confidence is ignored there).
    """
    favored, _r, _tier = _spec(semantic_class)
    alpha = np.ones(N_SUBTYPES, dtype=float)
    if favored is not None:
        alpha[favored] += reliability_to_strength(gated_reliability(semantic_class, confidence), N_SUBTYPES)
    return alpha


def selective_subtype_decision(alpha: np.ndarray, commit_threshold: float) -> Tuple[Optional[int], float]:
    """The subtype-layer 'unable to determine' decision on a fused posterior.

    Commit to the most-probable SUBTYPE iff its posterior probability `p_max >= commit_threshold`,
    else abstain (return ``(None, p_max)`` = "unable to determine"). The ontology has no explicit
    indeterminate CLASS -- abstention is the selective outcome (exactly as the Stage-P6 mechanism
    abstention): a flat posterior (data abstained AND prior flat) has a low `p_max` and is declined.
    `commit_threshold` is fit by a risk-coverage analysis (scripts/stageQ_subtype_layer.py) to a
    target committed-accuracy. Thin subtype-aware wrapper over `dirichlet_evidence.argmax_decision`.
    """
    return argmax_decision(alpha, commit_threshold)
