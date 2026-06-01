"""
lacuna.priors.subtype_ontology

The per-column SUBTYPE ontology for the ADR-0008 commitment-6 layer (the restored Q3).

Two distinct label spaces meet here, and keeping them separate is the whole point of the design:

  O -- the fused Dirichlet SUPPORT (`SUBTYPES`, K=5). The space the prior x likelihood posterior
       lives on and the dataset subtype-composition is reported over:
         threshold_mnar      -- sharp value-threshold censoring (one/two-sided cutoff)   [LOUD]
         detection_mnar      -- detection-limit / assay-LOD censoring                    [LOUD]
         self_censoring_mnar -- graded own-value MNAR with no sharp cliff (the catch-all  [QUIET]
                                quiet MNAR: self-censoring + social/strategic/selection)
         mcar                -- value-independent / by-design random                     [data-silent]
         mar                 -- predictor-driven / skip-gated                            [data-silent]

  L -- the data readout's LABEL space (`LIKELIHOOD_LABELS`, K=3). What the frozen-probe per-column
       readout is trained to emit, given Stage 4 (single-mechanism, frozen probe): threshold and
       detection MNAR leave a sharp observed-distribution cliff and are detectable at ~0.9-1.0;
       self-censoring is at the floor and MAR/MCAR have no own-value footprint, so the honest data
       target for ALL of those is an explicit REJECT class:
         threshold_mnar / detection_mnar / indeterminate(reject)

Why L is NOT O, and why this is NOT the Stage-5 trap. Stage 5 showed a forced per-column 3-way
MCAR/MAR/MNAR softmax COLLAPSES at matched miss rate (winner-take-all, seed-unstable) -- the
non-identifiable MAR-vs-MNAR boundary has no stable gradient. This layer never asks the data for
that call: the readout separates LOUD fingerprint (threshold/detection) from REJECT, and folds the
entire non-identifiable region (self-censoring, MAR, MCAR) into one reject class. Whether that
detector is stable where the 3-way was not is the Stage-Q empirical question.

The L -> O EMBEDDING is where the honest seam is enforced (`embed_likelihood_evidence`): the data's
threshold/detection evidence lands on those O classes; its reject mass becomes ZERO evidence (it
DEFERS to the prior) rather than competing against the prior-only classes. So a self-censoring column
the data correctly rejects does not pull the posterior AWAY from a sensitive-disclosure prior's
self_censoring lean -- the data is simply silent there.

Determinism (Coding Bible Rule 6): pure, no RNG. Fails loud (Rule 1) on an invalid mechanism / shape.
"""

from typing import Tuple

import numpy as np

from lacuna.core.types import MCAR as MECH_MCAR, MAR as MECH_MAR, MNAR as MECH_MNAR
from lacuna.data.mnar_column_pool import MNAR_SUBTYPES

# --- O: the fused Dirichlet support (subtype-composition is reported over this) ----------------
SUBTYPES: Tuple[str, ...] = (
    "threshold_mnar",
    "detection_mnar",
    "self_censoring_mnar",
    "mcar",
    "mar",
)
THRESHOLD_MNAR, DETECTION_MNAR, SELF_CENSORING_MNAR, MCAR_SUB, MAR_SUB = range(len(SUBTYPES))
N_SUBTYPES = len(SUBTYPES)

# The subtypes the DATA channel can place evidence on (Stage 4: loud, ~0.9-1.0 single-mechanism).
# Everything else in O is prior-only (the data is silent / abstains there).
DETECTABLE_SUBTYPES: Tuple[int, ...] = (THRESHOLD_MNAR, DETECTION_MNAR)

# Parent mechanism (MCAR/MAR/MNAR) of each O subtype -- for rolling the subtype-composition up to
# the ADR-0007 mechanism-composition (a query on the finer layer).
SUBTYPE_PARENT_MECHANISM: Tuple[int, ...] = (
    MECH_MNAR, MECH_MNAR, MECH_MNAR, MECH_MCAR, MECH_MAR,
)

# --- L: the data readout label space -----------------------------------------------------------
LIKELIHOOD_LABELS: Tuple[str, ...] = ("threshold_mnar", "detection_mnar", "indeterminate")
LIKE_THRESHOLD, LIKE_DETECTION, LIKE_INDETERMINATE = range(len(LIKELIHOOD_LABELS))
N_LIKELIHOOD_LABELS = len(LIKELIHOOD_LABELS)

# The LOUD MNAR generator families (subsets of the diverse MNAR pool). Threshold = sharp value
# cutoff; detection = assay LOD. Validated against the pool at import so a pool rename fails loud
# here rather than silently mis-folding a loud subtype into the quiet reject.
THRESHOLD_SUBTYPE_NAMES: frozenset = frozenset({
    "threshold_left", "threshold_right", "threshold_two_sided", "soft_threshold", "col_specific_thresh",
})
DETECTION_SUBTYPE_NAMES: frozenset = frozenset({
    "detection_lower", "detection_upper", "detection_both",
})
_unknown = (THRESHOLD_SUBTYPE_NAMES | DETECTION_SUBTYPE_NAMES) - set(MNAR_SUBTYPES)
if _unknown:
    raise RuntimeError(
        f"subtype_ontology lists MNAR subtypes not in the pool: {sorted(_unknown)}; "
        f"the loud-family enumeration is stale vs lacuna.data.mnar_column_pool.MNAR_SUBTYPES"
    )


def _require_mechanism(mechanism: int) -> None:
    if mechanism not in (MECH_MCAR, MECH_MAR, MECH_MNAR):
        raise ValueError(
            f"mechanism must be MCAR({MECH_MCAR})/MAR({MECH_MAR})/MNAR({MECH_MNAR}), got {mechanism!r} "
            f"(OBSERVED/padding columns are not supervised and must not be mapped)"
        )


def realised_subtype_to_ontology(mechanism: int, subtype: str) -> int:
    """Ground-truth O index for a realised (mechanism, generator-subtype) column.

    MNAR: a loud threshold/detection family -> that O class; any OTHER MNAR subtype (self-censoring,
    social, strategic, selection, ...) -> `self_censoring_mnar` (the catch-all quiet MNAR -- they are
    all own-value-dependent with no sharp cliff, the non-identifiable MNAR region). MCAR -> `mcar`;
    MAR -> `mar`. The `subtype` string is ignored for MCAR/MAR (their parent mechanism IS the class).
    """
    _require_mechanism(mechanism)
    if mechanism == MECH_MCAR:
        return MCAR_SUB
    if mechanism == MECH_MAR:
        return MAR_SUB
    # MNAR
    if subtype in THRESHOLD_SUBTYPE_NAMES:
        return THRESHOLD_MNAR
    if subtype in DETECTION_SUBTYPE_NAMES:
        return DETECTION_MNAR
    return SELF_CENSORING_MNAR


def realised_subtype_to_likelihood_label(mechanism: int, subtype: str) -> int:
    """Data-readout training label (L index) for a realised column.

    Only the LOUD MNAR families get a non-reject label (threshold / detection); EVERY other supervised
    column -- quiet MNAR, all MAR, all MCAR -- is labelled `indeterminate` (reject). This is the honest
    target: the data can read the loud cliff and should ABSTAIN on the rest (the prior supplies it).
    """
    _require_mechanism(mechanism)
    if mechanism == MECH_MNAR:
        if subtype in THRESHOLD_SUBTYPE_NAMES:
            return LIKE_THRESHOLD
        if subtype in DETECTION_SUBTYPE_NAMES:
            return LIKE_DETECTION
    return LIKE_INDETERMINATE


def embed_likelihood_evidence(p_like: np.ndarray, kappa: float) -> np.ndarray:
    """Embed data-readout probabilities [.., 3] over L into Dirichlet evidence [.., 5] over O.

    `alpha_like = 1` on every O class, plus `kappa * p` added to threshold_mnar / detection_mnar from
    the corresponding L probabilities. The reject (`indeterminate`) probability is DROPPED -- it
    becomes zero evidence, so the data DEFERS to the prior on the non-detectable classes rather than
    competing with it (the honest seam; see module docstring). `kappa` is the data channel's total
    concentration: how much Dirichlet evidence a fully-confident loud detection contributes.

    Args:
        p_like: [..., N_LIKELIHOOD_LABELS] readout probabilities (rows finite, >= 0).
        kappa: non-negative data-evidence concentration scalar.

    Raises:
        ValueError: on a bad last dim, non-finite / negative probabilities, or kappa < 0.
    """
    p = np.asarray(p_like, dtype=float)
    if p.shape[-1] != N_LIKELIHOOD_LABELS:
        raise ValueError(f"p_like last dim must be {N_LIKELIHOOD_LABELS}, got shape {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("p_like must be finite and >= 0 (readout probabilities)")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError(f"kappa must be finite and >= 0, got {kappa}")
    alpha = np.ones(p.shape[:-1] + (N_SUBTYPES,), dtype=float)
    alpha[..., THRESHOLD_MNAR] += kappa * p[..., LIKE_THRESHOLD]
    alpha[..., DETECTION_MNAR] += kappa * p[..., LIKE_DETECTION]
    return alpha
