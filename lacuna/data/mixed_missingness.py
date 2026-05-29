"""
lacuna.data.mixed_missingness

Compose a SINGLE dataset whose columns carry DIFFERENT missingness mechanisms.

This is a per-column generalization of `lacuna.data.semisynthetic.apply_missingness`.
Where `apply_missingness` applies one generator to the whole matrix (one mechanism
label per dataset), this module applies a (possibly different) mechanism per column
and returns a per-column class label vector. It is the data primitive for the
column-level missingness experiment (see docs/decisions/0006).

Contract
--------
Input:
  - raw:            a complete RawDataset (no missing values) from the catalog.
  - column_classes: length-d assignment, each entry in {OBSERVED, MCAR, MAR, MNAR}.
  - rng:            an explicit RNGState (no global randomness; Coding Bible Rule 6).
Output:
  - MixedMissingnessResult: the masked ObservedDataset, the per-column class vector
    (echoed back as ground truth), the *actual* per-column missing rates, and the
    predictor column chosen for each MAR target.

Design choices (Stage 0, ADR-0006)
-----------------------------------
- Mechanisms are realised with the SAME generator classes the model trained on
  (`MCARBernoulli`, `MARLogistic`, `MNARLogistic`), instantiated with an explicit
  `target_col_idx` so each column's mask is honest. Only three canonical generators
  are used here — a deliberate, documented scoping of the ~110-generator registry to
  keep Stage 0 mixtures controlled and interpretable. The mechanisms are still
  in-distribution per-column; only the *mixing across columns* is novel.
- CLEAN MAR regime (ADR-0006 §"MAR-predictor policy"): a MAR column's predictor is
  drawn only from columns that are themselves fully observed (OBSERVED) or MCAR, so
  the predictor is observed wherever the MAR column's missingness is decided. The
  entangled regime (MAR predictors that are themselves missing) is deliberately NOT
  implemented here; it is a later ablation.
- Missingness is decided on a per-column z-scored *predictor view* of X — identical to
  `apply_missingness` — so logistic generators parameterised for standard-normal X do
  not saturate on real catalog scales. The OBSERVED VALUES handed to the model remain
  on the original scale (`raw.data * R`), exactly as the training path does.
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset, MCAR, MAR, MNAR
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mcar.bernoulli import MCARBernoulli
from lacuna.generators.families.mar.simple import MARLogistic
from lacuna.generators.families.mnar.self_censoring import MNARLogistic
from lacuna.data.mnar_column_pool import sample_mnar_column_generator
from lacuna.data.mar_column_pool import sample_mar_column_generator
from lacuna.data.ingestion import RawDataset
# Reuse the EXACT predictor-view scaling the training path uses, so generator
# saturation behaviour matches apply_missingness. Intentional, explicit coupling.
from lacuna.data.semisynthetic import _zscore_columns

# Sentinel for "this column is fully observed" (no missingness applied). Distinct
# from the mechanism class ids MCAR=0 / MAR=1 / MNAR=2.
OBSERVED = -1

_VALID_CLASSES = frozenset({OBSERVED, MCAR, MAR, MNAR})


@dataclass(frozen=True)
class MixedMissingnessResult:
    """Result of composing a mixed-mechanism dataset.

    Attributes:
        observed: ObservedDataset with per-column mechanisms applied (missing = 0).
        column_classes: length-d tuple echoing the requested per-column assignment
            (the per-column ground truth, including OBSERVED columns).
        per_column_miss_rate: length-d tuple of the ACTUAL missing fraction realised
            in each column (reported, never silently assumed — Coding Bible Rule 1).
        mar_predictors: {target_col -> predictor_col} for every MAR column.
        mnar_subtypes: {target_col -> subtype_name} for every MNAR column. Always
            "self_censoring" in the default regime; varied (threshold/detection/...) when
            `mnar_diverse=True`. Reported so the realised MNAR subtype mix is never a
            silent assumption (Coding Bible Rule 1).
        mar_subtypes: {target_col -> subtype_name} for every MAR column. Always "logistic"
            in the default regime; varied (probit/threshold/realistic/...) when
            `mar_diverse=True`. Reported for the same reason as mnar_subtypes.
        source_name: name of the source RawDataset.
    """
    observed: ObservedDataset
    column_classes: Tuple[int, ...]
    per_column_miss_rate: Tuple[float, ...]
    mar_predictors: Dict[int, int]
    mnar_subtypes: Dict[int, str]
    mar_subtypes: Dict[int, str]
    source_name: str


def _logit(p: float) -> float:
    """Inverse sigmoid. Used to set a logistic intercept targeting marginal rate p."""
    return math.log(p / (1.0 - p))


def _validate(
    raw: RawDataset,
    column_classes: Tuple[int, ...],
    target_miss_rate: float,
) -> Tuple[int, ...]:
    """Boundary validation (Coding Bible Rules 1 & 2). Returns the clean predictor pool."""
    if len(column_classes) != raw.d:
        raise ValueError(
            f"column_classes has length {len(column_classes)} but dataset "
            f"'{raw.name}' has d={raw.d} columns"
        )
    bad = sorted({c for c in column_classes if c not in _VALID_CLASSES})
    if bad:
        raise ValueError(
            f"column_classes contains invalid entries {bad}; allowed: "
            f"OBSERVED({OBSERVED}), MCAR({MCAR}), MAR({MAR}), MNAR({MNAR})"
        )
    if not 0.0 < target_miss_rate < 1.0:
        raise ValueError(f"target_miss_rate must be in (0, 1), got {target_miss_rate}")

    has_mar = any(c == MAR for c in column_classes)
    has_mnar = any(c == MNAR for c in column_classes)
    if (has_mar or has_mnar) and raw.d < 2:
        raise ValueError(
            f"MAR/MNAR logistic generators require d >= 2; dataset '{raw.name}' has d={raw.d}"
        )

    # Clean-regime predictor pool: fully-observed columns, else MCAR columns.
    pool = tuple(j for j, c in enumerate(column_classes) if c == OBSERVED)
    if not pool:
        pool = tuple(j for j, c in enumerate(column_classes) if c == MCAR)
    if has_mar and not pool:
        raise ValueError(
            "MAR columns require at least one OBSERVED or MCAR column to serve as a "
            "clean predictor, but none was assigned. Reserve a clean predictor column "
            "(this enforces the ADR-0006 clean-MAR regime)."
        )
    return pool


def compose_mixed_missingness(
    raw: RawDataset,
    column_classes: Tuple[int, ...],
    rng: RNGState,
    *,
    target_miss_rate: float = 0.25,
    mar_strength: float = 1.5,
    mnar_strength: float = 1.5,
    mnar_diverse: bool = False,
    mar_diverse: bool = False,
) -> MixedMissingnessResult:
    """Apply a per-column mixture of MCAR/MAR/MNAR mechanisms to a complete dataset.

    Args:
        raw: Complete RawDataset (no missing values).
        column_classes: length-d assignment; each entry in {OBSERVED, MCAR, MAR, MNAR}.
        rng: Explicit RNG state for reproducibility.
        target_miss_rate: Marginal missing fraction each non-observed column aims for.
            Intercepts are set to logit(target_miss_rate) so all three mechanisms produce
            comparable missingness amounts — controlling the miss-rate confound so the
            model's response reflects mechanism *type*, not *quantity*.
        mar_strength: Logistic slope (alpha1) coupling a MAR column to its predictor.
        mnar_strength: Logistic slope (beta2) coupling an MNAR column to its own value
            (used only for the self-censoring subtype).
        mnar_diverse: if False (default), every MNAR column is logistic self-censoring —
            the original Stage 0–3 behaviour, unchanged. If True, each MNAR column draws a
            subtype from the full per-column-targetable MNAR pool (self-censoring + threshold
            + detection-limit + social + strategic + ... families; see `mnar_column_pool`), so
            mixtures contain real MNAR subtype diversity (ADR-0006 / Stage 4–5 follow-up). The
            drawn subtype per column is recorded in the result's `mnar_subtypes`.
        mar_diverse: if False (default), every MAR column is `MARLogistic` — the original
            Stage 0–4b behaviour, unchanged. If True, each MAR column draws a subtype from the
            single-predictor MAR pool (logistic/probit/polynomial/threshold/step/binary/
            discrete/realistic; see `mar_column_pool`), preserving the clean-MAR regime (the
            predictor is still a clean observed/MCAR column). Recorded in `mar_subtypes`. Setting
            BOTH diverse flags yields the full-diversity true-mixture regime (Stage 5).

    Returns:
        MixedMissingnessResult.

    Raises:
        ValueError: on contract violations (see _validate).
    """
    pool = _validate(raw, column_classes, target_miss_rate)

    X = torch.from_numpy(raw.data.astype("float32"))
    n, d = X.shape
    # Predictor view: per-column z-score, exactly as apply_missingness does, so the
    # logistic generators behave on real data as they did in training.
    Z = _zscore_columns(X)

    intercept = _logit(target_miss_rate)
    R = torch.ones(n, d, dtype=torch.bool)
    mar_predictors: Dict[int, int] = {}
    mnar_subtypes: Dict[int, str] = {}
    mar_subtypes: Dict[int, str] = {}

    for j, cls in enumerate(column_classes):
        if cls == OBSERVED:
            continue

        if cls == MCAR:
            gen = MCARBernoulli(0, "mixed_mcar", GeneratorParams(miss_rate=target_miss_rate))
            r_full = gen.apply_to(Z, rng.spawn())

        elif cls == MAR:
            # Pick a clean predictor (observed/MCAR), never the target column itself.
            candidates = tuple(k for k in pool if k != j)
            if not candidates:
                raise ValueError(
                    f"No clean predictor available for MAR column {j} in '{raw.name}'"
                )
            predictor = candidates[rng.randint(0, len(candidates), (1,)).item()]
            mar_predictors[j] = predictor
            if mar_diverse:
                # Draw a diverse single-predictor MAR subtype (clean regime: the predictor is a
                # clean observed/MCAR column), targeting THIS column at ~target_miss_rate.
                subtype, gen = sample_mar_column_generator(
                    j, predictor, rng.spawn(),
                    target_miss_rate=target_miss_rate, strength=mar_strength,
                )
            else:
                # Default Stage 0–4b regime: a single logistic MAR family.
                subtype = "logistic"
                gen = MARLogistic(
                    0, "mixed_mar",
                    GeneratorParams(
                        alpha0=intercept, alpha1=mar_strength,
                        target_col_idx=j, predictor_col_idx=predictor,
                    ),
                )
            mar_subtypes[j] = subtype
            r_full = gen.apply_to(Z, rng.spawn())

        else:  # MNAR — missingness depends on the column's own value.
            if mnar_diverse:
                # Draw a diverse MNAR subtype (self-censoring / threshold / detection),
                # each targeting THIS column at ~target_miss_rate.
                subtype, gen = sample_mnar_column_generator(
                    j, rng.spawn(),
                    target_miss_rate=target_miss_rate, strength=mnar_strength,
                )
            else:
                # Default Stage 0–3 regime: pure logistic self-censoring.
                subtype = "self_censoring"
                gen = MNARLogistic(
                    0, "mixed_mnar",
                    GeneratorParams(
                        beta0=intercept, beta1=0.0, beta2=mnar_strength,
                        target_col_idx=j,
                    ),
                )
            mnar_subtypes[j] = subtype
            r_full = gen.apply_to(Z, rng.spawn())

        R[:, j] = r_full[:, j]

    # Guard: never leave a column fully missing (mirrors apply_missingness per-column
    # safeguard). Forcing one observed cell keeps tokenization well-defined.
    for j in range(d):
        if not bool(R[:, j].any()):
            R[rng.randint(0, n, (1,)).item(), j] = True

    X_observed = X * R.float()
    observed = ObservedDataset(
        x=X_observed,
        r=R,
        n=n,
        d=d,
        feature_names=raw.feature_names,
        dataset_id=f"{raw.name}_mixed",
        meta={"source": raw.source, "is_mixed_mechanism": True},
    )

    miss_rate = (1.0 - R.float().mean(dim=0)).tolist()

    return MixedMissingnessResult(
        observed=observed,
        column_classes=tuple(column_classes),
        per_column_miss_rate=tuple(round(float(m), 4) for m in miss_rate),
        mar_predictors=mar_predictors,
        mnar_subtypes=mnar_subtypes,
        mar_subtypes=mar_subtypes,
        source_name=raw.name,
    )
