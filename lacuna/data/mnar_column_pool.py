"""
lacuna.data.mnar_column_pool

Build a DIVERSE per-column MNAR generator targeting one specified column.

`mixed_missingness.compose_mixed_missingness` realises every MNAR column with logistic
self-censoring (`MNARLogistic`) — historically the only MNAR family that supported
`target_col_idx`. Stage 4 showed self-censoring is only one corner of the MNAR space and
that other subtypes (threshold, detection-limit, quantile, truncation) leave very different
per-column footprints. To test whether the Stage 0–3 "per-column MNAR is hard on mixtures"
conclusion is an artefact of using only self-censoring, we need mixtures whose MNAR columns
are drawn from the FULL MNAR subtype space, each targeting its column (ADR-0006).

This module owns exactly that one job: given a target column index and an RNG, pick an MNAR
subtype and return a generator configured to censor that column at approximately the requested
marginal missing rate (so the miss-rate confound stays controlled, matching the composer's
intent). All these families now accept `target_col_idx` (see
`generators.families.mnar._affected_cols`).

Determinism (Coding Bible Rule 6): the subtype choice and all generator randomness flow through
the injected RNGState.

Contract:
    sample_mnar_column_generator(target_col_idx, rng, *, target_miss_rate, strength) -> Generator
The returned generator's `class_id` is always MNAR; `apply_to(Z, rng)` censors only
`target_col_idx`. `target_col_idx` must be a non-negative int (the composer always passes a
concrete column index); failures are loud (Rule 1).
"""

from typing import List, Tuple

from lacuna.core.rng import RNGState
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mnar.self_censoring import MNARLogistic
from lacuna.generators.families.mnar.censoring import (
    MNARThresholdLeft,
    MNARThresholdRight,
    MNARThresholdTwoSided,
    MNARSoftThreshold,
)
from lacuna.generators.families.mnar.detection import (
    MNARDetectionLower,
    MNARDetectionUpper,
    MNARDetectionBoth,
)
import math


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# Subtype names exposed for the composer's bookkeeping / tests. Order is fixed (determinism).
MNAR_SUBTYPES: Tuple[str, ...] = (
    "self_censoring",
    "threshold_left",
    "threshold_right",
    "threshold_two_sided",
    "soft_threshold",
    "detection_lower",
    "detection_upper",
    "detection_both",
)


def _builders(target_col_idx: int, target_miss_rate: float, strength: float) -> List:
    """Return one (subtype_name, generator) per subtype, all targeting `target_col_idx`.

    Percentile-style subtypes are parameterised so their marginal missing rate is ~target_miss_rate:
    a one-sided threshold at percentile (1 - rate) with miss_prob ~1 removes ~rate of the column;
    two-sided splits the rate across both tails. This keeps mechanism TYPE the manipulated variable
    and missing QUANTITY roughly fixed (the composer's existing design goal).
    """
    r = target_miss_rate
    one_sided_pct = (1.0 - r) * 100.0          # e.g. r=0.25 -> 75th percentile
    lo_pct = (r / 2.0) * 100.0                  # two-sided tails
    hi_pct = (1.0 - r / 2.0) * 100.0
    tc = target_col_idx
    return [
        ("self_censoring", MNARLogistic(
            0, "mix_mnar_selfcensor",
            GeneratorParams(beta0=_logit(r), beta1=0.0, beta2=strength, target_col_idx=tc))),
        ("threshold_left", MNARThresholdLeft(
            0, "mix_mnar_threshleft",
            GeneratorParams(percentile=one_sided_pct, miss_prob=0.95,
                            use_absolute=False, target_col_idx=tc))),
        ("threshold_right", MNARThresholdRight(
            0, "mix_mnar_threshright",
            GeneratorParams(percentile=r * 100.0, miss_prob=0.95, target_col_idx=tc))),
        ("threshold_two_sided", MNARThresholdTwoSided(
            0, "mix_mnar_threshtwo",
            GeneratorParams(lower_percentile=lo_pct, upper_percentile=hi_pct,
                            miss_prob=0.95, target_col_idx=tc))),
        ("soft_threshold", MNARSoftThreshold(
            0, "mix_mnar_softthresh",
            GeneratorParams(percentile=one_sided_pct, steepness=3.0, target_col_idx=tc))),
        ("detection_lower", MNARDetectionLower(
            0, "mix_mnar_detlower",
            GeneratorParams(detection_percentile=r * 100.0, target_col_idx=tc))),
        ("detection_upper", MNARDetectionUpper(
            0, "mix_mnar_detupper",
            GeneratorParams(detection_percentile=one_sided_pct, target_col_idx=tc))),
        ("detection_both", MNARDetectionBoth(
            0, "mix_mnar_detboth",
            GeneratorParams(lower_percentile=lo_pct, upper_percentile=hi_pct, target_col_idx=tc))),
    ]


def sample_mnar_column_generator(
    target_col_idx: int,
    rng: RNGState,
    *,
    target_miss_rate: float = 0.25,
    strength: float = 1.5,
) -> Tuple[str, Generator]:
    """Pick a diverse MNAR subtype and return (subtype_name, generator) targeting the column.

    Args:
        target_col_idx: Non-negative index of the column to censor.
        rng: Explicit RNG (subtype choice + generator draws flow through it).
        target_miss_rate: Approximate marginal missing fraction for the column (confound control).
        strength: Self-censoring logistic slope (beta2) when that subtype is drawn.

    Raises:
        ValueError: if target_col_idx is negative/non-int or target_miss_rate not in (0, 1).
    """
    if not isinstance(target_col_idx, int) or isinstance(target_col_idx, bool):
        raise ValueError(f"target_col_idx must be an int, got {target_col_idx!r}")
    if target_col_idx < 0:
        raise ValueError(f"target_col_idx must be >= 0, got {target_col_idx}")
    if not 0.0 < target_miss_rate < 1.0:
        raise ValueError(f"target_miss_rate must be in (0, 1), got {target_miss_rate}")

    builders = _builders(target_col_idx, target_miss_rate, strength)
    pick = rng.randint(0, len(builders), (1,)).item()
    name, gen = builders[pick]
    return name, gen
