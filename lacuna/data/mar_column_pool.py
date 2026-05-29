"""
lacuna.data.mar_column_pool

Build a DIVERSE per-column MAR generator targeting one specified column, with its missingness
driven by ONE specified (clean) predictor column.

`mixed_missingness.compose_mixed_missingness` historically realised every MAR column with a single
family (`MARLogistic`). That is the MAR analogue of the self-censoring MNAR monoculture that
Stage 4b showed was responsible for the per-column instability. To test per-column detectability at
full subtype diversity on true mixtures (Stage 5, ADR-0006), this pool draws a MAR column's
mechanism from the broad set of SINGLE-PREDICTOR MAR families — the ones that accept both
`target_col_idx` (which column goes missing) and `predictor_col_idx` (which clean column drives it),
so the clean-MAR regime is preserved (the predictor is a fully-observed / MCAR column).

Curation — single-predictor, rate-controllable MAR
--------------------------------------------------
Included (all accept target + predictor, all parameterisable to ≈ target_miss_rate on a
standard-normal predictor view):

  - Logistic, Probit            — linear logit / probit link.
  - Polynomial                  — quadratic logit (nonlinear in the predictor).
  - Threshold, StepFunction     — piecewise-constant rate in the predictor.
  - BinaryPredictor             — median-split predictor → two rates.
  - DiscretePredictor           — quantile-binned predictor → per-bin rates.
  - RealisticSingle             — z-scored-predictor logistic (the survey-realistic single-col MAR).

Excluded: multi-predictor families (MultiCol/ManyPred/TwoPred/Interactive/...), structural /
skip-logic families (Section/SkipLogic/ColBlocks/...) whose predictor set is not a single clean
column, and Kernel (its sigmoid(scale·RBF) form floors the marginal rate near 0.5, so it cannot be
tuned to ~0.25). These remain available in the registry; they are simply not drawn here.

Determinism (Coding Bible Rule 6): the subtype choice and all generator randomness flow through the
injected RNGState.

Contract:
    sample_mar_column_generator(target_col_idx, predictor_col_idx, rng, *, target_miss_rate,
                                strength) -> (name, Generator)
The returned generator's `class_id` is always MAR; `apply_to(Z, rng)` censors only
`target_col_idx`, driven by `predictor_col_idx`. Both indices must be non-negative ints and
distinct; failures are loud (Rule 1).
"""

import math
from typing import List, Tuple

from lacuna.core.rng import RNGState
from lacuna.data._column_pool_math import comp_beta0 as _comp_beta0, norm_ppf as _norm_ppf
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mar.simple import (
    MARLogistic, MARProbit, MARThreshold, MARPolynomial, MARStepFunction,
)
from lacuna.generators.families.mar.predictor_types import (
    MARBinaryPredictor, MARDiscretePredictor,
)
from lacuna.generators.families.mar.realistic import MARRealisticSingle


# Subtype names exposed for the composer's bookkeeping / tests. Order is fixed (determinism).
MAR_SUBTYPES: Tuple[str, ...] = (
    "mar_logistic",
    "mar_probit",
    "mar_polynomial",
    "mar_threshold",
    "mar_step",
    "mar_binary",
    "mar_discrete",
    "mar_realistic",
)


def _builders(target_col_idx: int, predictor_col_idx: int, target_miss_rate: float,
              strength: float) -> List[Tuple[str, Generator]]:
    """Return one (subtype_name, generator) per MAR subtype, all targeting `target_col_idx` with
    predictor `predictor_col_idx`, each parameterised for marginal rate ≈ target_miss_rate.

    Rates are set for a standard-normal predictor view (the composer feeds z-scored X):
    logit/probit use the analytic intercept; threshold/step/binary/discrete split the rate around
    a median/quantile so the marginal averages to ~r. Guarded by `test_mar_column_pool.py`.
    """
    r = target_miss_rate
    s = strength
    t = target_col_idx
    p = predictor_col_idx
    hi = min(0.95, 1.5 * r)              # above-median / high-predictor rate
    lo = max(0.02, 0.5 * r)             # below-median / low-predictor rate
    common = dict(target_col_idx=t, predictor_col_idx=p)
    return [
        ("mar_logistic", MARLogistic(
            0, "mix_mar_logistic",
            GeneratorParams(alpha0=_comp_beta0(r, s), alpha1=s, **common))),
        ("mar_probit", MARProbit(
            0, "mix_mar_probit",
            # Probit marginal is exactly Phi(alpha0 / sqrt(1 + alpha1^2)); choose alpha0 to hit r.
            GeneratorParams(alpha0=_norm_ppf(r) * math.sqrt(1.0 + s * s), alpha1=s, **common))),
        ("mar_polynomial", MARPolynomial(
            0, "mix_mar_poly",
            # Quadratic term lifts the marginal; pull the intercept down to re-centre near r.
            GeneratorParams(alpha0=_comp_beta0(r, s) - 0.45, alpha1=s, alpha2=0.3, **common))),
        ("mar_threshold", MARThreshold(
            0, "mix_mar_threshold",
            GeneratorParams(threshold=0.0, miss_prob_above=hi, miss_prob_below=lo, **common))),
        ("mar_step", MARStepFunction(
            0, "mix_mar_step",
            # Two segments around the predictor median: high-predictor rows miss more.
            GeneratorParams(thresholds=[-1.0, 0.0, 1.0], rates=[lo, hi], **common))),
        ("mar_binary", MARBinaryPredictor(
            0, "mix_mar_binary",
            GeneratorParams(miss_rate_high=hi, miss_rate_low=lo, **common))),
        ("mar_discrete", MARDiscretePredictor(
            0, "mix_mar_discrete",
            GeneratorParams(n_levels=4,
                            alpha_per_level=[0.5 * r, 0.83 * r, 1.17 * r, 1.5 * r], **common))),
        ("mar_realistic", MARRealisticSingle(
            0, "mix_mar_realistic",
            GeneratorParams(target_miss_rate=r, slope=s, **common))),
    ]


def sample_mar_column_generator(
    target_col_idx: int,
    predictor_col_idx: int,
    rng: RNGState,
    *,
    target_miss_rate: float = 0.25,
    strength: float = 1.5,
) -> Tuple[str, Generator]:
    """Pick a diverse MAR subtype and return (subtype_name, generator) targeting the column.

    Args:
        target_col_idx: Non-negative index of the column to censor.
        predictor_col_idx: Non-negative index of the clean predictor column (distinct from target).
        rng: Explicit RNG (subtype choice + generator draws flow through it).
        target_miss_rate: Approximate marginal missing fraction for the column (confound control).
        strength: Logistic/probit slope coupling the column to its predictor.

    Raises:
        ValueError: if either index is negative/non-int, the two are equal, or rate not in (0, 1).
    """
    for label, idx in (("target_col_idx", target_col_idx), ("predictor_col_idx", predictor_col_idx)):
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError(f"{label} must be an int, got {idx!r}")
        if idx < 0:
            raise ValueError(f"{label} must be >= 0, got {idx}")
    if target_col_idx == predictor_col_idx:
        raise ValueError(
            f"MAR predictor must differ from target (clean regime), got both = {target_col_idx}")
    if not 0.0 < target_miss_rate < 1.0:
        raise ValueError(f"target_miss_rate must be in (0, 1), got {target_miss_rate}")

    builders = _builders(target_col_idx, predictor_col_idx, target_miss_rate, strength)
    pick = rng.randint(0, len(builders), (1,)).item()
    name, gen = builders[pick]
    return name, gen
