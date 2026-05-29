"""
lacuna.data.mnar_column_pool

Build a DIVERSE per-column MNAR generator targeting one specified column.

`mixed_missingness.compose_mixed_missingness` realised every MNAR column with logistic
self-censoring (`MNARLogistic`) historically. Stage 4 showed self-censoring is only one corner
of the MNAR space and that other subtypes (threshold, detection-limit, social-desirability,
strategic) leave very different per-column footprints. To test whether per-column MNAR
detectability holds at FULL subtype diversity on true mixtures (Stage 5, ADR-0006), this pool
draws an MNAR column's mechanism from the BROAD set of per-column-targetable MNAR families.

Curation — CLEAN per-column MNAR only (own-value dependence)
-----------------------------------------------------------
Every MNAR family is now per-column-targetable (commit `feat(generators): per-column
target_col_idx for all remaining MNAR families`). But targeting a *single* column only yields a
genuine per-column MNAR label when the column's missingness depends on the column's OWN value.
Families whose targeted missingness is driven by OTHER columns / row aggregates / sequence /
within-column-constant rates are MAR-adjacent or degenerate per column, and including them as
"MNAR" would inject label noise — exactly the confound the clean-MAR regime exists to avoid. They
are therefore EXCLUDED from this pool (the capability exists for registry uniformity, not here):

  - Truncation, Berkson         — select on another column / the row sum (MAR-adjacent).
  - RiskBasedMonitoring         — driven by the row's |value| sum (MAR-adjacent).
  - OutcomeDependent            — driven by another column (MAR-adjacent).
  - AdaptiveSampling            — per-column rate constant across rows (MCAR-like per column).
  - Attrition                   — sequential across columns (not a single-column mechanism).
  - LatentHealth/SES/...        — apply_to approximates the latent as the row mean (MAR-adjacent).
  - ModuleRefusal               — a column battery; one-item module is redundant with self-censor.
  - Q90                         — fixed 90th-|value| percentile caps the marginal rate at ~0.10.
  - Competitive                 — identical mechanism to Gaming (near-a-value); redundant.

What remains is a 25-subtype pool spanning the full easy→hard footprint spectrum: sharp
threshold/detection (easiest), social/strategic value-censoring (medium), smooth logistic
self-censoring (hardest), plus center-censoring (Gaming/Volunteer) and inverse-MNAR
(SymptomTriggered). Every subtype is parameterised so its marginal missing rate is ≈
`target_miss_rate` (confound control: mechanism TYPE varies, missing QUANTITY held ~fixed). The
realised rates are guarded by `tests/unit/data/test_mnar_column_pool.py`.

Determinism (Coding Bible Rule 6): the subtype choice and all generator randomness flow through
the injected RNGState.

Contract:
    sample_mnar_column_generator(target_col_idx, rng, *, target_miss_rate, strength) -> (name, Generator)
The returned generator's `class_id` is always MNAR; `apply_to(Z, rng)` censors only
`target_col_idx`. `target_col_idx` must be a non-negative int; failures are loud (Rule 1).
"""

from typing import List, Tuple

from lacuna.core.rng import RNGState
from lacuna.data._column_pool_math import logit as _logit, comp_beta0 as _comp_beta0, norm_ppf as _norm_ppf
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mnar.self_censoring import (
    MNARLogistic, MNARSelfCensorHigh, MNARSelfCensorLow, MNARSelfCensorExtreme,
    MNARSelfCensorWeak, MNARSelfCensorStrong, MNARValueDependentStrength,
    MNARColumnSpecificCensor, MNARDemographicDependent,
)
from lacuna.generators.families.mnar.censoring import (
    MNARThresholdLeft, MNARThresholdRight, MNARThresholdTwoSided, MNARSoftThreshold,
    MNARColumnSpecificThreshold,
)
from lacuna.generators.families.mnar.detection import (
    MNARDetectionLower, MNARDetectionUpper, MNARDetectionBoth,
)
from lacuna.generators.families.mnar.social import (
    MNARUnderReport, MNAROverReport, MNARNonLinearSocial,
)
from lacuna.generators.families.mnar.strategic import MNARGaming, MNARPrivacy
from lacuna.generators.families.mnar.informative import MNARSymptomTriggered
from lacuna.generators.families.mnar.selection import MNARVolunteer, MNARCompetingEvents


# Subtype names exposed for the composer's bookkeeping / tests. Order is fixed (determinism).
MNAR_SUBTYPES: Tuple[str, ...] = (
    # --- logistic self-censoring family (sigmoid on own value; hardest footprint) ---
    "self_censoring",
    "selfcensor_high",
    "selfcensor_low",
    "selfcensor_extreme",
    "selfcensor_weak",
    "selfcensor_strong",
    "value_dep_strength",
    "col_specific_censor",
    "demographic_dependent",
    # --- hard/soft threshold family (sharp footprint; easiest) ---
    "threshold_left",
    "threshold_right",
    "threshold_two_sided",
    "soft_threshold",
    "col_specific_thresh",
    # --- detection-limit family ---
    "detection_lower",
    "detection_upper",
    "detection_both",
    # --- social-desirability family ---
    "under_report",
    "over_report",
    "nonlinear_social",
    # --- strategic / privacy ---
    "gaming",
    "privacy",
    # --- informative observation (inverse MNAR) ---
    "symptom_triggered",
    # --- selection (own-value) ---
    "volunteer",
    "competing_events",
)


def _builders(target_col_idx: int, target_miss_rate: float, strength: float) -> List[Tuple[str, Generator]]:
    """Return one (subtype_name, generator) per subtype, all targeting `target_col_idx`.

    Sigmoid-on-own-value subtypes use `_comp_beta0` so their marginal rate ≈ target_miss_rate.
    Percentile subtypes set their percentile/miss_prob directly for ≈ target_miss_rate. The
    center-censoring (gaming/volunteer) and inverse (symptom_triggered) subtypes are tuned for the
    standard-normal predictor view the composer feeds (z-scored X). See module docstring.
    """
    r = target_miss_rate
    tc = target_col_idx
    s = strength
    weak_s = min(abs(s), 1.4) if abs(s) > 0 else 1.0      # MNARSelfCensorWeak needs 0 < |b1| <= 1.5
    strong_s = max(abs(s), 2.6)                            # MNARSelfCensorStrong needs |b1| >= 2.5
    one_sided_pct = (1.0 - r) * 100.0                      # e.g. r=0.25 -> 75th percentile
    lo_pct = (r / 2.0) * 100.0                             # two-sided tails
    hi_pct = (1.0 - r / 2.0) * 100.0
    # Gaming/volunteer censor the CENTRE: pick a radius capturing ~r of standard-normal mass.
    center_radius = abs(_norm_ppf(0.5 + r / 2.0))

    return [
        ("self_censoring", MNARLogistic(
            0, "mix_mnar_selfcensor",
            GeneratorParams(beta0=_comp_beta0(r, s), beta1=0.0, beta2=s, target_col_idx=tc))),
        ("selfcensor_high", MNARSelfCensorHigh(
            0, "mix_mnar_sc_high",
            GeneratorParams(beta0=_comp_beta0(r, s), beta1=s, target_col_idx=tc))),
        ("selfcensor_low", MNARSelfCensorLow(
            0, "mix_mnar_sc_low",
            GeneratorParams(beta0=_comp_beta0(r, s), beta1=-s, target_col_idx=tc))),
        ("selfcensor_extreme", MNARSelfCensorExtreme(
            0, "mix_mnar_sc_extreme",
            GeneratorParams(beta0=_logit(r) - 0.55, beta_quadratic=0.8, target_col_idx=tc))),
        ("selfcensor_weak", MNARSelfCensorWeak(
            0, "mix_mnar_sc_weak",
            GeneratorParams(beta0=_comp_beta0(r, weak_s), beta1=weak_s, target_col_idx=tc))),
        ("selfcensor_strong", MNARSelfCensorStrong(
            0, "mix_mnar_sc_strong",
            GeneratorParams(beta0=_comp_beta0(r, strong_s), beta1=strong_s, target_col_idx=tc))),
        ("value_dep_strength", MNARValueDependentStrength(
            0, "mix_mnar_valdep",
            # Both tails censor (low values use -s, high use +s), so the effective rate is higher
            # than a single-slope sigmoid — push the intercept down to re-centre near r.
            GeneratorParams(beta0=_comp_beta0(r, s) - 0.6, beta1_low=-s, beta1_high=s, target_col_idx=tc))),
        ("col_specific_censor", MNARColumnSpecificCensor(
            0, "mix_mnar_colspec",
            GeneratorParams(beta0_range=[_comp_beta0(r, s) - 0.3, _comp_beta0(r, s) + 0.3],
                            beta1_range=[max(0.5, s - 0.5), s + 0.5], target_col_idx=tc))),
        ("demographic_dependent", MNARDemographicDependent(
            0, "mix_mnar_demo",
            GeneratorParams(n_groups=2,
                            beta_per_group=[(_comp_beta0(r, s), s), (_comp_beta0(r, 0.6 * s), 0.6 * s)],
                            target_col_idx=tc))),
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
        ("col_specific_thresh", MNARColumnSpecificThreshold(
            0, "mix_mnar_colspecthresh",
            GeneratorParams(percentile_range=[(1.0 - 1.4 * r) * 100.0, (1.0 - 0.6 * r) * 100.0],
                            miss_prob=0.95, target_col_idx=tc))),
        ("detection_lower", MNARDetectionLower(
            0, "mix_mnar_detlower",
            GeneratorParams(detection_percentile=r * 100.0, target_col_idx=tc))),
        ("detection_upper", MNARDetectionUpper(
            0, "mix_mnar_detupper",
            GeneratorParams(detection_percentile=one_sided_pct, target_col_idx=tc))),
        ("detection_both", MNARDetectionBoth(
            0, "mix_mnar_detboth",
            GeneratorParams(lower_percentile=lo_pct, upper_percentile=hi_pct, target_col_idx=tc))),
        ("under_report", MNARUnderReport(
            0, "mix_mnar_underrep",
            GeneratorParams(threshold_percentile=one_sided_pct, under_report_prob=0.95, target_col_idx=tc))),
        ("over_report", MNAROverReport(
            0, "mix_mnar_overrep",
            GeneratorParams(threshold_percentile=r * 100.0, over_report_prob=0.95, target_col_idx=tc))),
        ("nonlinear_social", MNARNonLinearSocial(
            0, "mix_mnar_nonlinsoc",
            # logits = sensitivity*(X)^2 - 1; the fixed -1 floors the rate ~0.27, so keep
            # sensitivity small to stay near r rather than over-censoring the tails.
            GeneratorParams(center_value=0.0, sensitivity=0.25, target_col_idx=tc))),
        ("gaming", MNARGaming(
            0, "mix_mnar_gaming",
            GeneratorParams(incentive_threshold=0.0, gaming_radius=center_radius,
                            miss_prob=0.95, target_col_idx=tc))),
        ("privacy", MNARPrivacy(
            0, "mix_mnar_privacy",
            GeneratorParams(privacy_threshold=one_sided_pct, miss_prob=0.95, target_col_idx=tc))),
        ("symptom_triggered", MNARSymptomTriggered(
            0, "mix_mnar_symptom",
            GeneratorParams(symptom_threshold=85.0, trigger_prob=0.2, target_col_idx=tc))),
        ("volunteer", MNARVolunteer(
            0, "mix_mnar_volunteer",
            # P(observe) = sigmoid(tendency * X^2): extremes observed, centre censored. Higher
            # tendency observes more, lowering the marginal rate toward r.
            GeneratorParams(volunteer_tendency=1.6, target_col_idx=tc))),
        ("competing_events", MNARCompetingEvents(
            0, "mix_mnar_compevents",
            GeneratorParams(event_threshold=one_sided_pct, event_prob=0.95, target_col_idx=tc))),
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
        strength: Logistic slope for the sigmoid-on-own-value subtypes (self-censoring etc.).

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
