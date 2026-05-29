"""
Tests for per-column targeting of the REMAINING MNAR families (ADR-0006 / column-level deployment).

`test_mnar_targeting.py` already covers the threshold (`censoring.py`) and detection (`detection.py`)
families. This file covers the families that gained `target_col_idx` afterwards:

  - self-censoring (`self_censoring.py`): High/Low/Extreme/Weak/Strong, ValueDependentStrength,
    ColumnSpecificCensor, DemographicDependent  — own-value, via `resolve_affected_cols`.
  - selection (`selection.py`): Truncation/Berkson/Volunteer/CompetingEvents (resolve), Attrition
    (restrict).
  - social (`social.py`): UnderReport/OverReport/NonLinearSocial (resolve), ModuleRefusal
    (single-item module).
  - strategic (`strategic.py`): Gaming/Privacy/Competitive (resolve).
  - informative (`informative.py`): SymptomTriggered/RiskBasedMonitoring (resolve), OutcomeDependent
    (guarded), AdaptiveSampling (guarded loop).
  - latent (`latent.py`): all 6 whole-matrix families, via `restrict_to_target_col`.

Contract under test: with `target_col_idx=k`, ONLY column k may carry missingness (every family);
negative indices wrap; targeting is deterministic. The DEFAULT (no target) path is guarded
bit-for-bit by `test_registry_bit_identical.py`, so here we focus on the targeted behaviour.

Normal / edge / failure per Coding Bible Rule 7.
"""

import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mnar._affected_cols import restrict_to_target_col
from lacuna.generators.families.mnar.self_censoring import (
    MNARSelfCensorHigh, MNARSelfCensorLow, MNARSelfCensorExtreme, MNARSelfCensorWeak,
    MNARSelfCensorStrong, MNARValueDependentStrength, MNARColumnSpecificCensor,
    MNARDemographicDependent,
)
from lacuna.generators.families.mnar.selection import (
    MNARTruncation, MNARBerkson, MNARVolunteer, MNARCompetingEvents, MNARAttrition,
)
from lacuna.generators.families.mnar.social import (
    MNARUnderReport, MNAROverReport, MNARNonLinearSocial, MNARModuleRefusal,
)
from lacuna.generators.families.mnar.strategic import MNARGaming, MNARPrivacy, MNARCompetitive
from lacuna.generators.families.mnar.informative import (
    MNARSymptomTriggered, MNARRiskBasedMonitoring, MNARAdaptiveSampling, MNAROutcomeDependent,
)
from lacuna.generators.families.mnar.latent import (
    MNARLatentHealth, MNARLatentSES, MNARLatentMotivation, MNARLatentMeasurementError,
    MNARLatentOrthogonal, MNARLatentCorrelated,
)


# Each entry: (class, required/strong params) chosen so the targeted column reliably goes
# partially (not fully) missing on a standard-normal 600x6 matrix.
_NEWLY_TARGETABLE = [
    # self-censoring (own-value)
    (MNARSelfCensorHigh, dict(beta0=-0.5, beta1=2.0)),
    (MNARSelfCensorLow, dict(beta0=-0.5, beta1=-2.0)),
    (MNARSelfCensorExtreme, dict(beta0=-1.0, beta_quadratic=1.5)),
    (MNARSelfCensorWeak, dict(beta0=-0.5, beta1=1.0)),
    (MNARSelfCensorStrong, dict(beta0=-0.5, beta1=3.0)),
    (MNARValueDependentStrength, dict(beta0=-0.5, beta1_low=-2.0, beta1_high=2.0)),
    (MNARColumnSpecificCensor, dict(beta0_range=[-0.5, 0.5], beta1_range=[1.0, 3.0])),
    (MNARDemographicDependent, dict(n_groups=2, beta_per_group=[(-0.5, 2.0), (-0.5, 1.0)])),
    # selection
    (MNARTruncation, dict(selection_threshold=50, miss_prob=0.9)),
    (MNARBerkson, dict(selection_strength=1.0)),
    (MNARVolunteer, dict(volunteer_tendency=0.5)),
    (MNARCompetingEvents, dict(event_threshold=50, event_prob=0.9)),
    (MNARAttrition, dict(attrition_rate=0.5, value_dependence=1.5)),
    # social
    (MNARUnderReport, dict(threshold_percentile=50, under_report_prob=0.9)),
    (MNAROverReport, dict(threshold_percentile=50, over_report_prob=0.9)),
    (MNARNonLinearSocial, dict(center_value=0.0, sensitivity=1.5)),
    (MNARModuleRefusal, dict(module_frac=0.5, baseline_refusal=0.3, selection_strength=1.5)),
    # strategic
    (MNARGaming, dict(incentive_threshold=0.0, gaming_radius=0.8, miss_prob=0.9)),
    (MNARPrivacy, dict(privacy_threshold=50, miss_prob=0.9)),
    (MNARCompetitive, dict(benchmark_value=0.0, competitive_radius=0.8, miss_prob=0.9)),
    # informative
    (MNARSymptomTriggered, dict(symptom_threshold=50, trigger_prob=0.9)),
    (MNARRiskBasedMonitoring, dict(base_miss_prob=0.6, monitoring_boost=0.5)),
    (MNARAdaptiveSampling, dict(adaptation_rate=1.0, baseline_prob=0.5)),
    (MNAROutcomeDependent, dict(beta0=0.0, beta1=2.0, outcome_col=0)),
    # latent (whole-matrix; restrict)
    (MNARLatentHealth, dict(latent_strength=2.0, obs_strength=1.0)),
    (MNARLatentSES, dict(latent_strength=2.0, obs_strength=1.0)),
    (MNARLatentMotivation, dict(latent_strength=2.0, obs_strength=1.0)),
    (MNARLatentMeasurementError, dict(latent_strength=2.0, obs_strength=1.0)),
    (MNARLatentOrthogonal, dict(n_factors=2, strengths=2.0)),
    (MNARLatentCorrelated, dict(n_factors=2, correlation=0.5, strengths=2.0)),
]


@pytest.fixture
def X():
    """A reproducible 600x6 standard-normal matrix."""
    return RNGState(seed=2026).randn(600, 6)


def _missing_cols(R: torch.Tensor) -> list:
    return (~R.all(dim=0)).nonzero().flatten().tolist()


# =============================================================================
# restrict_to_target_col — the whole-matrix targeting helper
# =============================================================================

class TestRestrictToTargetCol:
    def test_no_target_is_identity(self):
        R = RNGState(seed=1).rand(50, 4) > 0.5
        out = restrict_to_target_col(R, GeneratorParams(), d=4)
        assert torch.equal(out, R)  # literal no-op when target absent

    def test_target_keeps_only_that_column(self):
        R = torch.zeros(50, 4, dtype=torch.bool)  # everything "missing"
        out = restrict_to_target_col(R, GeneratorParams(target_col_idx=2), d=4)
        # Columns 0,1,3 forced observed (all True); column 2 keeps its (all-False) mask.
        assert out[:, [0, 1, 3]].all()
        assert not out[:, 2].any()

    def test_negative_index_wraps(self):
        R = torch.zeros(10, 5, dtype=torch.bool)
        out = restrict_to_target_col(R, GeneratorParams(target_col_idx=-1), d=5)
        assert not out[:, 4].any()
        assert out[:, :4].all()

    def test_out_of_range_wraps_modulo(self):
        R = torch.zeros(10, 5, dtype=torch.bool)
        out = restrict_to_target_col(R, GeneratorParams(target_col_idx=6), d=5)
        assert not out[:, 1].any()  # 6 % 5 == 1

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError, match="d >= 1"):
            restrict_to_target_col(torch.ones(2, 0, dtype=torch.bool), GeneratorParams(target_col_idx=0), d=0)

    def test_non_int_target_raises(self):
        with pytest.raises(ValueError, match="must be an int"):
            restrict_to_target_col(torch.ones(2, 3, dtype=torch.bool), GeneratorParams(target_col_idx=1.5), d=3)


# =============================================================================
# Per-family targeting behaviour (all newly-targetable MNAR generators)
# =============================================================================

class TestExtendedFamilyTargeting:
    @pytest.mark.parametrize("gen_cls,extra", _NEWLY_TARGETABLE)
    def test_only_target_column_missing(self, gen_cls, extra, X):
        """With target_col_idx=2, missingness is confined to column 2 (subset, never more)."""
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=2, **extra))
        R = gen.apply_to(X, RNGState(seed=7))
        assert set(_missing_cols(R)) <= {2}, f"{gen_cls.__name__}: {_missing_cols(R)}"

    @pytest.mark.parametrize("gen_cls,extra", _NEWLY_TARGETABLE)
    def test_targeted_column_loses_some_values(self, gen_cls, extra, X):
        """The targeted column actually goes partially missing (the rule fired, not fully missing)."""
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=2, **extra))
        R = gen.apply_to(X, RNGState(seed=7))
        obs = R[:, 2].float().mean().item()
        assert 0.0 < obs < 1.0, f"{gen_cls.__name__}: obs frac {obs}"

    @pytest.mark.parametrize("gen_cls,extra", _NEWLY_TARGETABLE)
    def test_negative_target_wraps_to_last(self, gen_cls, extra, X):
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=-1, **extra))
        R = gen.apply_to(X, RNGState(seed=7))
        assert set(_missing_cols(R)) <= {5}, f"{gen_cls.__name__}: {_missing_cols(R)}"

    @pytest.mark.parametrize("gen_cls,extra", _NEWLY_TARGETABLE)
    def test_targeting_is_deterministic(self, gen_cls, extra, X):
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=1, **extra))
        R1 = gen.apply_to(X, RNGState(seed=8))
        R2 = gen.apply_to(X, RNGState(seed=8))
        assert torch.equal(R1, R2)

    @pytest.mark.parametrize("gen_cls,extra", _NEWLY_TARGETABLE)
    def test_class_id_is_mnar(self, gen_cls, extra):
        gen = gen_cls(0, "t", GeneratorParams(target_col_idx=0, **extra))
        assert gen.class_id == MNAR


# =============================================================================
# Family-specific targeting nuances
# =============================================================================

def test_outcome_dependent_shifts_outcome_when_target_equals_outcome(X):
    """If target == outcome_col, the outcome shifts so cross-column dependence is preserved
    (the targeted column still goes missing rather than driving itself trivially)."""
    gen = MNAROutcomeDependent(0, "t", GeneratorParams(beta0=0.0, beta1=2.0, outcome_col=2, target_col_idx=2))
    R = gen.apply_to(X, RNGState(seed=7))
    assert set(_missing_cols(R)) <= {2}
    assert 0.0 < R[:, 2].float().mean().item() < 1.0


def test_module_refusal_collapses_to_single_item_module(X):
    """ModuleRefusal under target_col_idx censors only the targeted column (one-item module)."""
    gen = MNARModuleRefusal(0, "t", GeneratorParams(
        module_frac=0.5, baseline_refusal=0.3, selection_strength=2.0, target_col_idx=3))
    R = gen.apply_to(X, RNGState(seed=7))
    assert set(_missing_cols(R)) <= {3}
    assert 0.0 < R[:, 3].float().mean().item() < 1.0


def test_latent_targeting_confines_to_one_column(X):
    """Latent families are whole-matrix by default; under targeting only one column is censored."""
    default = MNARLatentHealth(0, "t", GeneratorParams(latent_strength=2.0, obs_strength=1.0))
    R_default = default.apply_to(X, RNGState(seed=7))
    targeted = MNARLatentHealth(0, "t", GeneratorParams(latent_strength=2.0, obs_strength=1.0, target_col_idx=2))
    R_targeted = targeted.apply_to(X, RNGState(seed=7))
    assert len(_missing_cols(R_default)) >= 2, "default latent should hit multiple columns"
    assert set(_missing_cols(R_targeted)) <= {2}
