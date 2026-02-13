"""MNAR (Missing Not At Random) generator family."""

from .self_censoring import (
    MNARLogistic,
    MNARSelfCensorHigh,
    MNARSelfCensorLow,
    MNARSelfCensorExtreme,
    MNARSelfCensorWeak,
    MNARSelfCensorStrong,
    MNARValueDependentStrength,
    MNARColumnSpecificCensor,
    MNARDemographicDependent,
)
from .censoring import (
    MNARThresholdLeft,
    MNARThresholdRight,
    MNARThresholdTwoSided,
    MNARSoftThreshold,
    MNARMultiThreshold,
    MNARQuantile70,
    MNARQuantile80,
    MNARQuantile90,
    MNARColumnSpecificThreshold,
)
from .detection import (
    MNARDetectionLower,
    MNARDetectionUpper,
    MNARDetectionBoth,
)
from .social import (
    MNARUnderReport,
    MNAROverReport,
    MNARNonLinearSocial,
)
from .strategic import (
    MNARGaming,
    MNARPrivacy,
    MNARCompetitive,
)
from .latent import (
    MNARLatentHealth,
    MNARLatentSES,
    MNARLatentMotivation,
    MNARLatentMeasurementError,
    MNARLatentOrthogonal,
    MNARLatentCorrelated,
)
from .selection import (
    MNARTruncation,
    MNARBerkson,
    MNARVolunteer,
    MNARAttrition,
    MNARCompetingEvents,
)
from .informative import (
    MNARSymptomTriggered,
    MNARRiskBasedMonitoring,
    MNARAdaptiveSampling,
    MNAROutcomeDependent,
)

__all__ = [
    # Self-censoring (9)
    "MNARLogistic",
    "MNARSelfCensorHigh",
    "MNARSelfCensorLow",
    "MNARSelfCensorExtreme",
    "MNARSelfCensorWeak",
    "MNARSelfCensorStrong",
    "MNARValueDependentStrength",
    "MNARColumnSpecificCensor",
    "MNARDemographicDependent",
    # Censoring/Threshold (8)
    "MNARThresholdLeft",
    "MNARThresholdRight",
    "MNARThresholdTwoSided",
    "MNARSoftThreshold",
    "MNARMultiThreshold",
    "MNARQuantile70",
    "MNARQuantile80",
    "MNARQuantile90",
    "MNARColumnSpecificThreshold",
    # Detection (3)
    "MNARDetectionLower",
    "MNARDetectionUpper",
    "MNARDetectionBoth",
    # Social (3)
    "MNARUnderReport",
    "MNAROverReport",
    "MNARNonLinearSocial",
    # Strategic (3)
    "MNARGaming",
    "MNARPrivacy",
    "MNARCompetitive",
    # Latent (6)
    "MNARLatentHealth",
    "MNARLatentSES",
    "MNARLatentMotivation",
    "MNARLatentMeasurementError",
    "MNARLatentOrthogonal",
    "MNARLatentCorrelated",
    # Selection (5)
    "MNARTruncation",
    "MNARBerkson",
    "MNARVolunteer",
    "MNARAttrition",
    "MNARCompetingEvents",
    # Informative (4)
    "MNARSymptomTriggered",
    "MNARRiskBasedMonitoring",
    "MNARAdaptiveSampling",
    "MNAROutcomeDependent",
]
