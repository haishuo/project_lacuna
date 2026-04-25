"""MAR (Missing At Random) generator family."""

from .simple import (
    MARLogistic,
    MARProbit,
    MARThreshold,
    MARPolynomial,
    MARSpline,
    MARStepFunction,
)
from .multiple import (
    MARMultiColumn,
    MARMultiPredictor,
    MARTwoPredictor,
    MARThreePredictor,
    MARManyPredictor,
    MARWeightedPredictor,
    MARInteractive,
    MARThreeWayInteraction,
    MARConditional,
)
from .strength import (
    MARWeak,
    MARModerate,
    MARStrong,
)
from .predictor_types import (
    MARContinuousPredictor,
    MARDiscretePredictor,
    MARBinaryPredictor,
    MARMixedPredictor,
)
from .complex import (
    MARPolynomialMulti,
    MARSplineMulti,
    MARTreeRules,
    MARKernel,
    MARDistance,
)
from .structural import (
    MARColumnBlocks,
    MARRowBlocks,
    MARNested,
    MARCrossClassified,
)
from .survey import (
    MARSkipLogic,
    MARBranching,
    MARSectionLevel,
    MARRequiredOptional,
    MARQuotaBased,
)
from .realistic import (
    MARRealisticSingle,
    MARPartialResponse,
    MARDemographicGated,
)

__all__ = [
    # simple.py
    "MARLogistic",
    "MARProbit",
    "MARThreshold",
    "MARPolynomial",
    "MARSpline",
    "MARStepFunction",
    # multiple.py
    "MARMultiColumn",
    "MARMultiPredictor",
    "MARTwoPredictor",
    "MARThreePredictor",
    "MARManyPredictor",
    "MARWeightedPredictor",
    "MARInteractive",
    "MARThreeWayInteraction",
    "MARConditional",
    # strength.py
    "MARWeak",
    "MARModerate",
    "MARStrong",
    # predictor_types.py
    "MARContinuousPredictor",
    "MARDiscretePredictor",
    "MARBinaryPredictor",
    "MARMixedPredictor",
    # complex.py
    "MARPolynomialMulti",
    "MARSplineMulti",
    "MARTreeRules",
    "MARKernel",
    "MARDistance",
    # structural.py
    "MARColumnBlocks",
    "MARRowBlocks",
    "MARNested",
    "MARCrossClassified",
    # survey.py
    "MARSkipLogic",
    "MARBranching",
    "MARSectionLevel",
    "MARRequiredOptional",
    "MARQuotaBased",
    # realistic.py
    "MARRealisticSingle",
    "MARPartialResponse",
    "MARDemographicGated",
]
