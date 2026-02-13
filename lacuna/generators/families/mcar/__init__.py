"""MCAR (Missing Completely At Random) generator family."""

from .bernoulli import MCARBernoulli
from .column_effects import (
    MCARColumnGaussian,
    MCARColumnGamma,
    MCARColumnBeta,
    MCARColumnMixture,
    MCARColumnOrdered,
    MCARColumnClustered,
)
from .row_effects import (
    MCARRowGaussian,
    MCARRowGamma,
    MCARRowBeta,
    MCARRowMixture,
    MCARRowExponential,
    MCARRowDiscrete,
)
from .blocks import (
    MCARRandomBlocks,
    MCARScattered,
    MCARClustered,
    MCARDiagonal,
    MCARCheckerboard,
)
from .distributional import (
    MCARCauchy,
    MCARPareto,
    MCARLogNormal,
    MCARTDist,
    MCARMixtureGaussian,
    MCARSparseMixture,
)
from .multilevel import (
    MCARRowColumnAdditive,
    MCARRowColumnInteraction,
    MCARNested,
)
from .conditional import (
    MCARCovariateDependent,
    MCARBatchEffects,
    MCARSubgroupSpecific,
)

__all__ = [
    # Bernoulli
    "MCARBernoulli",
    # Column effects
    "MCARColumnGaussian",
    "MCARColumnGamma",
    "MCARColumnBeta",
    "MCARColumnMixture",
    "MCARColumnOrdered",
    "MCARColumnClustered",
    # Row effects
    "MCARRowGaussian",
    "MCARRowGamma",
    "MCARRowBeta",
    "MCARRowMixture",
    "MCARRowExponential",
    "MCARRowDiscrete",
    # Block-structured
    "MCARRandomBlocks",
    "MCARScattered",
    "MCARClustered",
    "MCARDiagonal",
    "MCARCheckerboard",
    # Distributional
    "MCARCauchy",
    "MCARPareto",
    "MCARLogNormal",
    "MCARTDist",
    "MCARMixtureGaussian",
    "MCARSparseMixture",
    # Multilevel
    "MCARRowColumnAdditive",
    "MCARRowColumnInteraction",
    "MCARNested",
    # Conditional
    "MCARCovariateDependent",
    "MCARBatchEffects",
    "MCARSubgroupSpecific",
]
