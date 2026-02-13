"""
lacuna.generators

Generator system for synthetic data with controlled missingness mechanisms.
"""

from .base import Generator
from .params import GeneratorParams
from .registry import GeneratorRegistry
from .priors import GeneratorPrior

from .families import (
    # Base data
    sample_gaussian,
    sample_gaussian_correlated,
    sample_uniform,
    sample_mixed,
    # MCAR
    MCARBernoulli,
    MCARColumnGaussian,
    # MAR
    MARLogistic,
    MARMultiColumn,
    MARMultiPredictor,
    # MNAR
    MNARLogistic,
    MNARSelfCensorHigh,
    MNARThresholdLeft,
)

from .families.registry_builder import (
    load_registry_from_config,
    load_registry_from_yaml,
    list_available_configs,
)

__all__ = [
    "Generator",
    "GeneratorParams",
    "GeneratorRegistry",
    "GeneratorPrior",
    "load_registry_from_config",
    "load_registry_from_yaml",
    "list_available_configs",
    "sample_gaussian",
    "sample_gaussian_correlated",
    "sample_uniform",
    "sample_mixed",
    "MCARBernoulli",
    "MCARColumnGaussian",
    "MARLogistic",
    "MARMultiColumn",
    "MARMultiPredictor",
    "MNARLogistic",
    "MNARSelfCensorHigh",
    "MNARThresholdLeft",
]
