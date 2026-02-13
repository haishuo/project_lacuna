"""
lacuna.generators.families

Generator family implementations.

Families:
- MCAR: Missing Completely At Random
- MAR: Missing At Random
- MNAR: Missing Not At Random

Individual classes are auto-discovered by registry_builder.py.
This module re-exports all classes for convenience.
"""

from .base_data import (
    sample_gaussian,
    sample_gaussian_correlated,
    sample_uniform,
    sample_mixed,
)

# Re-export all from each family
from .mcar import *  # noqa: F401,F403
from .mar import *  # noqa: F401,F403
from .mnar import *  # noqa: F401,F403

# Explicit list of base data samplers
__all__ = [
    "sample_gaussian",
    "sample_gaussian_correlated",
    "sample_uniform",
    "sample_mixed",
]
