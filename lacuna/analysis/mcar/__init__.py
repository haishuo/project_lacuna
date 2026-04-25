"""Lacuna-local MCAR tests.

These tests were motivated by Lacuna's mcar-alternatives-bakeoff and
are not general-purpose statistical tools; they live in Lacuna rather
than pystatistics per the scope-boundary rule in CLAUDE.md §8. The
canonical Little (1988) MLE-plug-in test remains in pystatistics as
``pystatistics.mvnmle.little_mcar_test`` — that one IS textbook and
belongs in a general-purpose statistics package.

Exports:

- ``mom_mcar_test`` — pairwise-deletion method-of-moments plug-in, a
  faster chi-square variant of Little's that trades asymptotic
  efficiency for cache-scale throughput.
- ``propensity_mcar_test`` — supervised detection of non-MCAR via a
  classifier's ability to predict the missingness indicator from the
  observed values.
- ``hsic_mcar_test`` — kernel independence test (Hilbert-Schmidt
  independence criterion) between observed values and missingness
  indicators.
- ``missmech_mcar_test`` — Jamshidian-Jalal-style nonparametric test
  of homogeneity of means across missingness patterns.
- ``NonparametricMCARResult`` — shared result dataclass for the three
  distribution-free tests (propensity / HSIC / MissMech). The
  MoM test returns ``pystatistics.mvnmle.MCARTestResult`` to match
  Little's API.
"""

from lacuna.analysis.mcar.result import NonparametricMCARResult
from lacuna.analysis.mcar.mom import mom_mcar_test
from lacuna.analysis.mcar.propensity import propensity_mcar_test
from lacuna.analysis.mcar.hsic import hsic_mcar_test
from lacuna.analysis.mcar.missmech import missmech_mcar_test

__all__ = [
    "NonparametricMCARResult",
    "mom_mcar_test",
    "propensity_mcar_test",
    "hsic_mcar_test",
    "missmech_mcar_test",
]
