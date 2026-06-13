"""
lacuna.analysis.realism_gate

The L1 realism gate (GENERATOR-DESIGN-charter §6): measure whether a generator's
injected missingness masks match real survey missingness.

Pipeline (one module, one job):
- ``mask_stats``  : binary mask -> row features (for C2ST) + footprint statistics.
- ``corpora``     : raw ESS / NHANES corpora -> ``MaskCorpus`` (real masks +
                    complete-case real X substrate, block-aware).
- ``c2st``        : classifier two-sample test + finite-sample p-values.
- ``fidelity``    : Dankar fidelity panel + Alaa authenticity guard.
- ``gate``        : orchestrate one (generator, corpus) -> ``RealismGateResult``.

This package is OFFLINE analysis. It is never imported from a training forward
pass. All randomness flows through injected ``numpy.random.Generator`` instances.
"""

from .gate import RealismGateResult, run_realism_gate, format_gate_table
from .corpora import MaskCorpus, load_ess_corpus, load_nhanes_corpus

__all__ = [
    "RealismGateResult",
    "run_realism_gate",
    "format_gate_table",
    "MaskCorpus",
    "load_ess_corpus",
    "load_nhanes_corpus",
]
