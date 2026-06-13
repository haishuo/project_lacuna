"""
lacuna.feasibility

Isolated feasibility-probe package for the P1 own-value self-censoring experiment
(docs/proposals/PROPOSAL-P1-implementation-audit.md). This package does NOT modify or import
the production generator registry or model heads — it is a self-contained probe.

Modules:
- delta_generator: the δ ≡ β₂ self-censoring generator + deterministic β₀ rate solver.
- xmodel:          X-model interface for the oracle (ConditionalGaussian: exact-synthetic / fitted-real).
- oracle:          the analytic Bayes-optimal oracle — the ONLY object permitted to be a "ceiling".
- manifest:        run-manifest builder/validator (no metric is interpretable without it).
- sweep:           two-stage oracle-surface orchestration (coarse → boundary refinement).

Vocabulary (charter §4.9): only the analytic Bayes oracle is a "ceiling", under its stated
X-model assumption. RF/MLP/handcrafted-feature models are baselines. Frozen-head / checkpoint
runs are transfer ablations. A weak-instrument negative never licenses a kill conclusion.
"""

from .delta_generator import (
    SelfCensorParams,
    SelfCensorResult,
    apply_self_censor,
    expected_missing_rate,
    solve_beta0_for_rate,
)
from .xmodel import XModel, ConditionalGaussian
from .oracle import (
    missing_prob,
    llr_rows,
    population_missing_rate,
    solve_beta0_population,
    bayes_error_nsample,
    per_row_kl,
    OracleCell,
)
from .profiled_oracle import (
    profile_beta1,
    mar_params,
    bayes_error_vs_mar,
    compute_profiled_cell,
)
from .manifest import build_manifest, validate_manifest, write_manifest, REQUIRED_FIELDS

__all__ = [
    "SelfCensorParams",
    "SelfCensorResult",
    "apply_self_censor",
    "expected_missing_rate",
    "solve_beta0_for_rate",
    "XModel",
    "ConditionalGaussian",
    "missing_prob",
    "llr_rows",
    "population_missing_rate",
    "solve_beta0_population",
    "bayes_error_nsample",
    "per_row_kl",
    "OracleCell",
    "profile_beta1",
    "mar_params",
    "bayes_error_vs_mar",
    "compute_profiled_cell",
    "build_manifest",
    "validate_manifest",
    "write_manifest",
    "REQUIRED_FIELDS",
]
