"""
lacuna.survey

P2 Lacuna-Survey δ-prior re-architecture — the survey-data and (later) δ-prior layer.

P2.1 (this phase) ships the DATA layer only: δ-parameterized own-value self-censoring on
REAL survey X with a known-δ answer sheet, matched missing rate, ordered δ-bins, and a
validate-before-trust generator manifest. No model, loss, or training here (later phases).
"""

from .answer_sheet import GENERATOR_FAMILY, SCHEMA_VERSION, AnswerSheet
from .delta_bins import NUM_BINS, assign_delta_bin, bin_edges
from .delta_generator import (
    SurveyCensorResult,
    check_realized_rate,
    generate_self_censor_example,
    rate_tolerance,
    select_target_predictor,
)

__all__ = [
    "AnswerSheet",
    "GENERATOR_FAMILY",
    "SCHEMA_VERSION",
    "NUM_BINS",
    "assign_delta_bin",
    "bin_edges",
    "SurveyCensorResult",
    "generate_self_censor_example",
    "select_target_predictor",
    "check_realized_rate",
    "rate_tolerance",
]
