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
from .delta_head import (
    DeltaBinHead,
    DeltaPriorModel,
    assert_fresh_and_trainable,
    count_parameters,
    create_delta_prior_model,
    init_parameters_,
)
from .loss import fit_temperature, log_score, rps_loss, uniform_rps
from .example_source import (
    ExampleSource,
    StratifiedRealXSource,
    SurveyExampleSource,
    SyntheticTwoColSource,
)
from .proxy_score import column_r2, proxy_score, r2_distribution, target_r2_table
from .column_stats import cardinality_distribution, target_cardinality_table
from .conditioned_head import (
    CONDITIONING_METHOD,
    TargetConditionedDeltaModel,
    create_target_conditioned_model,
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
    "DeltaBinHead",
    "DeltaPriorModel",
    "create_delta_prior_model",
    "init_parameters_",
    "assert_fresh_and_trainable",
    "count_parameters",
    "rps_loss",
    "log_score",
    "uniform_rps",
    "fit_temperature",
    "ExampleSource",
    "SurveyExampleSource",
    "SyntheticTwoColSource",
    "StratifiedRealXSource",
    "column_r2",
    "proxy_score",
    "target_r2_table",
    "r2_distribution",
    "target_cardinality_table",
    "cardinality_distribution",
    "TargetConditionedDeltaModel",
    "create_target_conditioned_model",
    "CONDITIONING_METHOD",
]
