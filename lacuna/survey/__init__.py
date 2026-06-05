"""
lacuna.survey

P2 Lacuna-Survey δ-prior re-architecture — the survey-data and (later) δ-prior layer.

P2.1 (this phase) ships the DATA layer only: δ-parameterized own-value self-censoring on
REAL survey X with a known-δ answer sheet, matched missing rate, ordered δ-bins, and a
validate-before-trust generator manifest. No model, loss, or training here (later phases).
"""

from .answer_sheet import (
    ALLOWED_FAMILIES,
    GENERATOR_FAMILY,
    LOD_FAMILY,
    SCHEMA_VERSION,
    AnswerSheet,
)
from .lod_generator import LODParams, apply_lod_censor, generate_lod_example
from .lod_oracle import lod_oracle_cell
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
    LODSurveyExampleSource,
    StratifiedRealXSource,
    SurveyExampleSource,
    SyntheticTwoColSource,
)
from .proxy_score import column_r2, proxy_score, r2_distribution, target_r2_table
from .column_stats import cardinality_distribution, target_cardinality_table
from .consequence_features import (
    FEATURE_NAMES,
    N_FEATURES,
    compute_consequence_features,
)
from .transfer_features import (
    N_TRANSFER_FEATURES,
    compute_transfer_features,
)
from .coarse_bins import assign_bins, scheme_num_bins, scheme_spec
from .feature_only_head import FeatureOnlyDeltaModel, create_feature_only_model
from .conditioned_head import (
    CONDITIONING_METHOD,
    TargetConditionedDeltaModel,
    create_target_conditioned_model,
)

__all__ = [
    "AnswerSheet",
    "GENERATOR_FAMILY",
    "LOD_FAMILY",
    "ALLOWED_FAMILIES",
    "SCHEMA_VERSION",
    "LODParams",
    "apply_lod_censor",
    "generate_lod_example",
    "lod_oracle_cell",
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
    "LODSurveyExampleSource",
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
