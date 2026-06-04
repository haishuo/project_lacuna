"""
lacuna.survey.answer_sheet

The per-example "answer sheet" — the supervised label + audit record for one
semi-synthetic δ-parameterized self-censoring example (PROPOSAL-P2 §3).

ONE job: hold the frozen ground-truth record of a generated example and
serialize/deserialize it losslessly (JSON-friendly dict round-trip).

This is the ONLY ground truth Lacuna ever gets for δ (charter §4.4): the generator
KNOWS the true δ because it imposed it, so it is recorded here alongside everything
needed to reproduce the example and audit for leakage (matched rate, the censored
target, the MAR-nuisance predictor, the seed).

Contract (Coding Bible Rules 1, 2): `from_dict` fails loud on a missing field, a
schema-version mismatch, or an unknown generator family — a malformed answer sheet
is never silently completed with defaults.
"""

from dataclasses import dataclass, asdict

# Bump when the field set changes; from_dict refuses mismatched payloads.
SCHEMA_VERSION = 1

# The one mechanism family P2.1 exercises (PROPOSAL §3, §13).
GENERATOR_FAMILY = "own_value_self_censoring"


@dataclass(frozen=True)
class AnswerSheet:
    """Frozen ground-truth record for one generated self-censoring example.

    Fields (all required — PROPOSAL §3):
        source_name: real survey dataset the X came from.
        n, d: shape of the (subsampled) X the mechanism was applied to.
        target_col_idx / target_col_name: the censored column.
        predictor_col_idx / predictor_col_name: the observed MAR-nuisance column.
        beta0: solved intercept (matched-rate solve).
        beta1: observed-coupling nuisance coefficient (>= 0).
        delta: own-value dependence (== β₂); the supervised label. MAR ⇔ delta=0.
        delta_bin: ordered bin index for `delta` (see delta_bins).
        generator_family: mechanism family id (must be GENERATOR_FAMILY).
        target_rate: requested expected marginal missing rate of the target column.
        realized_rate: actually-sampled fraction missing in the target column.
        corr_target_predictor: measured Pearson corr(target, predictor) on observed X.
        seed: RNGState seed that produced this example (determinism anchor).
    """

    source_name: str
    n: int
    d: int
    target_col_idx: int
    target_col_name: str
    predictor_col_idx: int
    predictor_col_name: str
    beta0: float
    beta1: float
    delta: float
    delta_bin: int
    generator_family: str
    target_rate: float
    realized_rate: float
    corr_target_predictor: float
    seed: int

    def to_dict(self) -> dict:
        """JSON-friendly dict, tagged with the schema version."""
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    @staticmethod
    def from_dict(payload: dict) -> "AnswerSheet":
        """Reconstruct from a `to_dict` payload. Fails loud on any defect."""
        if not isinstance(payload, dict):
            raise ValueError(f"answer-sheet payload must be a dict, got {type(payload)}")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"answer-sheet schema_version mismatch: expected {SCHEMA_VERSION}, "
                f"got {version!r}"
            )
        field_names = [f for f in AnswerSheet.__dataclass_fields__]
        missing = [f for f in field_names if f not in payload]
        if missing:
            raise ValueError(f"answer-sheet payload missing field(s): {missing}")
        family = payload["generator_family"]
        if family != GENERATOR_FAMILY:
            raise ValueError(
                f"unknown generator_family {family!r}; "
                f"P2.1 only produces {GENERATOR_FAMILY!r}"
            )
        return AnswerSheet(**{f: payload[f] for f in field_names})
