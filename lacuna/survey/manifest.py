"""
lacuna.survey.manifest

Run-manifest builder and validator for the P2 δ-generator DATA layer (PROPOSAL-P2 §9).

ONE job: record — and fail loud to certify — the generator configuration behind a batch
of semi-synthetic answer-sheeted examples, following the validate-before-trust pattern of
`lacuna.feasibility.manifest`. P2.1 produces NO model and NO metric, so this manifest
covers the generator/data contract only (family, δ-grid + bins, matched rate, β₁ range,
dataset pool, answer-sheet schema). The model/loss/calibration manifest is a later phase.

Charter discipline: "No metric is interpreted unless the manifest validates." Here there
is no metric yet, but the same gate protects the DATA: a batch of examples is not trusted
unless this manifest certifies how it was generated. `timestamp` and `git_commit` are
INJECTED by the caller (Rule 6) — never read from a wall clock inside library code.
"""

import json
from pathlib import Path
from typing import Dict, List

from .answer_sheet import GENERATOR_FAMILY, SCHEMA_VERSION
from .delta_bins import bin_edges

REQUIRED_FIELDS: List[str] = [
    "run_id",
    "git_commit",
    "timestamp",
    "kind",  # "main" | "ablation" | "smoke"
    "generator_family",  # must be GENERATOR_FAMILY
    "generator_path",  # import path of the generator entry point
    "delta_formula",  # human-readable formula string
    "beta0_solver",  # description of the matched-rate solver
    "delta_bins",  # bin-edge metadata (from delta_bins.bin_edges())
    "delta_grid",  # list of δ values sampled (must include 0.0 for MAR)
    "beta1_range",  # [min, max] swept nuisance range
    "target_rate",  # matched expected marginal missing rate
    "dataset_pool",  # list of source survey dataset names
    "seed",  # base RNG seed
    "answer_sheet_schema_version",  # must equal answer_sheet.SCHEMA_VERSION
    "answer_sheets_saved",  # bool — answer sheets persisted for this batch
]

_VALID_KINDS = ("main", "ablation", "smoke")

GENERATOR_PATH = "lacuna.survey.delta_generator.generate_self_censor_example"
DELTA_FORMULA = (
    "P(missing_target)=sigmoid(beta0 + beta1*z_pred + delta*z_target); "
    "delta==beta2; MAR<=>delta==0"
)
BETA0_SOLVER = (
    "bisection on monotone mean-sigmoid; expected/population target-column rate "
    "matched to target_rate for ANY delta (no rate cue)"
)


def build_manifest(
    *,
    run_id: str,
    git_commit: str,
    timestamp: str,
    kind: str,
    delta_grid: List[float],
    beta1_range: List[float],
    target_rate: float,
    dataset_pool: List[str],
    seed: int,
    answer_sheets_saved: bool,
) -> Dict:
    """Assemble a generator-data manifest dict. Does not write; pair with validate + write."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "timestamp": timestamp,
        "kind": kind,
        "generator_family": GENERATOR_FAMILY,
        "generator_path": GENERATOR_PATH,
        "delta_formula": DELTA_FORMULA,
        "beta0_solver": BETA0_SOLVER,
        "delta_bins": bin_edges(),
        "delta_grid": list(delta_grid),
        "beta1_range": list(beta1_range),
        "target_rate": target_rate,
        "dataset_pool": list(dataset_pool),
        "seed": seed,
        "answer_sheet_schema_version": SCHEMA_VERSION,
        "answer_sheets_saved": answer_sheets_saved,
    }


def validate_manifest(manifest: Dict) -> None:
    """Raise loudly if the manifest cannot certify how the data batch was generated."""
    missing = [k for k in REQUIRED_FIELDS if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing required field(s): {missing}")
    null_fields = [k for k in REQUIRED_FIELDS if manifest.get(k) is None]
    if null_fields:
        raise ValueError(f"manifest field(s) are null but required: {null_fields}")

    if manifest["kind"] not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {manifest['kind']!r}")
    if manifest["generator_family"] != GENERATOR_FAMILY:
        raise ValueError(
            f"generator_family must be {GENERATOR_FAMILY!r}, got {manifest['generator_family']!r}"
        )
    if manifest["answer_sheet_schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"answer_sheet_schema_version must be {SCHEMA_VERSION}, "
            f"got {manifest['answer_sheet_schema_version']!r}"
        )
    grid = manifest["delta_grid"]
    if not isinstance(grid, list) or len(grid) == 0:
        raise ValueError("delta_grid must be a non-empty list")
    if not any(float(g) == 0.0 for g in grid):
        raise ValueError("delta_grid must include 0.0 (the MAR / δ=0 case)")
    if any(float(g) < 0.0 for g in grid):
        raise ValueError(f"delta_grid must be non-negative, got {grid}")
    rng_lo_hi = manifest["beta1_range"]
    if not (isinstance(rng_lo_hi, list) and len(rng_lo_hi) == 2):
        raise ValueError(f"beta1_range must be [min, max], got {rng_lo_hi!r}")
    if rng_lo_hi[0] < 0.0 or rng_lo_hi[1] < rng_lo_hi[0]:
        raise ValueError(f"beta1_range must satisfy 0 <= min <= max, got {rng_lo_hi}")
    if not (0.0 < float(manifest["target_rate"]) < 1.0):
        raise ValueError(f"target_rate must be in (0, 1), got {manifest['target_rate']}")
    if not isinstance(manifest["dataset_pool"], list) or len(manifest["dataset_pool"]) == 0:
        raise ValueError("dataset_pool must be a non-empty list of dataset names")


def write_manifest(path: Path, manifest: Dict) -> Path:
    """Validate then write the manifest as pretty JSON. Returns the path."""
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path
