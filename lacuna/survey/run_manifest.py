"""
lacuna.survey.run_manifest

Run-manifest builder/validator for a P2.2 δ-prior TRAINING run (PROPOSAL-P2 §9; audit §11).

ONE job: record — and fail loud to certify — the model/objective/calibration/eval/leakage
configuration of a δ-prior run, extending the P2.1 generator-config block (`survey.manifest`)
with the model side. Charter discipline: "No metric is interpreted unless the manifest
validates." For a `kind="main"` run the validator additionally enforces the from-scratch
guarantees (checkpoint_loaded=False, all_layers_trainable=True) and the blocking leakage gate
(leakage_pass=True). `timestamp`/`git_commit` are INJECTED (Rule 6).
"""

import json
from pathlib import Path
from typing import Dict, List

from .answer_sheet import ALLOWED_FAMILIES, GENERATOR_FAMILY, LOD_FAMILY, SCHEMA_VERSION
from .delta_bins import NUM_BINS, bin_edges
from .manifest import BETA0_SOLVER, DELTA_FORMULA, GENERATOR_PATH

_VALID_KINDS = ("main", "ablation", "smoke")
MODEL_PATH = "lacuna.survey.delta_head.DeltaPriorModel"
LOSS = "RPS"

# Per-idiom generator provenance (recorded in the manifest by family).
_FAMILY_PROVENANCE = {
    GENERATOR_FAMILY: {
        "generator_path": GENERATOR_PATH,
        "delta_formula": DELTA_FORMULA,
        "beta0_solver": BETA0_SOLVER,
    },
    LOD_FAMILY: {
        "generator_path": "lacuna.survey.lod_generator.generate_lod_example",
        "delta_formula": "P(missing_target)=sigmoid(beta0 + beta1*z_pred + delta*1[z_target>tau]); "
                         "delta=log-odds jump at threshold; MAR<=>delta==0",
        "beta0_solver": "bisection on monotone mean-sigmoid; population/sample target rate matched "
                        "to target_rate for ANY delta (only value-localization carries delta)",
    },
}

REQUIRED_FIELDS: List[str] = [
    "run_id",
    "git_commit",
    "timestamp",
    "kind",
    # generator/data block (mirrors survey.manifest)
    "generator_family",
    "generator_path",
    "delta_formula",
    "beta0_solver",
    "delta_bins",
    "delta_grid",
    "beta1_range",
    "target_rate",
    "seed",
    "answer_sheet_schema_version",
    # model block
    "model_path",
    "model_arch",
    "trainable_param_count",
    "checkpoint_loaded",
    "all_layers_trainable",
    "num_bins",
    # objective / calibration
    "loss",
    "temperature",
    # eval
    "split_scheme",
    "metrics",
    "calibration",
    # leakage gate
    "leakage",
    "leakage_pass",
    "wall_clock_seconds",
]


def build_manifest(
    *,
    run_id: str,
    git_commit: str,
    timestamp: str,
    kind: str,
    delta_grid: List[float],
    beta1_range: List[float],
    target_rate: float,
    seed: int,
    model_arch: Dict,
    trainable_param_count: int,
    checkpoint_loaded: bool,
    all_layers_trainable: bool,
    temperature: float,
    split_scheme: Dict,
    metrics: Dict,
    calibration: Dict,
    leakage: Dict,
    leakage_pass: bool,
    wall_clock_seconds: float,
    generator_family: str = GENERATOR_FAMILY,
    num_bins: int = NUM_BINS,
    model_path: str = MODEL_PATH,
    named_prior: Dict = None,
) -> Dict:
    """Assemble a δ-prior run manifest. Does not write; pair with validate + write.

    `model_path` lets a non-P2.2 model (e.g. the Level-1 φ-spine) record its own path. `named_prior`
    (Level-1, MASTER §7) records the explicit semi-synthetic training measure P_prior; validate it
    separately with `validate_named_prior`.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    if generator_family not in _FAMILY_PROVENANCE:
        raise ValueError(f"unknown generator_family {generator_family!r}; "
                         f"known: {sorted(_FAMILY_PROVENANCE)}")
    prov = _FAMILY_PROVENANCE[generator_family]
    return {
        "named_prior": named_prior,
        "run_id": run_id,
        "git_commit": git_commit,
        "timestamp": timestamp,
        "kind": kind,
        "generator_family": generator_family,
        "generator_path": prov["generator_path"],
        "delta_formula": prov["delta_formula"],
        "beta0_solver": prov["beta0_solver"],
        "delta_bins": bin_edges(),
        "delta_grid": list(delta_grid),
        "beta1_range": list(beta1_range),
        "target_rate": target_rate,
        "seed": seed,
        "answer_sheet_schema_version": SCHEMA_VERSION,
        "model_path": model_path,
        "model_arch": model_arch,
        "trainable_param_count": trainable_param_count,
        "checkpoint_loaded": checkpoint_loaded,
        "all_layers_trainable": all_layers_trainable,
        "num_bins": num_bins,
        "loss": LOSS,
        "temperature": temperature,
        "split_scheme": split_scheme,
        "metrics": metrics,
        "calibration": calibration,
        "leakage": leakage,
        "leakage_pass": leakage_pass,
        "wall_clock_seconds": wall_clock_seconds,
    }


def validate_manifest(manifest: Dict) -> None:
    """Raise loudly if the manifest cannot certify the intended δ-prior run."""
    missing = [k for k in REQUIRED_FIELDS if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing required field(s): {missing}")
    null_fields = [k for k in REQUIRED_FIELDS if manifest.get(k) is None]
    if null_fields:
        raise ValueError(f"manifest field(s) are null but required: {null_fields}")

    if manifest["kind"] not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {manifest['kind']!r}")
    if manifest["generator_family"] not in ALLOWED_FAMILIES:
        raise ValueError(f"generator_family must be in {sorted(ALLOWED_FAMILIES)}")
    if manifest["loss"] != LOSS:
        raise ValueError(f"P2.2 main loss must be {LOSS!r} (no 3-class CE / binary), got {manifest['loss']!r}")
    if manifest["answer_sheet_schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"answer_sheet_schema_version must be {SCHEMA_VERSION}")
    grid = manifest["delta_grid"]
    if not isinstance(grid, list) or len(grid) == 0 or not any(float(g) == 0.0 for g in grid):
        raise ValueError("delta_grid must be a non-empty list including 0.0 (MAR)")

    # from-scratch guarantees for a MAIN run (charter §4.9; audit §3)
    if manifest["kind"] == "main":
        if manifest["checkpoint_loaded"]:
            raise ValueError("kind='main' with checkpoint_loaded=True is forbidden (no transfer)")
        if manifest["all_layers_trainable"] is not True:
            raise ValueError("kind='main' must have all_layers_trainable=True (no frozen encoder)")
        if manifest["leakage_pass"] is not True:
            raise ValueError(
                "kind='main' with leakage_pass=False is invalid: a rate cue means the δ-prior "
                "may be reading δ off the missing rate (audit §10) — result not interpretable"
            )


_NAMED_PRIOR_KEYS = (
    "dataset_catalog", "contaminants_excluded", "idiom_vocabulary", "delta_grid",
    "delta_grid_weights", "prior_marginal", "rate_regime", "phi_config", "data_role",
)


def validate_named_prior(named_prior: Dict) -> None:
    """Fail loud unless the Level-1 named-prior block is complete and role-B (MASTER §7).

    The dataset catalog IS part of the scientific prior, so a Level-1 result is uninterpretable
    without an explicit, auditable P_prior. Supervised data must be role B (never natural missingness).
    """
    if not isinstance(named_prior, dict):
        raise ValueError("named_prior must be a dict for a Level-1 run")
    missing = [k for k in _NAMED_PRIOR_KEYS if k not in named_prior]
    if missing:
        raise ValueError(f"named_prior missing required key(s): {missing}")
    if named_prior["data_role"] != "B_complete_projection":
        raise ValueError(
            f"named_prior.data_role must be 'B_complete_projection' (never train on natural "
            f"missingness, MASTER §7), got {named_prior['data_role']!r}"
        )
    if not isinstance(named_prior["idiom_vocabulary"], list) or not named_prior["idiom_vocabulary"]:
        raise ValueError("named_prior.idiom_vocabulary must be a non-empty list of survey idioms")


def write_manifest(path: Path, manifest: Dict) -> Path:
    """Validate then write the manifest as pretty JSON. Returns the path."""
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path
