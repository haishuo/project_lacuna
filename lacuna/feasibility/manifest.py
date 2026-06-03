"""
lacuna.feasibility.manifest

Run-manifest builder and validator for the feasibility probe.

Charter discipline (user ruling #5): "No metric is interpretable until the manifest
confirms the intended experiment was actually run." Therefore `validate_manifest`
FAILS LOUD (Rule 1) if any required field is missing or None, and the experiment
runners must validate before any result is trusted.

Determinism (Rule 6): `timestamp` and `git_commit` are INJECTED by the caller (the
script), never read from a wall clock inside library code — so manifests are
reproducible and testable.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# Every run (oracle or model) must record exactly these. Model-arm-only fields
# (trainable_param_count, all_layers_trainable, wall_clock_seconds, metrics, calibration)
# are still present for oracle runs but may be explicitly null where N/A — recorded,
# never silently omitted.
REQUIRED_FIELDS: List[str] = [
    "run_id",
    "git_commit",
    "timestamp",
    "arm",  # "oracle" | "model"
    "kind",  # "main" | "baseline" | "ablation"
    "generator_path",  # import path of the δ-generator used
    "delta_formula",  # human-readable formula string
    "beta0_solver",  # description of the rate-matching solver
    "grid",  # dict with delta/beta1/target_rate/rho/n value lists
    "xmodel",  # X-model descriptor (type + assumption)
    "checkpoint_loaded",  # bool — MUST be False for a main feasibility result
    "all_layers_trainable",  # bool | None (None for oracle arm)
    "trainable_param_count",  # int | None (None for oracle arm)
    "wall_clock_seconds",  # float
    "split_scheme",  # description of train/val / held-out-family split (or "n/a" for oracle)
    "metrics",  # dict. Oracle: the bayes_error is a Monte-Carlo ESTIMATE of the theoretical
    # Bayes error of the optimal LLR test — report it WITH bayes_error_se / 95% CI / n_mc, never
    # as an exact value. Model arm: auc / logloss / gap-to-ceiling.
    "calibration",  # dict | None (model arm; None for oracle)
]

# Values that are legitimately allowed to be null, by arm.
_NULLABLE_FOR_ORACLE = {"all_layers_trainable", "trainable_param_count", "calibration"}


def build_manifest(
    *,
    run_id: str,
    git_commit: str,
    timestamp: str,
    arm: str,
    kind: str,
    grid: Dict,
    xmodel: Dict,
    wall_clock_seconds: float,
    metrics: Dict,
    checkpoint_loaded: bool = False,
    all_layers_trainable: Optional[bool] = None,
    trainable_param_count: Optional[int] = None,
    split_scheme: str = "n/a",
    calibration: Optional[Dict] = None,
    generator_path: str = "lacuna.feasibility.delta_generator.apply_self_censor",
    delta_formula: str = "P(missing)=sigmoid(beta0 + beta1*z_pred + beta2*z_target); delta==beta2; MAR<=>beta2==0",
    beta0_solver: str = "bisection on monotone mean-sigmoid; expected/population rate matched to target_rate",
) -> Dict:
    """Assemble a manifest dict. Does not write; pair with validate + write."""
    if arm not in ("oracle", "model"):
        raise ValueError(f"arm must be 'oracle' or 'model', got {arm!r}")
    if kind not in ("main", "baseline", "ablation"):
        raise ValueError(f"kind must be 'main'/'baseline'/'ablation', got {kind!r}")
    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "timestamp": timestamp,
        "arm": arm,
        "kind": kind,
        "generator_path": generator_path,
        "delta_formula": delta_formula,
        "beta0_solver": beta0_solver,
        "grid": grid,
        "xmodel": xmodel,
        "checkpoint_loaded": checkpoint_loaded,
        "all_layers_trainable": all_layers_trainable,
        "trainable_param_count": trainable_param_count,
        "wall_clock_seconds": wall_clock_seconds,
        "split_scheme": split_scheme,
        "metrics": metrics,
        "calibration": calibration,
    }


def validate_manifest(manifest: Dict) -> None:
    """Raise loudly if the manifest cannot certify the intended experiment was run."""
    missing = [k for k in REQUIRED_FIELDS if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing required field(s): {missing}")

    arm = manifest["arm"]
    nullable = _NULLABLE_FOR_ORACLE if arm == "oracle" else set()
    null_fields = [
        k
        for k in REQUIRED_FIELDS
        if manifest.get(k) is None and k not in nullable
    ]
    if null_fields:
        raise ValueError(
            f"manifest field(s) are null but required for arm={arm!r}: {null_fields}"
        )

    # A 'main' feasibility result may never have loaded a checkpoint (charter §4.9).
    if manifest["kind"] == "main" and manifest["checkpoint_loaded"]:
        raise ValueError(
            "kind='main' with checkpoint_loaded=True is forbidden: a checkpoint-loaded "
            "run is a transfer ablation, not a main feasibility result (charter §4.9)"
        )
    # A main MODEL run must train all layers.
    if manifest["kind"] == "main" and arm == "model" and manifest["all_layers_trainable"] is not True:
        raise ValueError(
            "kind='main' model run must have all_layers_trainable=True "
            "(no frozen encoder / stapled head — charter §4.9, §5.2)"
        )


def write_manifest(path: Path, manifest: Dict) -> Path:
    """Validate then write the manifest as pretty JSON. Returns the path."""
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path
