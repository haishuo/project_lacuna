"""
One-time migration scanner: discover existing run directories and
register them in the RunRegistry.

Single responsibility (Rule 3): scan filesystem artifacts and populate
the registry.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from lacuna.experiments.registry import RunRegistry, RunEntry


# Folder-name patterns that encode a timestamp.
_TS_PATTERNS = [
    # lacuna_semisyn_YYYYMMDD_HHMMSS  or  lacuna_YYYYMMDD_HHMMSS
    re.compile(r"(\d{8}_\d{6})$"),
]


def scan_run_directory(run_dir: Path) -> dict:
    """
    Extract metadata from whatever artifacts exist in *run_dir*.

    Returns a dict suitable for passing to ``RunRegistry.register()``.
    Raises ``ValueError`` if the directory is not a recognisable run.
    """
    if not run_dir.is_dir():
        raise ValueError(f"Not a directory: {run_dir}")

    info: Dict = {
        "folder_path": str(run_dir.resolve()),
        "config_path": "",
        "description": "",
        "metrics": {},
        "mnar_variants": [],
        "n_experts": 0,
        "tags": [],
    }

    # -- timestamp ----------------------------------------------------------
    info["timestamp"] = _extract_timestamp(run_dir)

    # -- config -------------------------------------------------------------
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        info["config_path"] = str(config_path)

    # -- experiment_meta.json -----------------------------------------------
    meta_path = run_dir / "experiment_meta.json"
    if meta_path.exists():
        meta = _load_json(meta_path)
        if meta is not None:
            info["mnar_variants"] = meta.get("mnar_variants", [])
            info["n_experts"] = 2 + len(info["mnar_variants"])
            if "config_path" in meta and meta["config_path"]:
                info["config_path"] = meta["config_path"]
            if "timestamp" in meta:
                info["timestamp"] = meta["timestamp"]

    # -- status (calibrated > evaluated > training) -------------------------
    info["status"] = _determine_status(run_dir)

    # -- metrics from eval_report.json --------------------------------------
    eval_path = run_dir / "eval_report.json"
    if eval_path.exists():
        report = _load_json(eval_path)
        if report is not None:
            info["metrics"] = _extract_metrics(report)

    # -- calibration metrics ------------------------------------------------
    cal_json = run_dir / "checkpoints" / "calibrated.json"
    if cal_json.exists():
        cal = _load_json(cal_json)
        if cal is not None:
            info["metrics"]["ece_calibrated"] = cal.get("ece_after", 0.0)
            info["metrics"]["ece_before_cal"] = cal.get("ece_before", 0.0)

    # -- description --------------------------------------------------------
    info["description"] = _build_description(run_dir, info)

    return info


def migrate_existing_runs(
    runs_dir: Path,
    registry: RunRegistry,
) -> List[RunEntry]:
    """
    Scan all subdirectories of *runs_dir*, sort chronologically, and register
    each one.  Already-registered directories are skipped (idempotent).

    Returns the list of newly created RunEntry objects.
    """
    if not runs_dir.is_dir():
        raise ValueError(f"Runs directory does not exist: {runs_dir}")

    subdirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=_sort_key_for_dir,
    )

    new_entries: List[RunEntry] = []
    for subdir in subdirs:
        # Skip if already registered.
        if registry.find_by_folder(subdir.name) is not None:
            continue

        try:
            info = scan_run_directory(subdir)
        except ValueError:
            continue

        entry = registry.register(
            folder_path=info["folder_path"],
            timestamp=info["timestamp"],
            config_path=info["config_path"],
            status=info["status"],
            description=info["description"],
            metrics=info["metrics"],
            mnar_variants=info["mnar_variants"],
            n_experts=info["n_experts"],
            tags=info["tags"],
        )
        new_entries.append(entry)

    return new_entries


# -- private helpers --------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    """Load a JSON file, returning None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _extract_timestamp(run_dir: Path) -> str:
    """
    Try to get an ISO-8601 timestamp from experiment_meta.json,
    then fall back to parsing the folder name.
    """
    meta_path = run_dir / "experiment_meta.json"
    if meta_path.exists():
        meta = _load_json(meta_path)
        if meta and "timestamp" in meta:
            return meta["timestamp"]

    # Parse from folder name.
    for pat in _TS_PATTERNS:
        m = pat.search(run_dir.name)
        if m:
            raw = m.group(1)  # e.g. "20260110_051028"
            try:
                dt = datetime.strptime(raw, "%Y%m%d_%H%M%S")
                return dt.isoformat()
            except ValueError:
                pass

    # NON-DETERMINISTIC: falls back to current time when no timestamp found.
    return datetime.now().isoformat()


def _determine_status(run_dir: Path) -> str:
    """Determine run status from the presence of artifacts."""
    cal_pt = run_dir / "checkpoints" / "calibrated.pt"
    cal_json = run_dir / "checkpoints" / "calibrated.json"
    eval_report = run_dir / "eval_report.json"

    if cal_pt.exists() or cal_json.exists():
        return "calibrated"
    if eval_report.exists():
        return "evaluated"
    return "training"


def _extract_metrics(report: dict) -> Dict[str, float]:
    """Pull key metrics from an eval_report.json dict."""
    metrics: Dict[str, float] = {}
    summary = report.get("summary", {})
    for key in ("accuracy", "mcar_acc", "mar_acc", "mnar_acc", "loss"):
        if key in summary:
            metrics[key] = summary[key]

    # ECE may be in confidence_analysis.
    conf = report.get("confidence_analysis", {})
    if "ece" in conf:
        metrics["ece"] = conf["ece"]

    return metrics


def _build_description(run_dir: Path, info: dict) -> str:
    """Build a short human-readable description."""
    parts = [run_dir.name]
    variants = info.get("mnar_variants", [])
    if variants and variants != ["self_censoring"]:
        parts.append(f"mnar={variants}")
    status = info.get("status", "training")
    if status != "training":
        parts.append(status)
    return " | ".join(parts)


def _sort_key_for_dir(d: Path):
    """Sort directories chronologically by embedded timestamp, else by name."""
    for pat in _TS_PATTERNS:
        m = pat.search(d.name)
        if m:
            return m.group(1)
    return d.name
