#!/usr/bin/env python3
"""
Consolidate all experiment results into a single portable JSON file.

Run this on Forge where the experiment artifacts live.

Usage:
    python scripts/consolidate_results.py --runs-dir /mnt/artifacts/project_lacuna/runs
    python scripts/consolidate_results.py --runs-dir /mnt/artifacts/project_lacuna/runs --output all_experiments.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_experiment_dirs(runs_dir: Path) -> list:
    """Find all experiment directories with eval reports."""
    experiments = []
    if not runs_dir.exists():
        print(f"Warning: {runs_dir} does not exist")
        return experiments

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        # Look for eval_report.json in the run dir or checkpoints subdir
        candidates = [
            d / "eval_report.json",
            d / "checkpoints" / "eval_report.json",
        ]
        for c in candidates:
            if c.exists():
                experiments.append({
                    "dir": str(d),
                    "name": d.name,
                    "report_path": str(c),
                })
                break

    return experiments


def load_calibration_info(exp_dir: Path) -> dict:
    """Load calibration JSON if it exists."""
    candidates = [
        exp_dir / "calibrated.json",
        exp_dir / "checkpoints" / "calibrated.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    return None


def load_experiment_meta(exp_dir: Path) -> dict:
    """Load experiment_meta.json if it exists."""
    candidates = [
        exp_dir / "experiment_meta.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    return None


def consolidate(runs_dir: Path) -> dict:
    """Consolidate all experiment results into a single dict."""
    experiments = find_experiment_dirs(runs_dir)

    consolidated = {
        "generated_at": datetime.now().isoformat(),
        "source_dir": str(runs_dir),
        "n_experiments": len(experiments),
        "experiments": [],
    }

    for exp in experiments:
        exp_dir = Path(exp["dir"])
        print(f"  Loading: {exp['name']}")

        # Load eval report
        with open(exp["report_path"]) as f:
            report = json.load(f)

        # Load optional metadata
        calibration = load_calibration_info(exp_dir)
        meta = load_experiment_meta(exp_dir)

        entry = {
            "name": exp["name"],
            "dir": exp["dir"],
            "report": report,
        }
        if calibration:
            entry["calibration"] = calibration
        if meta:
            entry["meta"] = meta

        consolidated["experiments"].append(entry)

    return consolidated


# Hard-coded experiment data for experiments that don't have eval_report.json
# on disk (reconstructed from JOURNAL.md and training output).
HISTORICAL_EXPERIMENTS = [
    {
        "id": 5,
        "name": "Experiment 5 — Semi-Synthetic Baseline (110 Generators)",
        "date": "2026-02-13",
        "architecture": "1/1/3 (5 experts)",
        "prior": "uniform",
        "loss": "cross_entropy",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "mnar_variants": ["self_censoring", "threshold", "latent"],
        "n_samples": 1280,
        "summary": {
            "accuracy": 0.770,
            "mcar_acc": 0.934,
            "mar_acc": 0.526,
            "mnar_acc": 0.853,
        },
        "confusion_matrix": {
            "matrix": [[255, 2, 16], [3, 132, 116], [105, 6, 645]],
            "labels": ["MCAR", "MAR", "MNAR"],
            "precision": [0.703, 0.943, 0.830],
            "recall": [0.934, 0.526, 0.853],
            "f1": [0.802, 0.675, 0.841],
        },
        "calibration": {"ece": 0.1338},
        "confidence": {
            "mean_correct": 0.865,
            "mean_incorrect": 0.818,
        },
        "probability_distributions": {
            "true_mcar": {"mean_p": [0.775, 0.028, 0.197], "count": 273},
            "true_mar": {"mean_p": [0.029, 0.522, 0.449], "count": 251},
            "true_mnar": {"mean_p": [0.173, 0.017, 0.810], "count": 756},
        },
    },
    {
        "id": 7,
        "name": "Experiment 7 — Calibration & Rebalancing (Brier + Balanced Prior)",
        "date": "2026-02-13",
        "architecture": "1/1/3 (5 experts)",
        "prior": "class_balanced",
        "loss": "brier",
        "label_smoothing": 0.1,
        "temperature": 1.0,
        "mnar_variants": ["self_censoring", "threshold", "latent"],
        "n_samples": 800,
        "training_time_seconds": 730.9,
        "summary": {
            "accuracy": 0.548,
            "mcar_acc": 0.651,
            "mar_acc": 0.307,
            "mnar_acc": 0.831,
        },
        "confusion_matrix": {
            "matrix": [[177, 0, 95], [0, 104, 235], [29, 3, 157]],
            "labels": ["MCAR", "MAR", "MNAR"],
            "precision": [0.859, 0.972, 0.322],
            "recall": [0.651, 0.307, 0.831],
            "f1": [0.741, 0.466, 0.465],
        },
        "calibration": {"ece": 0.2867},
        "confidence": {
            "mean_incorrect": 0.789,
        },
        "probability_distributions": {
            "true_mcar": {"mean_p": [0.643, 0.012, 0.345], "count": 272},
            "true_mar": {"mean_p": [0.019, 0.406, 0.575], "count": 339},
            "true_mnar": {"mean_p": [0.158, 0.107, 0.735], "count": 189},
        },
        "status": "FAILED",
        "failure_reason": "Changed 3 variables at once (Brier, label smoothing, balanced prior)",
    },
    {
        "id": 8,
        "name": "Experiment 8 — Class-Balanced Prior Only",
        "date": "2026-02-14",
        "architecture": "1/1/3 (5 experts)",
        "prior": "class_balanced",
        "loss": "cross_entropy",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "mnar_variants": ["self_censoring", "threshold", "latent"],
        "n_samples": 1171,
        "training_time_seconds": 729.2,
        "summary": {
            "accuracy": 0.626,
            "mcar_acc": 0.890,
            "mar_acc": 0.348,
            "mnar_acc": 0.746,
        },
        "confusion_matrix": {
            "matrix": [[258, 2, 30], [119, 98, 64], [148, 14, 438]],
            "labels": ["MCAR", "MAR", "MNAR"],
            "precision": [0.491, 0.860, 0.823],
            "recall": [0.890, 0.349, 0.730],
            "f1": [0.633, 0.496, 0.774],
        },
        "calibration": {"ece": 0.2329},
        "confidence": {
            "mean_correct": 0.782,
            "mean_incorrect": 0.720,
        },
        "probability_distributions": {
            "true_mcar": {"mean_p": [0.705, 0.045, 0.250], "count": 290},
            "true_mar": {"mean_p": [0.295, 0.361, 0.344], "count": 281},
            "true_mnar": {"mean_p": [0.247, 0.060, 0.692], "count": 600},
        },
        "status": "FAILED",
        "failure_reason": "Class-balanced prior alone regresses all metrics",
    },
    {
        "id": 9,
        "name": "Experiment 9 — 1/1/1 Expert Ablation (Uniform Prior)",
        "date": "2026-02-14",
        "architecture": "1/1/1 (3 experts)",
        "prior": "uniform",
        "loss": "cross_entropy",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "mnar_variants": ["self_censoring"],
        "n_samples": 800,
        "training_time_seconds": 1679,
        "summary": {
            "accuracy": 0.784,
            "mcar_acc": 0.895,
            "mar_acc": 0.693,
            "mnar_acc": 0.794,
        },
        "confusion_matrix": {
            "matrix": [[195, 0, 23], [1, 205, 90], [33, 26, 227]],
            "labels": ["MCAR", "MAR", "MNAR"],
            "precision": [0.852, 0.887, 0.668],
            "recall": [0.895, 0.693, 0.794],
            "f1": [0.872, 0.778, 0.725],
        },
        "calibration": {"ece": 0.1157},
        "confidence": {
            "mean_correct": 0.934,
            "mean_incorrect": 0.772,
        },
        "selective_accuracy": {
            "thresholds": [
                {"threshold": 0.50, "accuracy": 0.793, "coverage": 0.979},
                {"threshold": 0.60, "accuracy": 0.811, "coverage": 0.928},
                {"threshold": 0.70, "accuracy": 0.845, "coverage": 0.869},
                {"threshold": 0.80, "accuracy": 0.861, "coverage": 0.819},
                {"threshold": 0.90, "accuracy": 0.909, "coverage": 0.702},
                {"threshold": 0.95, "accuracy": 0.928, "coverage": 0.591},
            ],
        },
    },
    {
        "id": 10,
        "name": "Experiment 10 — Post-Hoc Temperature Scaling",
        "date": "2026-02-14",
        "architecture": "1/1/1 (3 experts)",
        "prior": "uniform",
        "loss": "cross_entropy",
        "label_smoothing": 0.0,
        "temperature": 1.963,
        "mnar_variants": ["self_censoring"],
        "n_samples": 800,
        "summary": {
            "accuracy": 0.826,
            "mcar_acc": 0.945,
            "mar_acc": 0.736,
            "mnar_acc": 0.845,
        },
        "confusion_matrix": {
            "matrix": [[154, 0, 9], [5, 212, 71], [35, 19, 295]],
            "labels": ["MCAR", "MAR", "MNAR"],
            "precision": [0.794, 0.918, 0.787],
            "recall": [0.945, 0.736, 0.845],
            "f1": [0.863, 0.817, 0.815],
        },
        "calibration": {"ece": 0.0383},
        "confidence": {
            "mean_correct": 0.818,
            "mean_incorrect": 0.733,
        },
        "probability_distributions": {
            "true_mcar": {"mean_p": [0.821, 0.032, 0.148], "count": 163},
            "true_mar": {"mean_p": [0.033, 0.667, 0.300], "count": 288},
            "true_mnar": {"mean_p": [0.152, 0.138, 0.709], "count": 349},
        },
        "selective_accuracy": {
            "thresholds": [
                {"threshold": 0.50, "accuracy": 0.832, "coverage": 0.970},
                {"threshold": 0.60, "accuracy": 0.843, "coverage": 0.899},
                {"threshold": 0.70, "accuracy": 0.877, "coverage": 0.774},
                {"threshold": 0.80, "accuracy": 0.903, "coverage": 0.618},
                {"threshold": 0.90, "accuracy": 0.915, "coverage": 0.308},
                {"threshold": 0.95, "accuracy": 0.923, "coverage": 0.049},
            ],
        },
    },
]


def add_historical_data(consolidated: dict):
    """Add hard-coded historical experiment data for experiments without JSON on disk."""
    for hist in HISTORICAL_EXPERIMENTS:
        # Check if this experiment is already in the consolidated data
        names_present = [e["name"] for e in consolidated["experiments"]]
        if any(hist["name"] in n for n in names_present):
            continue

        consolidated["experiments"].append({
            "name": hist["name"],
            "dir": None,
            "historical": True,
            "report": hist,
        })

    # Sort by experiment number (extract from name)
    def sort_key(e):
        name = e["name"]
        for i in range(20, 0, -1):
            if f"Experiment {i}" in name or f"experiment {i}" in name.lower():
                return i
        return 99
    consolidated["experiments"].sort(key=sort_key)
    consolidated["n_experiments"] = len(consolidated["experiments"])


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate experiment results for dissertation",
    )
    parser.add_argument(
        "--runs-dir", type=str,
        default="/mnt/artifacts/project_lacuna/runs",
        help="Directory containing experiment run folders",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "docs" / "data" / "all_experiments.json"),
        help="Output path for consolidated JSON",
    )
    parser.add_argument(
        "--include-historical", action="store_true", default=True,
        help="Include hard-coded data for experiments without JSON on disk",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output)

    print(f"Consolidating experiment results from: {runs_dir}")
    consolidated = consolidate(runs_dir)

    if args.include_historical:
        print(f"Adding historical experiment data...")
        add_historical_data(consolidated)

    print(f"\nTotal experiments: {consolidated['n_experiments']}")
    for exp in consolidated["experiments"]:
        status = ""
        if exp.get("historical"):
            status = " [historical]"
        report = exp.get("report", {})
        summary = report.get("summary", {})
        acc = summary.get("accuracy", "?")
        if isinstance(acc, float):
            acc = f"{acc*100:.1f}%"
        print(f"  {exp['name']}: accuracy={acc}{status}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
