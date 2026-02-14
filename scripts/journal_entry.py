#!/usr/bin/env python3
"""
Generate a journal entry from experiment artifacts.

Reads eval_report.json and calibrated.json (if present) and prints a
formatted experiment entry suitable for pasting into JOURNAL.md.

The eval_report.json structure (produced by lacuna.training.report):
    report["summary"]                  -> {accuracy, mcar_acc, mar_acc, mnar_acc, loss}
    report["confusion_matrix"]         -> {matrix: [[...]], labels, precision, recall, f1}
    report["confidence_analysis"]      -> {mean_confidence, mean_confidence_correct, ...}
    report["probability_distributions"]-> {mcar: {mean, std, ...}, mar: ..., mnar: ...}
    report["entropy"]                  -> {mcar: {mean, std}, mar: ..., correct: ..., incorrect: ...}
    report["calibration"]              -> {ece, bins: [...]}
    report["selective_accuracy"]       -> {thresholds: [...], threshold_for_90_accuracy, ...}
    report["n_samples"]                -> int

Usage:
    python scripts/journal_entry.py --report /path/to/eval_report.json
    python scripts/journal_entry.py --report /path/to/eval_report.json --calibration /path/to/calibrated.json
    python scripts/journal_entry.py --report /path/to/eval_report.json --name "Experiment 7"
    python scripts/journal_entry.py --run-dir /path/to/run/directory
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate journal entry from experiment artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Path to eval_report.json",
    )
    parser.add_argument(
        "--calibration", type=str, default=None,
        help="Path to calibrated.json (from calibrate.py)",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Run directory (auto-discovers report and calibration files)",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name/number",
    )
    parser.add_argument(
        "--hypothesis", type=str, default=None,
        help="Hypothesis text",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file (default: stdout)",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_confusion_matrix(cm_section: dict) -> str:
    """Format confusion matrix as markdown table.

    cm_section is report["confusion_matrix"] which has:
        matrix: [[row0], [row1], [row2]]
        labels: ["MCAR", "MAR", "MNAR"]
        precision: [p0, p1, p2]
        recall: [r0, r1, r2]
        f1: [f0, f1, f2]
    """
    # Support both formats: dict with "matrix" key, or raw list of lists
    if isinstance(cm_section, dict):
        cm = cm_section.get("matrix", [])
    else:
        cm = cm_section

    if not cm:
        return "*No confusion matrix data available.*"

    labels = ["MCAR", "MAR", "MNAR"]
    lines = []
    lines.append("|  | Pred MCAR | Pred MAR | Pred MNAR |")
    lines.append("|--|-----------|----------|-----------|")
    for i, label in enumerate(labels):
        if i < len(cm):
            row = cm[i]
            lines.append(f"| **True {label}** | {row[0]} | {row[1]} | {row[2]} |")
    return "\n".join(lines)


def format_per_class_metrics(report: dict) -> str:
    """Format per-class precision/recall/F1 table.

    Extracts from report["confusion_matrix"]["precision"/"recall"/"f1"]
    and report["confusion_matrix"]["matrix"] for support counts.
    """
    cm_section = report.get("confusion_matrix", {})

    if isinstance(cm_section, dict):
        precision = cm_section.get("precision", [])
        recall = cm_section.get("recall", [])
        f1 = cm_section.get("f1", [])
        matrix = cm_section.get("matrix", [])
    else:
        return "*No per-class metrics available.*"

    if not precision:
        return "*No per-class metrics available.*"

    labels = ["MCAR", "MAR", "MNAR"]
    lines = []
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|----|---------|")

    for i, label in enumerate(labels):
        if i < len(precision):
            p = precision[i] * 100
            r = recall[i] * 100 if i < len(recall) else 0
            f = f1[i] * 100 if i < len(f1) else 0
            support = sum(matrix[i]) if i < len(matrix) else 0
            lines.append(f"| {label} | {p:.1f}% | {r:.1f}% | {f:.1f}% | {support} |")

    return "\n".join(lines)


def format_probability_distributions(report: dict) -> str:
    """Format mean predicted probabilities by true class."""
    prob_dist = report.get("probability_distributions", {})
    if not prob_dist:
        return "*No probability distribution data available.*"

    lines = []
    lines.append("| True Class | P(MCAR) | P(MAR) | P(MNAR) |")
    lines.append("|------------|---------|--------|---------|")

    for class_name in ["MCAR", "MAR", "MNAR"]:
        key = class_name.lower()
        if key in prob_dist:
            mean_probs = prob_dist[key].get("mean", [0, 0, 0])
            lines.append(
                f"| {class_name} | {mean_probs[0]:.3f} | {mean_probs[1]:.3f} | {mean_probs[2]:.3f} |"
            )

    return "\n".join(lines)


def format_calibration(report: dict) -> str:
    """Format calibration metrics."""
    ece_data = report.get("calibration", {})
    if not ece_data:
        return "*No calibration data available.*"

    ece = ece_data.get("ece", 0)
    lines = [f"- **ECE:** {ece:.4f}"]

    # Confidence analysis
    conf = report.get("confidence_analysis", {})
    if conf:
        mean_correct = conf.get("mean_confidence_correct", None)
        mean_incorrect = conf.get("mean_confidence_incorrect", None)
        mean_overall = conf.get("mean_confidence", None)

        if mean_overall is not None:
            lines.append(f"- **Mean confidence:** {mean_overall:.3f}")
        if mean_correct is not None:
            lines.append(f"- **Mean confidence (correct):** {mean_correct:.3f}")
        if mean_incorrect is not None:
            lines.append(f"- **Mean confidence (incorrect):** {mean_incorrect:.3f}")

    return "\n".join(lines)


def format_selective_accuracy(report: dict) -> str:
    """Format selective accuracy table."""
    sel = report.get("selective_accuracy", {})
    if not sel:
        return "*No selective accuracy data available.*"

    thresholds = sel.get("thresholds", [])
    if not thresholds:
        return "*No selective accuracy data available.*"

    lines = []
    lines.append("| Threshold (τ) | Accuracy | Coverage |")
    lines.append("|---------------|----------|----------|")

    for entry in thresholds:
        t = entry.get("threshold", 0)
        acc = entry.get("accuracy", 0) * 100
        cov = entry.get("coverage", 0) * 100
        lines.append(f"| {t:.2f} | {acc:.1f}% | {cov:.1f}% |")

    # Add landmarks
    for key, label in [("threshold_for_90_accuracy", "90% accuracy"), ("threshold_for_95_accuracy", "95% accuracy")]:
        if key in sel and sel[key] is not None:
            lines.append(f"\n*{label} reached at τ = {sel[key]:.2f}*")

    return "\n".join(lines)


def generate_entry(report: dict, calibration: dict = None, name: str = None, hypothesis: str = None) -> str:
    """Generate a complete journal entry from eval_report.json data."""
    lines = []

    # Header
    exp_name = name or "Experiment N"
    lines.append(f"## {exp_name}")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")

    # Source info
    checkpoint = report.get("checkpoint")
    config = report.get("config")
    exp_dir = report.get("experiment_dir")
    if checkpoint:
        lines.append(f"**Checkpoint:** `{checkpoint}`")
    if config:
        lines.append(f"**Config:** `{config}`")
    if exp_dir:
        lines.append(f"**Run dir:** `{exp_dir}`")
    lines.append("")

    # Hypothesis
    if hypothesis:
        lines.append("### Hypothesis")
        lines.append(hypothesis)
        lines.append("")

    # Results summary
    lines.append("### Results")
    lines.append("")

    # Overall metrics — from report["summary"]
    summary = report.get("summary", {})
    overall_acc = summary.get("accuracy", 0) * 100
    n_samples = report.get("n_samples", 0)
    training_time = report.get("training_time_seconds")

    result_line = f"**Overall accuracy: {overall_acc:.1f}%** ({n_samples} samples)"
    if training_time:
        result_line += f" | Training: {training_time:.0f}s"
    lines.append(result_line)
    lines.append("")

    # Architecture info
    n_experts = report.get("n_experts")
    mnar_variants = report.get("mnar_variants")
    if n_experts:
        lines.append(f"**Architecture:** {n_experts} experts (mnar_variants={mnar_variants})")
        lines.append("")

    # Per-class metrics
    lines.append("**Per-class metrics:**")
    lines.append("")
    lines.append(format_per_class_metrics(report))
    lines.append("")

    # Confusion matrix
    if "confusion_matrix" in report:
        lines.append("**Confusion matrix** (rows = true, cols = predicted):")
        lines.append("")
        lines.append(format_confusion_matrix(report["confusion_matrix"]))
        lines.append("")

    # Probability distributions
    lines.append("**Mean predicted probabilities by true class:**")
    lines.append("")
    lines.append(format_probability_distributions(report))
    lines.append("")

    # Calibration
    lines.append("**Calibration:**")
    lines.append("")
    lines.append(format_calibration(report))
    lines.append("")

    # Selective accuracy
    lines.append("**Selective accuracy:**")
    lines.append("")
    lines.append(format_selective_accuracy(report))
    lines.append("")

    # Calibration results (from calibrate.py)
    if calibration:
        lines.append("### Temperature Scaling")
        lines.append("")
        opt_t = calibration.get("optimal_temperature", "N/A")
        lines.append(f"- **Optimal temperature:** {opt_t}")
        lines.append(f"- **NLL:** {calibration.get('nll_before', 'N/A')} → {calibration.get('nll_after', 'N/A')}")
        lines.append(f"- **ECE:** {calibration.get('ece_before', 'N/A')} → {calibration.get('ece_after', 'N/A')}")

        acc_before = calibration.get("accuracy_before")
        acc_after = calibration.get("accuracy_after")
        if acc_before is not None and acc_after is not None:
            lines.append(f"- **Accuracy:** {acc_before*100:.1f}% → {acc_after*100:.1f}%")
        lines.append("")

    # Entropy stats — stored under "entropy" key
    entropy = report.get("entropy", {})
    if entropy:
        lines.append("**Entropy:**")
        lines.append("")
        lines.append("| True Class | Mean Entropy | Std Entropy |")
        lines.append("|------------|-------------|-------------|")
        for class_name in ["MCAR", "MAR", "MNAR"]:
            key = class_name.lower()
            if key in entropy:
                mean_e = entropy[key].get("mean", 0)
                std_e = entropy[key].get("std", 0)
                lines.append(f"| {class_name} | {mean_e:.3f} | {std_e:.3f} |")
        if "correct" in entropy:
            lines.append(f"\n- Entropy (correct predictions): {entropy['correct'].get('mean', 0):.3f}")
        if "incorrect" in entropy:
            lines.append(f"- Entropy (incorrect predictions): {entropy['incorrect'].get('mean', 0):.3f}")
        lines.append("")

    # Interpretation placeholder
    lines.append("### Interpretation")
    lines.append("")
    lines.append("*TODO: Add interpretation.*")
    lines.append("")
    lines.append("### Next Decision")
    lines.append("")
    lines.append("*TODO: Add next decision.*")
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()

    # Auto-discover files from run directory
    report_path = args.report
    calibration_path = args.calibration

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if report_path is None:
            candidate = run_dir / "eval_report.json"
            if candidate.exists():
                report_path = str(candidate)
        if calibration_path is None:
            candidate = run_dir / "checkpoints" / "calibrated.json"
            if candidate.exists():
                calibration_path = str(candidate)

    if report_path is None:
        print("Error: --report or --run-dir required", file=sys.stderr)
        sys.exit(1)

    # Load data
    report = load_json(report_path)
    calibration = load_json(calibration_path) if calibration_path else None

    # Generate entry
    entry = generate_entry(
        report=report,
        calibration=calibration,
        name=args.name,
        hypothesis=args.hypothesis,
    )

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(entry)
        print(f"Journal entry written to {args.output}")
    else:
        print(entry)


if __name__ == "__main__":
    main()
