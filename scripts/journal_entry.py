#!/usr/bin/env python3
"""
Generate a journal entry from experiment artifacts.

Reads eval_report.json and calibrated.json (if present) and prints a
formatted experiment entry suitable for pasting into JOURNAL.md.

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


def format_confusion_matrix(cm: list) -> str:
    """Format confusion matrix as markdown table."""
    labels = ["MCAR", "MAR", "MNAR"]
    lines = []
    lines.append("|  | Pred MCAR | Pred MAR | Pred MNAR |")
    lines.append("|--|-----------|----------|-----------|")
    for i, label in enumerate(labels):
        row = cm[i]
        lines.append(f"| **True {label}** | {row[0]} | {row[1]} | {row[2]} |")
    return "\n".join(lines)


def format_per_class_metrics(report: dict) -> str:
    """Format per-class precision/recall/F1 table."""
    lines = []
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|----|---------|")

    for class_name in ["MCAR", "MAR", "MNAR"]:
        key = class_name.lower()
        if key in report.get("per_class", {}):
            cls = report["per_class"][key]
            prec = cls.get("precision", 0) * 100
            rec = cls.get("recall", 0) * 100
            f1 = cls.get("f1", 0) * 100
            support = cls.get("support", 0)
            lines.append(f"| {class_name} | {prec:.1f}% | {rec:.1f}% | {f1:.1f}% | {support} |")

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
        lines.append(f"- **Mean confidence (correct):** {conf.get('mean_confidence_correct', 0):.3f}")
        lines.append(f"- **Mean confidence (incorrect):** {conf.get('mean_confidence_incorrect', 0):.3f}")

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
    """Generate a complete journal entry."""
    lines = []

    # Header
    exp_name = name or "Experiment N"
    lines.append(f"## {exp_name}")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")

    if "checkpoint_path" in report:
        lines.append(f"**Checkpoint:** `{report['checkpoint_path']}`")
    if "config_path" in report:
        lines.append(f"**Config:** `{report['config_path']}`")
    lines.append("")

    # Hypothesis
    if hypothesis:
        lines.append(f"### Hypothesis")
        lines.append(hypothesis)
        lines.append("")

    # Results summary
    lines.append("### Results")
    lines.append("")

    # Overall metrics table
    overall_acc = report.get("overall_accuracy", 0) * 100
    n_samples = report.get("n_samples", 0)
    lines.append(f"**Overall accuracy: {overall_acc:.1f}%** ({n_samples} samples)")
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
        lines.append(f"- **Accuracy:** {calibration.get('accuracy_before', 0)*100:.1f}% → {calibration.get('accuracy_after', 0)*100:.1f}%")
        lines.append("")

    # Entropy stats
    entropy = report.get("entropy_stats", {})
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
