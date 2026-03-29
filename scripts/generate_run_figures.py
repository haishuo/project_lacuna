#!/usr/bin/env python3
"""
Generate per-run dissertation figures from a single experiment run directory.

Reads eval_report.json (preferring the calibrated model's report in
checkpoints/ when it exists) and produces the full figure suite:

  confusion_matrix.{pdf,png}        — row-normalised heatmap with raw counts
  per_class_metrics.{pdf,png}       — grouped bar: precision / recall / F1
  reliability_diagram.{pdf,png}     — calibration bins (confidence vs accuracy)
  selective_accuracy.{pdf,png}      — accuracy vs coverage curve
  probability_distributions.{pdf,png} — mean predicted probs per true class

All figures are written to <run_dir>/figures/ (created if absent).

Usage:
    python scripts/generate_run_figures.py --run-dir /mnt/artifacts/project_lacuna/runs/my_run
    python scripts/generate_run_figures.py --run-dir /mnt/artifacts/project_lacuna/runs/my_run --output-dir /tmp/figs
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# ---------------------------------------------------------------------------
# Colour palette — shared with generate_roc_curves.py
# ---------------------------------------------------------------------------

CLASS_COLORS = {"MCAR": "#2196F3", "MAR": "#F44336", "MNAR": "#4CAF50"}
CLASS_NAMES = ["MCAR", "MAR", "MNAR"]
_METRIC_COLORS = {"Precision": "#2196F3", "Recall": "#F44336", "F1": "#4CAF50"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_report(run_dir: Path, report_path: Path | None = None) -> dict:
    """
    Return the best available eval report for this run.

    If report_path is given explicitly, load that file directly.

    Otherwise, search in priority order:
      1. checkpoints/eval_report.json  (calibrated model — preferred)
      2. eval_report.json              (pre-calibration)

    Raises FileNotFoundError if no report is found.
    """
    if report_path is not None:
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        with open(report_path) as fh:
            return json.load(fh)

    candidates = [
        run_dir / "checkpoints" / "eval_report.json",
        run_dir / "eval_report.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
    raise FileNotFoundError(
        f"No eval_report.json found in {run_dir} or {run_dir}/checkpoints/"
    )


# ---------------------------------------------------------------------------
# Figure 1 — Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(report: dict, output_dir: Path) -> None:
    """Row-normalised confusion matrix heatmap with absolute counts annotated."""
    cm_data = report.get("confusion_matrix", {})
    matrix = cm_data.get("matrix")
    if not matrix:
        print("  Skipping confusion matrix: no data in report")
        return

    matrix = np.array(matrix, dtype=float)
    labels = cm_data.get("labels", CLASS_NAMES)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(matrix[i, j])
            pct = normalised[i, j] * 100
            color = "white" if normalised[i, j] > 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=11, fontweight=weight, color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([f"Pred {l}" for l in labels], fontsize=11)
    ax.set_yticklabels([f"True {l}" for l in labels], fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)

    plt.tight_layout()
    _save(fig, output_dir / "confusion_matrix")


# ---------------------------------------------------------------------------
# Figure 2 — Per-class precision / recall / F1
# ---------------------------------------------------------------------------

def plot_per_class_metrics(report: dict, output_dir: Path) -> None:
    """Grouped bar chart showing precision, recall, and F1 per class."""
    cm = report.get("confusion_matrix", {})
    precision = cm.get("precision", [])
    recall = cm.get("recall", [])
    f1 = cm.get("f1", [])
    if not (precision and recall and f1):
        print("  Skipping per-class metrics: incomplete confusion matrix data")
        return

    labels = cm.get("labels", CLASS_NAMES)
    n = len(labels)
    x = np.arange(n)
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_p = ax.bar(x - width, [v * 100 for v in precision], width,
                    label="Precision", color=_METRIC_COLORS["Precision"],
                    edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x,         [v * 100 for v in recall],    width,
                    label="Recall",    color=_METRIC_COLORS["Recall"],
                    edgecolor="white", linewidth=0.5)
    bars_f = ax.bar(x + width, [v * 100 for v in f1],        width,
                    label="F1",        color=_METRIC_COLORS["F1"],
                    edgecolor="white", linewidth=0.5)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    summary = report.get("summary", {})
    acc = summary.get("accuracy", None)
    title = "Per-Class Metrics: Precision / Recall / F1"
    if acc is not None:
        title += f"  (overall accuracy {acc*100:.1f}%)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir / "per_class_metrics")


# ---------------------------------------------------------------------------
# Figure 3 — Reliability diagram (calibration)
# ---------------------------------------------------------------------------

def plot_reliability_diagram(report: dict, output_dir: Path) -> None:
    """Calibration reliability diagram from binned confidence/accuracy data."""
    cal = report.get("calibration", {})
    bins = [b for b in cal.get("bins", []) if b.get("count", 0) > 0]
    if not bins:
        print("  Skipping reliability diagram: no calibration bins with data")
        return

    conf = [b["mean_confidence"] for b in bins]
    acc  = [b["accuracy"]         for b in bins]
    counts = [b["count"]          for b in bins]
    ece = cal.get("ece", None)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Perfect calibration")

    # Gap shading
    for c, a in zip(conf, acc):
        ax.fill_between([c - 0.05, c + 0.05], [c - 0.05, c + 0.05],
                         [a - 0.05, a + 0.05], alpha=0.15,
                         color="#F44336" if a < c else "#4CAF50")

    # Scatter sized by bin count
    max_count = max(counts)
    sizes = [max(30, 300 * cnt / max_count) for cnt in counts]
    sc = ax.scatter(conf, acc, s=sizes, color="#2196F3", edgecolors="white",
                    linewidth=1.5, zorder=5, label="Confidence bins")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)

    title = "Reliability Diagram"
    if ece is not None:
        title += f"  (ECE = {ece:.4f})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate each point with its count
    for c, a, cnt in zip(conf, acc, counts):
        ax.annotate(f"n={cnt}", (c, a),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, color="#555")

    plt.tight_layout()
    _save(fig, output_dir / "reliability_diagram")


# ---------------------------------------------------------------------------
# Figure 4 — Selective accuracy (accuracy vs coverage)
# ---------------------------------------------------------------------------

def plot_selective_accuracy(report: dict, output_dir: Path) -> None:
    """Accuracy-coverage trade-off curve across confidence thresholds."""
    sel = report.get("selective_accuracy", {})
    thresholds = sel.get("thresholds", [])
    if not thresholds:
        print("  Skipping selective accuracy: no threshold data in report")
        return

    coverages = [t["coverage"] * 100 for t in thresholds]
    accuracies = [t["accuracy"] * 100 for t in thresholds]
    thresh_vals = [t["threshold"]     for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, accuracies, "o-", color="#2196F3",
            linewidth=2.5, markersize=8, zorder=3)

    # Label each point with its threshold
    for cov, acc, thr in zip(coverages, accuracies, thresh_vals):
        ax.annotate(f"τ={thr}", (cov, acc),
                    textcoords="offset points", xytext=(4, 6),
                    fontsize=8, color="#444")

    # Full-coverage accuracy reference
    full_acc = accuracies[0] if coverages[0] >= 99 else None
    if full_acc is not None:
        ax.axhline(full_acc, color="#BDBDBD", linestyle=":", linewidth=1.5,
                   label=f"Full-coverage accuracy ({full_acc:.1f}%)")

    ax.set_xlabel("Coverage (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Selective Accuracy vs. Coverage", fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=max(0, min(accuracies) - 5))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()   # high coverage on the left, selective on the right

    plt.tight_layout()
    _save(fig, output_dir / "selective_accuracy")


# ---------------------------------------------------------------------------
# Figure 5 — Predicted probability distributions per true class
# ---------------------------------------------------------------------------

def plot_probability_distributions(report: dict, output_dir: Path) -> None:
    """
    For each true class, show the mean predicted probability assigned to each
    class — gives an intuitive picture of confusion patterns.
    """
    prob_dist = report.get("probability_distributions", {})
    key_map = {"true_mcar": "MCAR", "true_mar": "MAR", "true_mnar": "MNAR"}
    rows = {label: prob_dist[key]["mean_p"]
            for key, label in key_map.items()
            if key in prob_dist and "mean_p" in prob_dist[key]}

    if not rows:
        print("  Skipping probability distributions: no data in report")
        return

    true_labels = list(rows.keys())
    n = len(true_labels)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (class_name, color) in enumerate(CLASS_COLORS.items()):
        vals = [rows[tl][idx] * 100 if tl in rows else 0 for tl in true_labels]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=f"P({class_name})",
                      color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, vals):
            if val >= 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"True {l}" for l in true_labels], fontsize=12)
    ax.set_ylabel("Mean predicted probability (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title("Mean Predicted Probabilities by True Class",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir / "probability_distributions")


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save(fig, stem: Path) -> None:
    """Save figure as both PDF and PNG at 300 dpi."""
    for ext in ("pdf", "png"):
        out = Path(f"{stem}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem.name}.pdf/png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(run_dir: Path, output_dir: Path, report_path: Path | None = None) -> None:
    """Load report and run all figure generators."""
    report = load_report(run_dir, report_path)

    ckpt = report.get("checkpoint", "?")
    acc  = report.get("summary", {}).get("accuracy")
    tag  = f"accuracy={acc*100:.2f}%" if acc is not None else ""
    print(f"  Report:     {Path(ckpt).name}  {tag}")
    print(f"  Output dir: {output_dir}")

    plot_confusion_matrix(report, output_dir)
    plot_per_class_metrics(report, output_dir)
    plot_reliability_diagram(report, output_dir)
    plot_selective_accuracy(report, output_dir)
    plot_probability_distributions(report, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-run figures for dissertation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--run-dir", required=True, type=str, metavar="PATH",
        help="Experiment run directory (must contain eval_report.json)",
    )
    p.add_argument(
        "--output-dir", type=str, default=None, metavar="PATH",
        help="Where to save figures (default: <run-dir>/figures/)",
    )
    p.add_argument(
        "--report", type=str, default=None, metavar="PATH",
        help="Explicit path to eval_report.json (overrides automatic search)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not HAS_DEPS:
        print("Error: matplotlib and numpy are required.")
        print("  pip install matplotlib numpy")
        sys.exit(1)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report) if args.report else None
    generate_all(run_dir, output_dir, report_path)
    print(f"\nDone. All figures in: {output_dir}")


if __name__ == "__main__":
    main()
