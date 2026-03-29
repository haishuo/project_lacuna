#!/usr/bin/env python3
"""
Generate dissertation-ready figures and tables from consolidated experiment data.

Produces:
  - LaTeX tables (copy-paste into dissertation)
  - CSV tables (for further analysis)
  - Matplotlib figures (PDF + PNG)

Usage:
    python scripts/generate_dissertation_figures.py --data all_experiments.json --output-dir figures/
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import matplotlib — not required for tables
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Figures will be skipped.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =========================================================================
# Data extraction helpers
# =========================================================================

# The canonical experiment progression for the dissertation
EXPERIMENT_ORDER = [5, 7, 8, 9, 10]

EXPERIMENT_LABELS = {
    5: "Exp 5\n(Baseline)",
    7: "Exp 7\n(Brier+Bal.)",
    8: "Exp 8\n(Bal. Prior)",
    9: "Exp 9\n(1/1/1)",
    10: "Exp 10\n(+Temp.)",
}

EXPERIMENT_SHORT = {
    5: "Baseline (1/1/3)",
    7: "Brier + Bal. Prior",
    8: "Bal. Prior Only",
    9: "1/1/1 Symmetric",
    10: "1/1/1 + T=1.96",
}


def extract_experiment_id(name: str) -> int:
    """Extract experiment number from name string."""
    import re
    m = re.search(r"[Ee]xperiment\s+(\d+)", name)
    if m:
        return int(m.group(1))
    return -1


def get_metric(report: dict, *keys, default=None):
    """Safely extract nested metric from report dict."""
    obj = report
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return default
    return obj


def extract_experiment_summary(experiments: list) -> list:
    """Extract key metrics from all experiments into a flat list."""
    rows = []
    for exp in experiments:
        report = exp.get("report", {})
        exp_id = extract_experiment_id(exp["name"])
        if exp_id not in EXPERIMENT_ORDER:
            continue

        summary = report.get("summary", {})
        cm = report.get("confusion_matrix", {})
        cal = report.get("calibration", {})
        conf = report.get("confidence", report.get("confidence_analysis", {}))

        row = OrderedDict([
            ("id", exp_id),
            ("name", EXPERIMENT_SHORT.get(exp_id, exp["name"])),
            ("accuracy", summary.get("accuracy", None)),
            ("mcar_recall", summary.get("mcar_acc", cm.get("recall", [None])[0] if isinstance(cm.get("recall"), list) else None)),
            ("mar_recall", summary.get("mar_acc", cm.get("recall", [None, None])[1] if isinstance(cm.get("recall"), list) and len(cm.get("recall", [])) > 1 else None)),
            ("mnar_recall", summary.get("mnar_acc", cm.get("recall", [None, None, None])[2] if isinstance(cm.get("recall"), list) and len(cm.get("recall", [])) > 2 else None)),
            ("mcar_precision", cm.get("precision", [None])[0] if isinstance(cm.get("precision"), list) else None),
            ("mar_precision", cm.get("precision", [None, None])[1] if isinstance(cm.get("precision"), list) and len(cm.get("precision", [])) > 1 else None),
            ("mnar_precision", cm.get("precision", [None, None, None])[2] if isinstance(cm.get("precision"), list) and len(cm.get("precision", [])) > 2 else None),
            ("mcar_f1", cm.get("f1", [None])[0] if isinstance(cm.get("f1"), list) else None),
            ("mar_f1", cm.get("f1", [None, None])[1] if isinstance(cm.get("f1"), list) and len(cm.get("f1", [])) > 1 else None),
            ("mnar_f1", cm.get("f1", [None, None, None])[2] if isinstance(cm.get("f1"), list) and len(cm.get("f1", [])) > 2 else None),
            ("ece", cal.get("ece", None)),
            ("architecture", report.get("architecture", f"{'1/1/3' if exp_id <= 8 else '1/1/1'}")),
            ("prior", report.get("prior", "uniform" if exp_id in [5, 9, 10] else "class_balanced")),
            ("loss", report.get("loss", "cross_entropy" if exp_id != 7 else "brier")),
            ("temperature", report.get("temperature", 1.96 if exp_id == 10 else 1.0)),
        ])
        rows.append(row)

    rows.sort(key=lambda r: r["id"])
    return rows


# =========================================================================
# Table generators
# =========================================================================

def generate_summary_csv(rows: list, output_path: Path):
    """Write experiment summary as CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {output_path}")


def fmt_pct(v, decimals=1):
    """Format a 0-1 float as percentage string."""
    if v is None:
        return "---"
    return f"{v*100:.{decimals}f}\\%"


def fmt_f(v, decimals=4):
    """Format a float."""
    if v is None:
        return "---"
    return f"{v:.{decimals}f}"


def generate_main_results_latex(rows: list, output_path: Path):
    """Generate the main results comparison table (LaTeX)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment progression: key metrics across ablation studies. "
        r"Best values in \textbf{bold}. Exp~7 changed three variables simultaneously and is included as a negative control.}",
        r"\label{tab:experiment-progression}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(rows) + "}",
        r"\toprule",
    ]

    # Header row
    header = r"\textbf{Metric}"
    for row in rows:
        header += f" & \\textbf{{Exp {row['id']}}}"
    header += r" \\"
    lines.append(header)

    # Sub-header with experiment descriptions
    subheader = r"\textit{Configuration}"
    for row in rows:
        subheader += f" & \\textit{{{row['name']}}}"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    # Find best values for bolding
    metrics = ["accuracy", "mar_recall", "mcar_recall", "mnar_recall", "mar_f1", "ece"]
    best = {}
    for m in metrics:
        vals = [(r["id"], r[m]) for r in rows if r[m] is not None]
        if not vals:
            continue
        if m == "ece":  # Lower is better
            best[m] = min(vals, key=lambda x: x[1])[0]
        else:  # Higher is better
            best[m] = max(vals, key=lambda x: x[1])[0]

    def maybe_bold(metric, row_id, text):
        if best.get(metric) == row_id:
            return f"\\textbf{{{text}}}"
        return text

    # Data rows
    metric_rows = [
        ("Overall accuracy", "accuracy"),
        ("MCAR recall", "mcar_recall"),
        ("MAR recall", "mar_recall"),
        ("MNAR recall", "mnar_recall"),
        ("MCAR precision", "mcar_precision"),
        ("MAR precision", "mar_precision"),
        ("MNAR precision", "mnar_precision"),
        ("MAR F1", "mar_f1"),
        ("ECE", "ece"),
    ]

    for label, key in metric_rows:
        line = label
        for row in rows:
            val = row[key]
            if key == "ece":
                text = fmt_f(val)
            else:
                text = fmt_pct(val)
            text = maybe_bold(key, row["id"], text)
            line += f" & {text}"
        line += r" \\"
        lines.append(line)

    lines.append(r"\midrule")

    # Configuration rows
    config_rows = [
        ("Architecture", "architecture"),
        ("Prior", "prior"),
        ("Loss", "loss"),
        ("Temperature", "temperature"),
    ]
    for label, key in config_rows:
        line = label
        for row in rows:
            val = row[key]
            if isinstance(val, float):
                line += f" & {val:.2f}"
            else:
                line += f" & {val}"
        line += r" \\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX (main table): {output_path}")


def generate_confusion_matrix_latex(cm_data: dict, exp_label: str, output_path: Path):
    """Generate LaTeX confusion matrix for a single experiment."""
    matrix = cm_data.get("matrix", [])
    if not matrix or len(matrix) != 3:
        return

    labels = ["MCAR", "MAR", "MNAR"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Confusion matrix for {exp_label} (rows = true class, columns = predicted class).}}",
        f"\\label{{tab:cm-{exp_label.lower().replace(' ', '-')}}}",
        r"\begin{tabular}{l|rrr|r}",
        r"\toprule",
        r" & \textbf{Pred MCAR} & \textbf{Pred MAR} & \textbf{Pred MNAR} & \textbf{Total} \\",
        r"\midrule",
    ]

    for i, label in enumerate(labels):
        row = matrix[i]
        total = sum(row)
        # Bold the diagonal (correct predictions)
        cells = []
        for j, val in enumerate(row):
            if i == j:
                cells.append(f"\\textbf{{{val}}}")
            else:
                cells.append(str(val))
        lines.append(f"\\textbf{{True {label}}} & {' & '.join(cells)} & {total} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX (CM {exp_label}): {output_path}")


# =========================================================================
# Figure generators (require matplotlib)
# =========================================================================

def plot_metric_progression(rows: list, output_dir: Path):
    """Bar chart showing metric progression across experiments."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    ids = [r["id"] for r in rows]
    labels = [EXPERIMENT_LABELS.get(r["id"], f"Exp {r['id']}") for r in rows]

    metrics = {
        "Overall Accuracy": [r["accuracy"] for r in rows],
        "MAR Recall": [r["mar_recall"] for r in rows],
        "MCAR Recall": [r["mcar_recall"] for r in rows],
        "MNAR Recall": [r["mnar_recall"] for r in rows],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    for ax, (metric_name, values) in zip(axes.flat, metrics.items()):
        vals = [v * 100 if v is not None else 0 for v in values]
        bars = ax.bar(range(len(ids)), vals, color=colors[:len(ids)], edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(metric_name, fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percentage (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Experiment Progression: Key Metrics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"metric_progression.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: metric_progression.pdf/png")


def plot_mar_journey(rows: list, output_dir: Path):
    """Line plot showing MAR recall + precision + F1 across experiments."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    ids = [r["id"] for r in rows]
    x = range(len(ids))
    labels = [f"Exp {r['id']}" for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    recall = [r["mar_recall"] * 100 if r["mar_recall"] else 0 for r in rows]
    precision = [r["mar_precision"] * 100 if r["mar_precision"] else 0 for r in rows]
    f1 = [r["mar_f1"] * 100 if r["mar_f1"] else 0 for r in rows]

    ax.plot(x, recall, "o-", color="#F44336", linewidth=2.5, markersize=10, label="MAR Recall", zorder=3)
    ax.plot(x, precision, "s--", color="#2196F3", linewidth=2, markersize=8, label="MAR Precision", zorder=2)
    ax.plot(x, f1, "D:", color="#4CAF50", linewidth=2, markersize=8, label="MAR F1", zorder=2)

    # Annotations for key points
    for i, (r, p, f) in enumerate(zip(recall, precision, f1)):
        if r > 0:
            ax.annotate(f"{r:.1f}%", (i, r), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, color="#F44336", fontweight="bold")

    # Highlight the two key improvements
    ax.axvspan(2.5, 3.5, alpha=0.1, color="#4CAF50", label="_nolegend_")
    ax.axvspan(3.5, 4.5, alpha=0.1, color="#FF9800", label="_nolegend_")
    ax.annotate("Architecture\nfix", xy=(3, 5), fontsize=9, ha="center",
                color="#4CAF50", fontstyle="italic")
    ax.annotate("Temperature\nscaling", xy=(4, 5), fontsize=9, ha="center",
                color="#FF9800", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("MAR Detection Performance Across Experiments", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"mar_journey.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: mar_journey.pdf/png")


def plot_calibration_comparison(rows: list, output_dir: Path):
    """ECE comparison across experiments."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    ids = [r["id"] for r in rows]
    labels = [f"Exp {r['id']}" for r in rows]
    ece_vals = [r["ece"] if r["ece"] else 0 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#F44336" if e > 0.15 else "#FF9800" if e > 0.05 else "#4CAF50" for e in ece_vals]
    bars = ax.bar(range(len(ids)), ece_vals, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, ece_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Target line
    ax.axhline(y=0.05, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7, label="Target (ECE < 0.05)")

    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=12)
    ax.set_title("Calibration Improvement Across Experiments", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(ece_vals) * 1.2)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"calibration_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: calibration_comparison.pdf/png")


def plot_confusion_matrix_heatmap(cm_data: dict, title: str, output_path: Path):
    """Plot a confusion matrix as a heatmap."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return

    matrix = np.array(cm_data.get("matrix", []))
    if matrix.size == 0:
        return

    labels = ["MCAR", "MAR", "MNAR"]

    # Normalize by row (recall perspective)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    # Add text annotations
    for i in range(3):
        for j in range(3):
            count = matrix[i][j]
            pct = normalized[i][j] * 100
            color = "white" if normalized[i][j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=11, fontweight="bold" if i == j else "normal", color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"Pred {l}" for l in labels], fontsize=11)
    ax.set_yticklabels([f"True {l}" for l in labels], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalized)", fontsize=10)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{output_path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {output_path.name}.pdf/png")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate dissertation figures and tables from experiment data",
    )
    parser.add_argument(
        "--data", type=str,
        default=str(PROJECT_ROOT / "docs" / "data" / "all_experiments.json"),
        help="Path to consolidated experiment JSON",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "docs" / "figures"),
        help="Output directory for figures and tables",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    # Load data
    if data_path.exists():
        print(f"Loading: {data_path}")
        with open(data_path) as f:
            consolidated = json.load(f)
        experiments = consolidated.get("experiments", [])
    else:
        print(f"No consolidated data at {data_path}")
        print("Run 'python scripts/consolidate_results.py' on Forge first,")
        print("or run this script with --data pointing to the JSON file.")
        print("Generating tables from hard-coded data only...")
        # Import historical data from consolidate_results.py
        try:
            from consolidate_results import HISTORICAL_EXPERIMENTS
        except ImportError:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from consolidate_results import HISTORICAL_EXPERIMENTS
        experiments = [{"name": h["name"], "report": h, "historical": True}
                       for h in HISTORICAL_EXPERIMENTS]

    # Extract summaries
    rows = extract_experiment_summary(experiments)
    print(f"\nFound {len(rows)} experiments with metrics: {[r['id'] for r in rows]}")

    # === Tables ===
    print("\n--- Tables ---")
    generate_summary_csv(rows, output_dir / "tables" / "experiment_summary.csv")
    generate_main_results_latex(rows, output_dir / "tables" / "main_results.tex")

    # Confusion matrices for key experiments
    for exp in experiments:
        exp_id = extract_experiment_id(exp["name"])
        if exp_id in [5, 9, 10]:
            report = exp.get("report", {})
            cm = report.get("confusion_matrix", {})
            if cm.get("matrix"):
                label = f"Exp {exp_id}"
                generate_confusion_matrix_latex(
                    cm, label,
                    output_dir / "tables" / f"confusion_matrix_exp{exp_id}.tex"
                )

    # === Figures ===
    print("\n--- Figures ---")
    if HAS_MATPLOTLIB:
        plot_metric_progression(rows, output_dir)
        plot_mar_journey(rows, output_dir)
        plot_calibration_comparison(rows, output_dir)

        # Confusion matrix heatmaps for key experiments
        for exp in experiments:
            exp_id = extract_experiment_id(exp["name"])
            if exp_id in [5, 9, 10]:
                report = exp.get("report", {})
                cm = report.get("confusion_matrix", {})
                if cm.get("matrix"):
                    plot_confusion_matrix_heatmap(
                        cm,
                        f"Confusion Matrix — Experiment {exp_id}",
                        output_dir / f"confusion_matrix_exp{exp_id}"
                    )
    else:
        print("  (matplotlib not available — skipping figures)")

    # === Summary Stats ===
    print("\n--- Dissertation Summary Statistics ---")
    if len(rows) >= 2:
        first = next((r for r in rows if r["id"] == 5), rows[0])
        last = next((r for r in rows if r["id"] == 10), rows[-1])
        print(f"  Baseline (Exp {first['id']}): {fmt_pct(first['accuracy'])} accuracy, "
              f"{fmt_pct(first['mar_recall'])} MAR recall, ECE={fmt_f(first['ece'])}")
        print(f"  Final (Exp {last['id']}):    {fmt_pct(last['accuracy'])} accuracy, "
              f"{fmt_pct(last['mar_recall'])} MAR recall, ECE={fmt_f(last['ece'])}")
        if first["accuracy"] and last["accuracy"]:
            print(f"  Δ accuracy:  +{(last['accuracy']-first['accuracy'])*100:.1f} points")
        if first["mar_recall"] and last["mar_recall"]:
            print(f"  Δ MAR recall: +{(last['mar_recall']-first['mar_recall'])*100:.1f} points")
        if first["ece"] and last["ece"]:
            print(f"  Δ ECE:       {last['ece']-first['ece']:.4f}")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
