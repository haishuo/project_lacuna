#!/usr/bin/env python3
"""
Generate ROC / AUC curves for the Lacuna missing-data mechanism classifier.

For a 3-class problem (MCAR / MAR / MNAR), ROC curves are produced using
the one-vs-rest (OvR) strategy: each mechanism class is treated as the
positive class against the combined other two.  A macro-average curve is
also computed by interpolating the three OvR curves onto a shared FPR grid
and averaging.

Two usage modes
---------------
Mode 1 — from a saved predictions.pt file (fastest, no GPU needed):

    python scripts/generate_roc_curves.py \\
        --predictions /path/to/predictions.pt

Mode 2 — run evaluation inline then plot (requires a checkpoint + config):

    python scripts/generate_roc_curves.py \\
        --checkpoint /path/to/calibrated.pt \\
        --config configs/training/semisynthetic_full.yaml \\
        --generators lacuna_tabular_110 \\
        --device cuda

Options
-------
--output-dir   Where to save the figures (default: docs/figures)
--batches      Number of eval batches for inline evaluation (default: 50)
--mnar-variants  MNAR expert variants, must match checkpoint (default: self_censoring)

Outputs
-------
docs/figures/roc_curves.pdf   — vector, for LaTeX / dissertation
docs/figures/roc_curves.png   — raster 300 dpi, for README / web
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional dependencies — checked at runtime
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend — safe for servers
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.metrics import roc_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Visual style — matches the existing dissertation figure palette
# ---------------------------------------------------------------------------

CLASS_NAMES  = ["MCAR", "MAR", "MNAR"]
CLASS_COLORS = {
    "MCAR":  "#2196F3",  # blue
    "MAR":   "#F44336",  # red
    "MNAR":  "#4CAF50",  # green
    "macro": "#9C27B0",  # purple
}
CLASS_LINESTYLES = {
    "MCAR":  "-",
    "MAR":   "-",
    "MNAR":  "-",
    "macro": "--",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_from_pt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load predictions.pt saved by evaluate.py / save_raw_predictions().

    Returns
    -------
    p_class   : float32 ndarray shape [N, 3]  — softmax probabilities
    true_class: int64   ndarray shape [N]     — ground-truth labels (0/1/2)
    """
    data = torch.load(path, weights_only=True)
    p_class    = data["p_class"].cpu().float().numpy()
    true_class = data["true_class"].cpu().long().numpy()
    return p_class, true_class


def run_inline_evaluation(
    checkpoint: str,
    config_path: str,
    generators_name: str | None,
    device: str | None,
    n_batches: int,
    mnar_variants: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load model + data pipeline, run validation, return (p_class, true_class).

    Mirrors the logic of scripts/evaluate.py so callers don't need to run
    evaluate.py separately before plotting.
    """
    from lacuna.config import load_config
    from lacuna.models import create_lacuna_model
    from lacuna.generators import load_registry_from_config
    from lacuna.generators.priors import GeneratorPrior
    from lacuna.data import create_default_catalog, SemiSyntheticDataLoader
    from lacuna.training import Trainer, TrainerConfig, load_model_weights

    config = load_config(config_path)
    if device:
        config.device = device

    gen_name = generators_name or config.generator.config_path or config.generator.config_name
    registry = load_registry_from_config(gen_name)
    prior    = GeneratorPrior.uniform(registry)

    catalog     = create_default_catalog()
    dataset_names = config.data.val_datasets or ["iris"]
    eval_raw = []
    for name in dataset_names:
        try:
            raw = catalog.load(name)
            if raw.d <= config.data.max_cols:
                eval_raw.append(raw)
        except Exception as e:
            print(f"  Warning: could not load '{name}': {e}", file=sys.stderr)

    if not eval_raw:
        raise RuntimeError("No evaluation datasets could be loaded.")

    eval_loader = SemiSyntheticDataLoader(
        raw_datasets=eval_raw,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=n_batches,
        seed=config.seed + 9999999,
    )

    model = create_lacuna_model(
        hidden_dim=config.model.hidden_dim,
        evidence_dim=config.model.evidence_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        max_cols=config.data.max_cols,
        dropout=config.model.dropout,
        mnar_variants=mnar_variants,
    )
    load_model_weights(model, checkpoint, device=config.device)

    trainer = Trainer(
        model,
        TrainerConfig(lr=1e-4, epochs=1, warmup_steps=0, patience=999),
        device=config.device,
    )

    result = trainer.validate_detailed(eval_loader)
    p_class    = result.all_p_class.cpu().float().numpy()
    true_class = result.all_true_class.cpu().long().numpy()
    return p_class, true_class


# ---------------------------------------------------------------------------
# ROC / AUC computation
# ---------------------------------------------------------------------------

def compute_roc_curves(
    p_class: np.ndarray,
    true_class: np.ndarray,
) -> dict:
    """
    Compute one-vs-rest ROC curves for each class plus a macro average.

    Parameters
    ----------
    p_class   : [N, 3] predicted probabilities
    true_class: [N]    integer ground-truth labels (0=MCAR, 1=MAR, 2=MNAR)

    Returns
    -------
    dict with keys: "MCAR", "MAR", "MNAR", "macro"
    Each value is a dict: {"fpr": array, "tpr": array, "auc": float}
    """
    n_classes = p_class.shape[1]
    curves = {}

    # --- per-class one-vs-rest curves ---
    for c, name in enumerate(CLASS_NAMES):
        y_binary = (true_class == c).astype(int)
        y_score  = p_class[:, c]

        # roc_curve requires at least one positive and one negative sample
        if y_binary.sum() == 0 or (1 - y_binary).sum() == 0:
            print(f"  Warning: class {name} has no positive or no negative samples — skipping.")
            continue

        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc     = auc(fpr, tpr)
        curves[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    # --- macro-average ROC ---
    # Interpolate all per-class curves onto a shared FPR grid, then average TPRs
    if len(curves) == n_classes:
        mean_fpr = np.linspace(0, 1, 500)
        tprs_interp = []
        for name in CLASS_NAMES:
            interp_tpr = np.interp(mean_fpr, curves[name]["fpr"], curves[name]["tpr"])
            interp_tpr[0] = 0.0  # anchor at origin
            tprs_interp.append(interp_tpr)

        mean_tpr     = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0          # anchor at (1, 1)
        macro_auc    = auc(mean_fpr, mean_tpr)
        curves["macro"] = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": macro_auc}

    return curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curves(
    curves: dict,
    n_samples: int,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """
    Render ROC curves to PDF and PNG.

    Layout
    ------
    Main panel  — all three OvR ROC curves + macro average
    Inset panel — zoomed to the top-left corner (FPR ≤ 0.20) to show
                  the separation between curves at low false-positive rates

    Parameters
    ----------
    curves       : output of compute_roc_curves()
    n_samples    : total number of evaluation samples (for subtitle)
    output_dir   : directory to write roc_curves.{pdf,png}
    title_suffix : optional extra text appended to the figure title
    """
    fig, ax_main = plt.subplots(figsize=(8, 7))

    # --- Chance-level diagonal ---
    ax_main.plot(
        [0, 1], [0, 1],
        linestyle=":",
        color="#BDBDBD",
        linewidth=1.5,
        label="Chance level (AUC = 0.50)",
        zorder=1,
    )

    # --- Per-class curves ---
    draw_order = CLASS_NAMES + ["macro"]
    linewidths = {"MCAR": 2.2, "MAR": 2.2, "MNAR": 2.2, "macro": 2.0}
    alphas     = {"MCAR": 0.95, "MAR": 0.95, "MNAR": 0.95, "macro": 0.80}

    for label in draw_order:
        if label not in curves:
            continue
        c    = curves[label]
        disp = "Macro avg" if label == "macro" else label
        ax_main.plot(
            c["fpr"], c["tpr"],
            color=CLASS_COLORS[label],
            linestyle=CLASS_LINESTYLES[label],
            linewidth=linewidths[label],
            alpha=alphas[label],
            label=f"{disp} (AUC = {c['auc']:.3f})",
            zorder=3 if label != "macro" else 2,
        )

    # --- Inset: zoomed top-left (low FPR region) ---
    ax_inset = ax_main.inset_axes(
        [0.42, 0.08, 0.54, 0.50],   # [x, y, width, height] in axes fraction
        xlim=(-0.01, 0.22),
        ylim=(0.60, 1.01),
    )
    ax_inset.plot([0, 1], [0, 1], linestyle=":", color="#BDBDBD", linewidth=1.0)
    for label in CLASS_NAMES:
        if label not in curves:
            continue
        c = curves[label]
        ax_inset.plot(
            c["fpr"], c["tpr"],
            color=CLASS_COLORS[label],
            linestyle=CLASS_LINESTYLES[label],
            linewidth=1.8,
            alpha=0.90,
        )
    ax_inset.set_xlabel("FPR", fontsize=8)
    ax_inset.set_ylabel("TPR", fontsize=8)
    ax_inset.set_title("Low-FPR detail", fontsize=8, style="italic")
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(alpha=0.25)
    ax_main.indicate_inset_zoom(ax_inset, edgecolor="#757575", linewidth=0.8)

    # --- Axes decoration ---
    ax_main.set_xlim(-0.01, 1.01)
    ax_main.set_ylim(-0.01, 1.05)
    ax_main.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=13)
    ax_main.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)

    title = "ROC Curves — One-vs-Rest (MCAR / MAR / MNAR)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax_main.set_title(title, fontsize=14, fontweight="bold", pad=12)

    ax_main.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax_main.grid(alpha=0.25)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Sample count note
    ax_main.annotate(
        f"n = {n_samples:,} samples",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color="#757575",
        style="italic",
    )

    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = output_dir / f"roc_curves.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ROC / AUC curves for the Lacuna classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input: either a pre-computed predictions file OR a checkpoint ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--predictions", type=str, metavar="PATH",
        help="Path to predictions.pt saved by evaluate.py (no GPU needed)",
    )
    src.add_argument(
        "--checkpoint", type=str, metavar="PATH",
        help="Path to calibrated.pt; requires --config as well",
    )

    # --- Inline evaluation options (only used with --checkpoint) ---
    p.add_argument(
        "--config", type=str, metavar="PATH",
        help="Training config YAML (required with --checkpoint)",
    )
    p.add_argument(
        "--generators", type=str, default=None, metavar="NAME",
        help="Generator config name or YAML path (overrides training config)",
    )
    p.add_argument(
        "--device", type=str, default=None, metavar="cpu|cuda",
        help="Override compute device",
    )
    p.add_argument(
        "--batches", type=int, default=50, metavar="N",
        help="Number of evaluation batches for inline mode (default: 50)",
    )
    p.add_argument(
        "--mnar-variants", type=str, nargs="+", default=["self_censoring"],
        help="MNAR expert variants; must match checkpoint architecture "
             "(default: self_censoring)",
    )

    # --- Output ---
    p.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "docs" / "figures"),
        metavar="DIR",
        help="Directory to write roc_curves.{pdf,png} (default: docs/figures)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Dependency checks ---
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required.  Install with: pip install matplotlib")
        sys.exit(1)
    if not HAS_SKLEARN:
        print("Error: scikit-learn is required.  Install with: pip install scikit-learn")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load / compute predictions ---
    title_suffix = ""

    if args.predictions:
        pt_path = Path(args.predictions)
        if not pt_path.exists():
            print(f"Error: predictions file not found: {pt_path}")
            sys.exit(1)
        print(f"Loading predictions from: {pt_path}")
        p_class, true_class = load_from_pt(pt_path)
        title_suffix = f"source: {pt_path.name}"

    else:  # --checkpoint mode
        if not args.config:
            print("Error: --config is required when using --checkpoint")
            sys.exit(1)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"Error: checkpoint not found: {ckpt_path}")
            sys.exit(1)
        print(f"Running inline evaluation: {ckpt_path.name}")
        p_class, true_class = run_inline_evaluation(
            checkpoint=str(ckpt_path),
            config_path=args.config,
            generators_name=args.generators,
            device=args.device,
            n_batches=args.batches,
            mnar_variants=args.mnar_variants,
        )
        title_suffix = f"checkpoint: {ckpt_path.name}"

    n_samples = len(true_class)
    print(f"Samples: {n_samples:,}  |  class distribution: "
          + "  ".join(f"{n}={int((true_class==i).sum())}" for i, n in enumerate(CLASS_NAMES)))

    # --- Compute ROC curves ---
    print("Computing ROC curves (one-vs-rest)...")
    curves = compute_roc_curves(p_class, true_class)

    for label, c in curves.items():
        tag = "macro-avg" if label == "macro" else f"{label} OvR"
        print(f"  {tag:12s}  AUC = {c['auc']:.4f}")

    # --- Plot and save ---
    print(f"\nRendering figures → {output_dir}/")
    plot_roc_curves(
        curves=curves,
        n_samples=n_samples,
        output_dir=output_dir,
        title_suffix=title_suffix,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
