#!/usr/bin/env python3
"""
Full Lacuna pipeline: train → evaluate → calibrate → evaluate calibrated → figures.

One command does everything:
    python scripts/run_pipeline.py --config configs/training/semisynthetic_full.yaml
    python scripts/run_pipeline.py --config configs/training/semisynthetic.yaml --device cpu
    python scripts/run_pipeline.py --config configs/training/semisynthetic_full.yaml --name my_experiment

All artifacts (checkpoints, reports, figures, logs) are saved to a single
run directory under the configured output_dir, and the run is registered
in experiments/registry.json automatically.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full Lacuna pipeline: train → calibrate → evaluate → figures",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cpu/cuda)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--mnar-variants", type=str, nargs="+", default=None,
        help="Override MNAR expert variants (e.g. --mnar-variants self_censoring)",
    )
    parser.add_argument(
        "--skip-calibration", action="store_true",
        help="Skip temperature scaling calibration step",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation step",
    )
    return parser.parse_args()


def run_step(name, cmd):
    """Run a pipeline step, printing status and checking for errors."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\n✗ FAILED: {name} (exit code {result.returncode})")
        print(f"  Command: {' '.join(cmd)}")
        return False

    print(f"\n✓ {name} complete")
    return True


def find_run_dir(config_path):
    """Find the most recently created run directory for this config."""
    # Read config to get output_dir
    sys.path.insert(0, str(PROJECT_ROOT))
    from lacuna.config import load_config
    config = load_config(config_path)
    runs_dir = Path(config.output_dir)

    if not runs_dir.exists():
        return None

    # Find most recent directory by modification time
    subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    return max(subdirs, key=lambda d: d.stat().st_mtime)


def generate_figures(run_dir, predictions_path):
    """Generate all figures for a run and save to run_dir/figures/."""
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Loss curves from training logs ---
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    if metrics_path.exists():
        epoch_train = {}
        val_epochs_map = {}

        with open(metrics_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "step":
                    ep = rec["epoch"]
                    if ep not in epoch_train:
                        epoch_train[ep] = []
                    epoch_train[ep].append(rec["total_loss"])
                elif rec.get("type") == "validation" and "epoch" in rec:
                    val_epochs_map[rec["epoch"]] = rec["val_loss"]

        if epoch_train and val_epochs_map:
            train_eps = sorted(epoch_train.keys())
            train_means = [np.mean(epoch_train[e]) for e in train_eps]
            val_eps = sorted(val_epochs_map.keys())
            val_losses = [val_epochs_map[e] for e in val_eps]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(train_eps, train_means, color="#2563eb", linewidth=1.8,
                    label="Train Loss", alpha=0.9)
            ax.plot(val_eps, val_losses, color="#dc2626", linewidth=1.8,
                    label="Validation Loss", alpha=0.9)

            best_idx = int(np.argmin(val_losses))
            ax.scatter([val_eps[best_idx]], [val_losses[best_idx]],
                       color="#dc2626", s=60, zorder=5,
                       edgecolors="white", linewidth=1.5)
            ax.annotate(
                f"Best: {val_losses[best_idx]:.4f} (epoch {val_eps[best_idx]})",
                xy=(val_eps[best_idx], val_losses[best_idx]),
                xytext=(val_eps[best_idx] + 1, val_losses[best_idx] + 0.015),
                fontsize=9, color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
            )
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title("Training Loss vs. Validation Loss",
                         fontsize=14, fontweight="bold")
            ax.legend(fontsize=11, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()

            fig.savefig(figures_dir / "loss_curves.pdf",
                        dpi=300, bbox_inches="tight")
            fig.savefig(figures_dir / "loss_curves.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {figures_dir}/loss_curves.pdf/png")

    # --- ROC curves from predictions ---
    if predictions_path and predictions_path.exists():
        roc_cmd = [
            PYTHON, str(PROJECT_ROOT / "scripts" / "generate_roc_curves.py"),
            "--predictions", str(predictions_path),
            "--output-dir", str(figures_dir),
        ]
        subprocess.run(roc_cmd, cwd=str(PROJECT_ROOT))
        print(f"  ROC curves: {figures_dir}/roc_curves.pdf/png")


def main():
    args = parse_args()

    # NON-DETERMINISTIC: wall-clock timing for pipeline duration
    pipeline_start = datetime.now()
    print(f"Lacuna Pipeline — {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")

    # === Step 1: Train with auto-report ===
    train_cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_semisynthetic.py"),
        "--config", args.config,
        "--quiet",
        "--report",
    ]
    if args.name:
        train_cmd.extend(["--name", args.name])
    if args.device:
        train_cmd.extend(["--device", args.device])
    if args.seed is not None:
        train_cmd.extend(["--seed", str(args.seed)])
    if args.mnar_variants:
        train_cmd.extend(["--mnar-variants"] + args.mnar_variants)

    if not run_step("Train + Evaluate", train_cmd):
        sys.exit(1)

    # Find the run directory that was just created
    run_dir = find_run_dir(args.config)
    if run_dir is None:
        print("✗ Could not find run directory after training")
        sys.exit(1)

    print(f"\nRun directory: {run_dir}")

    # Find the saved config (which may differ from the input config)
    saved_config = run_dir / "config.yaml"
    config_for_eval = str(saved_config) if saved_config.exists() else args.config

    # Find checkpoint
    best_ckpt = run_dir / "checkpoints" / "best_model.pt"
    if not best_ckpt.exists():
        print(f"✗ No best_model.pt found in {run_dir}/checkpoints/")
        sys.exit(1)

    # === Step 2: Calibrate ===
    if not args.skip_calibration:
        cal_cmd = [
            PYTHON, str(PROJECT_ROOT / "scripts" / "calibrate.py"),
            "--checkpoint", str(best_ckpt),
            "--config", config_for_eval,
        ]
        if args.device:
            cal_cmd.extend(["--device", args.device])

        if run_step("Temperature Scaling", cal_cmd):
            # === Step 3: Evaluate calibrated model ===
            calibrated_ckpt = run_dir / "checkpoints" / "calibrated.pt"
            if calibrated_ckpt.exists():
                eval_cmd = [
                    PYTHON, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                    "--checkpoint", str(calibrated_ckpt),
                    "--config", config_for_eval,
                ]
                if args.device:
                    eval_cmd.extend(["--device", args.device])

                run_step("Evaluate Calibrated Model", eval_cmd)
        else:
            print("  Calibration failed, continuing with uncalibrated model")

    # === Step 4: Generate figures ===
    if not args.skip_figures:
        print(f"\n{'='*60}")
        print(f"  STEP: Generate Figures")
        print(f"{'='*60}\n")

        # Find predictions file
        predictions = run_dir / "predictions.pt"
        if not predictions.exists():
            predictions = run_dir / "checkpoints" / "predictions.pt"

        generate_figures(run_dir, predictions if predictions.exists() else None)
        print(f"\n✓ Figures saved to {run_dir}/figures/")

    # === Summary ===
    # NON-DETERMINISTIC: wall-clock timing
    elapsed = (datetime.now() - pipeline_start).total_seconds()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — {minutes}m {seconds}s")
    print(f"{'='*60}")
    print(f"\nAll artifacts: {run_dir}")
    print(f"  Checkpoints:  {run_dir}/checkpoints/")
    print(f"  Logs:         {run_dir}/logs/")
    print(f"  Figures:      {run_dir}/figures/")
    print(f"  Eval report:  {run_dir}/eval_report.json")

    # Show registry entry if available
    registry_path = PROJECT_ROOT / "experiments" / "registry.json"
    if registry_path.exists():
        try:
            from lacuna.experiments import RunRegistry
            reg = RunRegistry(registry_path)
            reg.load()
            entry = reg.find_by_folder(run_dir.name)
            if entry:
                print(f"\n  Registry: {entry.run_id}")
                if entry.metrics.get("accuracy"):
                    print(f"  Accuracy: {entry.metrics['accuracy']:.1%}")
        except Exception:
            pass

    print(f"\nTo inspect this run:")
    print(f"  ls {run_dir}/figures/")
    print(f"  cat {run_dir}/eval_report.json | python -m json.tool")


if __name__ == "__main__":
    main()
