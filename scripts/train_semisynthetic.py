#!/usr/bin/env python3
"""
Training script for semi-synthetic data.

Uses real datasets with synthetic missingness mechanisms.

Usage:
    python scripts/train_semisynthetic.py --config configs/training/semisynthetic.yaml
    python scripts/train_semisynthetic.py --config configs/training/semisynthetic.yaml --quiet
    python scripts/train_semisynthetic.py --config configs/training/semisynthetic.yaml --quiet --report
    python scripts/train_semisynthetic.py --config configs/training/semisynthetic.yaml --name my_experiment
    python scripts/train_semisynthetic.py --config configs/training/semisynthetic.yaml --mnar-variants self_censoring
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config import LacunaConfig, load_config, save_config
from lacuna.models import create_lacuna_model
from lacuna.generators import load_registry_from_config
from lacuna.generators.priors import GeneratorPrior
from lacuna.data import (
    create_default_catalog,
    SemiSyntheticDataLoader,
)
from lacuna.training import (
    Trainer,
    TrainerConfig,
    CheckpointData,
    save_checkpoint,
    create_logger,
    generate_eval_report,
    save_raw_predictions,
    print_eval_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lacuna on semi-synthetic data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--generators", type=str, default=None,
                        help="Generator config name or YAML path (overrides training config)")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal console output (epoch summaries only). "
                             "Warnings and early stopping are always printed.")
    parser.add_argument("--report", action="store_true",
                        help="Generate detailed evaluation report after training "
                             "(eval_report.json + predictions.pt + journal entry)")
    parser.add_argument("--mnar-variants", type=str, nargs="+", default=None,
                        help="Override MNAR expert variants "
                             "(default: self_censoring threshold latent). "
                             "Use a single value for 1/1/1 ablation, e.g.: "
                             "--mnar-variants self_censoring")
    parser.add_argument("--journal", type=str, default=None,
                        help="Path to experiment journal (default: experiments/JOURNAL.md). "
                             "Set to 'none' to disable auto-journaling.")
    return parser.parse_args()


def setup_experiment(config: LacunaConfig, name: str = None) -> Path:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = name or f"lacuna_semisyn_{timestamp}"

    output_dir = Path(config.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    return output_dir


def load_raw_datasets(catalog, dataset_names: list, max_cols: int, quiet: bool = False):
    """Load raw datasets from catalog, filtering by max_cols."""
    datasets = []
    skipped = []

    for name in dataset_names:
        try:
            raw = catalog.load(name)
            if raw.d <= max_cols:
                datasets.append(raw)
                if not quiet:
                    print(f"    Loaded: {name} ({raw.n} samples, {raw.d} features)")
            else:
                skipped.append((name, raw.d))
        except Exception as e:
            # Warnings always printed
            print(f"    Warning: Could not load '{name}': {e}")

    if skipped:
        print(f"    Skipped (too many columns): {skipped}")

    return datasets


def append_journal_entry(journal_path: Path, report: dict, exp_dir: Path,
                         mnar_variants: list, exp_name: str = None):
    """Append a formatted journal entry to JOURNAL.md from eval report data."""
    # Import the journal entry generator
    from journal_entry import generate_entry

    # Check for calibration data
    calibration = None
    cal_path = exp_dir / "checkpoints" / "calibrated.json"
    if cal_path.exists():
        with open(cal_path) as f:
            calibration = json.load(f)

    # Generate the markdown entry
    entry = generate_entry(
        report=report,
        calibration=calibration,
        name=exp_name,
    )

    # Add architecture note if non-default expert structure
    if mnar_variants and mnar_variants != ["self_censoring", "threshold", "latent"]:
        n_experts = 2 + len(mnar_variants)
        expert_map = [0, 1] + [2] * len(mnar_variants)
        arch_note = (
            f"\n**Architecture:** {n_experts} experts "
            f"(expert_to_class = {expert_map}, "
            f"mnar_variants = {mnar_variants})\n"
        )
        # Insert after the date line
        entry = entry.replace("\n### Results", f"{arch_note}\n### Results", 1)

    # Append to journal
    separator = "\n---\n\n"

    if journal_path.exists():
        content = journal_path.read_text()
        # Find the "Planned Experiments" section and insert before it
        marker = "## Planned Experiments"
        if marker in content:
            idx = content.index(marker)
            new_content = (
                content[:idx].rstrip() + "\n\n" + separator +
                entry + "\n" + separator +
                content[idx:]
            )
            journal_path.write_text(new_content)
        else:
            # Just append at the end
            with open(journal_path, "a") as f:
                f.write(separator + entry + "\n")
    else:
        # Create new journal with this entry
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(journal_path, "w") as f:
            f.write("# Lacuna Experiment Journal\n\n")
            f.write(entry + "\n")

    return journal_path


def main():
    args = parse_args()
    quiet = args.quiet

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed

    # Resolve MNAR variants
    mnar_variants = args.mnar_variants  # None means use default (3 variants)

    # Setup experiment
    exp_dir = setup_experiment(config, args.name)

    # Save config
    save_config(config, exp_dir / "config.yaml")

    # Also save experiment metadata for reproducibility
    exp_meta = {
        "mnar_variants": mnar_variants or ["self_censoring", "threshold", "latent"],
        "prior": "class_balanced",
        "loss": "cross_entropy",
        "label_smoothing": 0.0,
        "config_path": args.config,
        "generators": args.generators or config.generator.config_path or config.generator.config_name,
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "experiment_meta.json", "w") as f:
        json.dump(exp_meta, f, indent=2)

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create generator registry
    generators_name = args.generators or config.generator.config_path or config.generator.config_name
    registry = load_registry_from_config(generators_name)
    prior = GeneratorPrior.class_balanced(registry)

    # Load datasets
    catalog = create_default_catalog()
    train_dataset_names = config.data.train_datasets or ["diabetes", "wine", "breast_cancer"]
    val_dataset_names = config.data.val_datasets or ["iris"]

    # Expert structure info
    n_experts = 2 + len(mnar_variants or ["self_censoring", "threshold", "latent"])
    expert_desc = f"{n_experts} experts"
    if mnar_variants:
        expert_desc += f" (mnar={mnar_variants})"

    if quiet:
        # Single-line setup summary
        print(f"LACUNA Semi-Synthetic | {registry.K} generators | {expert_desc} | "
              f"hidden={config.model.hidden_dim} layers={config.model.n_layers} | "
              f"{config.device}")
    else:
        print("=" * 60)
        print("LACUNA TRAINING (Semi-Synthetic)")
        print("=" * 60)
        print(f"\nLoading config: {args.config}")
        print(f"Experiment directory: {exp_dir}")
        print(f"\nConfiguration:")
        print(f"  Device: {config.device}")
        print(f"  Seed: {config.seed}")
        print(f"  Model: hidden={config.model.hidden_dim}, layers={config.model.n_layers}, heads={config.model.n_heads}")
        print(f"  Experts: {expert_desc}")
        print(f"  Training: epochs={config.training.epochs}, lr={config.training.lr}, batch={config.training.batch_size}")
        print(f"  Data: max_rows={config.data.max_rows}, max_cols={config.data.max_cols}")
        print(f"\nSetting seed: {config.seed}")
        print(f"\nLoading generator registry: {generators_name}")
        print(f"  Generators: {registry.K}")
        print(f"  Class distribution: {registry.class_counts()}")
        print("\nLoading datasets...")
        print("  Training datasets:")

    train_raw = load_raw_datasets(catalog, train_dataset_names, config.data.max_cols, quiet=quiet)

    if not quiet:
        print("  Validation datasets:")
    val_raw = load_raw_datasets(catalog, val_dataset_names, config.data.max_cols, quiet=quiet)

    if len(train_raw) == 0:
        print("\nError: No training datasets loaded. Exiting.")
        return

    if len(val_raw) == 0:
        # Warning always printed
        print("\nWarning: No validation datasets. Using last training dataset.")
        val_raw = [train_raw[-1]]
        train_raw = train_raw[:-1]

    if quiet:
        print(f"Datasets: {len(train_raw)} train, {len(val_raw)} val | "
              f"epochs={config.training.epochs} batch={config.training.batch_size}")

    # Create data loaders
    train_loader = SemiSyntheticDataLoader(
        raw_datasets=train_raw,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=config.training.batches_per_epoch,
        seed=config.seed,
    )

    val_loader = SemiSyntheticDataLoader(
        raw_datasets=val_raw,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=config.training.val_batches,
        seed=config.seed + 1000000,
    )

    if not quiet:
        print(f"\nCreating data loaders...")
        print(f"  Train: {len(train_loader)} batches/epoch")
        print(f"  Val: {len(val_loader)} batches")

    # Create model (with optional mnar_variants override)
    model = create_lacuna_model(
        hidden_dim=config.model.hidden_dim,
        evidence_dim=config.model.evidence_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        max_cols=config.data.max_cols,
        dropout=config.model.dropout,
        mnar_variants=mnar_variants,
    )

    if not quiet:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nCreating model...")
        print(f"  Parameters: {n_params:,} ({n_trainable:,} trainable)")

    # Create trainer
    trainer_config = TrainerConfig(
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
        epochs=config.training.epochs,
        warmup_steps=config.training.warmup_steps,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        checkpoint_dir=str(exp_dir / "checkpoints"),
        save_best_only=True,
        quiet=quiet,
    )

    logger = create_logger(exp_dir)
    trainer = Trainer(model, trainer_config, device=config.device, log_fn=logger)

    if not quiet:
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

    # Train!
    start_time = time.time()
    result = trainer.fit(train_loader, val_loader)
    total_time = time.time() - start_time

    # Final results — always printed
    if quiet:
        print(f"\nBest: {result['best_val_acc']*100:.1f}% @ epoch {result['best_epoch']} | "
              f"Time: {total_time:.1f}s ({total_time/60:.1f}m)")
    else:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Epochs completed: {result['final_epoch'] + 1}")
        print(f"  Total steps: {result['total_steps']}")
        print(f"  Best val loss: {result['best_val_loss']:.4f}")
        print(f"  Best val acc: {result['best_val_acc']*100:.1f}%")
        print(f"  Training time: {total_time:.1f}s ({total_time/60:.1f}m)")

    # Save final checkpoint
    final_ckpt = CheckpointData(
        model_state=model.state_dict(),
        step=result['total_steps'],
        epoch=result['final_epoch'],
        best_val_loss=result['best_val_loss'],
        best_val_acc=result['best_val_acc'],
        metrics={
            "training_time": total_time,
            "data_mode": "semisynthetic",
            "train_datasets": [ds.name for ds in train_raw],
            "val_datasets": [ds.name for ds in val_raw],
            "mnar_variants": mnar_variants or ["self_censoring", "threshold", "latent"],
        },
    )
    save_checkpoint(final_ckpt, exp_dir / "checkpoints" / "final.pt")

    if not quiet:
        print(f"\nCheckpoints saved to: {exp_dir / 'checkpoints'}")

    # Generate detailed evaluation report (optional, via --report)
    if args.report:
        print("\nGenerating evaluation report...")
        report_start = time.time()
        detailed_result = trainer.validate_detailed(val_loader)
        report_time = time.time() - report_start

        report = generate_eval_report(
            result=detailed_result,
            registry=registry,
            checkpoint_path=str(exp_dir / "checkpoints" / "best_model.pt"),
            config_path=args.config,
        )
        report["training_time_seconds"] = round(total_time, 2)
        report["report_eval_time_seconds"] = round(report_time, 2)
        report["train_datasets"] = [ds.name for ds in train_raw]
        report["val_datasets"] = [ds.name for ds in val_raw]
        report["mnar_variants"] = mnar_variants or ["self_censoring", "threshold", "latent"]
        report["n_experts"] = n_experts
        report["experiment_dir"] = str(exp_dir)

        # Save JSON report
        json_path = exp_dir / "eval_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save raw predictions
        pt_path = exp_dir / "predictions.pt"
        save_raw_predictions(
            p_class=detailed_result.all_p_class,
            true_class=detailed_result.all_true_class,
            generator_ids=detailed_result.all_generator_ids,
            path=pt_path,
        )

        # Print summary
        print_eval_summary(report)
        print(f"\nReport: {json_path}")
        print(f"Predictions: {pt_path}")

        # Auto-append journal entry
        journal_opt = args.journal
        if journal_opt != "none":
            if journal_opt:
                journal_path = Path(journal_opt)
            else:
                # Default: experiments/JOURNAL.md relative to project root
                journal_path = PROJECT_ROOT / "experiments" / "JOURNAL.md"

            try:
                saved_path = append_journal_entry(
                    journal_path=journal_path,
                    report=report,
                    exp_dir=exp_dir,
                    mnar_variants=mnar_variants or ["self_censoring", "threshold", "latent"],
                    exp_name=args.name,
                )
                print(f"Journal: {saved_path}")
            except Exception as e:
                print(f"Warning: Could not append journal entry: {e}")
    else:
        if not quiet:
            print("\nDone! (use --report to generate detailed evaluation)")
        else:
            print(f"Output: {exp_dir}")


if __name__ == "__main__":
    main()
