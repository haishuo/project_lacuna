#!/usr/bin/env python3
"""
Evaluate a trained Lacuna model and produce a comprehensive report.

Loads a checkpoint, runs detailed evaluation on validation/test data,
and outputs:
  1. Concise console summary (confusion matrix, accuracy, confidence)
  2. Full JSON report (eval_report.json)
  3. Raw prediction tensors (predictions.pt) for downstream analysis

Usage:
    python scripts/evaluate.py --checkpoint CKPT --config CONFIG
    python scripts/evaluate.py --checkpoint CKPT --config CONFIG --device cpu
    python scripts/evaluate.py --checkpoint CKPT --config CONFIG --output /path/to/report.json
    python scripts/evaluate.py --checkpoint CKPT --config CONFIG --generators lacuna_tabular_110
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config import LacunaConfig, load_config
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
    load_checkpoint,
    load_model_weights,
    generate_eval_report,
    save_raw_predictions,
    print_eval_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Lacuna model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--generators", type=str, default=None,
        help="Generator config name or YAML path (overrides training config)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cpu/cuda)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: {checkpoint_dir}/eval_report.json)",
    )
    parser.add_argument(
        "--batches", type=int, default=None,
        help="Number of evaluation batches (default: from config val_batches)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Override which datasets to evaluate on",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--mnar-variants", type=str, nargs="+", default=None,
        help="Override MNAR expert variants (must match checkpoint architecture)",
    )
    parser.add_argument(
        "--littles-cache", type=str, default=None,
        help="Path to Little's MCAR JSON cache. Required when the model "
             "has include_littles_approx=True. Build via scripts/build_littles_cache.py.",
    )
    return parser.parse_args()


def load_raw_datasets(catalog, dataset_names: list, max_cols: int):
    """Load raw datasets from catalog, filtering by max_cols."""
    datasets = []
    skipped = []

    for name in dataset_names:
        try:
            raw = catalog.load(name)
            if raw.d <= max_cols:
                datasets.append(raw)
            else:
                skipped.append((name, raw.d))
        except Exception as e:
            print(f"  Warning: Could not load '{name}': {e}")

    if skipped:
        print(f"  Skipped (too many columns): {skipped}")

    return datasets


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed

    device = config.device
    seed = config.seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine output paths
    ckpt_path = Path(args.checkpoint)
    if args.output:
        json_path = Path(args.output)
    else:
        json_path = ckpt_path.parent / "eval_report.json"
    pt_path = json_path.with_suffix(".pt").parent / "predictions.pt"

    print(f"Checkpoint: {ckpt_path}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    # Load generator registry
    generators_name = args.generators or config.generator.config_path or config.generator.config_name
    print(f"Generator config: {generators_name}")
    registry = load_registry_from_config(generators_name)
    prior = GeneratorPrior.uniform(registry)
    print(f"  Generators: {registry.K} | Classes: {registry.class_counts()}")

    # Load datasets
    catalog = create_default_catalog()
    dataset_names = args.datasets or config.data.val_datasets or ["iris"]
    print(f"Evaluation datasets: {dataset_names}")

    eval_raw = load_raw_datasets(catalog, dataset_names, config.data.max_cols)
    if len(eval_raw) == 0:
        print("Error: No datasets loaded. Exiting.")
        return

    print(f"  Loaded {len(eval_raw)} datasets")

    # Load Little's cache if provided.
    littles_cache = None
    if args.littles_cache:
        from lacuna.data.littles_cache import load_cache
        littles_cache = load_cache(args.littles_cache)
        print(f"Little's cache: {args.littles_cache} "
              f"({len(littles_cache.entries)} entries)")

    # Create evaluation data loader
    n_batches = args.batches or config.training.val_batches or 20
    eval_loader = SemiSyntheticDataLoader(
        raw_datasets=eval_raw,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=n_batches,
        seed=seed + 9999999,  # Different seed from training
        littles_cache=littles_cache,
    )

    # Create model (with optional mnar_variants override)
    mnar_variants = args.mnar_variants
    model = create_lacuna_model(
        hidden_dim=config.model.hidden_dim,
        evidence_dim=config.model.evidence_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        max_cols=config.data.max_cols,
        dropout=config.model.dropout,
        mnar_variants=mnar_variants,
    )

    # Load checkpoint weights
    load_model_weights(model, str(ckpt_path), device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Create trainer (needed for validate_detailed)
    trainer_config = TrainerConfig(
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
        epochs=1,  # Not training
        warmup_steps=0,
        patience=999,
    )
    trainer = Trainer(model, trainer_config, device=device)

    # Run detailed evaluation
    print(f"\nRunning evaluation ({n_batches} batches)...")
    start_time = time.time()
    result = trainer.validate_detailed(eval_loader)
    eval_time = time.time() - start_time
    print(f"Evaluation complete in {eval_time:.1f}s ({result.n_samples} samples)")

    # Generate report
    report = generate_eval_report(
        result=result,
        registry=registry,
        checkpoint_path=str(ckpt_path),
        config_path=args.config,
    )
    report["eval_time_seconds"] = round(eval_time, 2)
    report["eval_datasets"] = dataset_names
    report["n_eval_batches"] = n_batches

    # Save JSON report
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save raw predictions
    save_raw_predictions(
        p_class=result.all_p_class,
        true_class=result.all_true_class,
        generator_ids=result.all_generator_ids,
        path=pt_path,
    )

    # Print console summary
    print_eval_summary(report)

    # Update run registry
    try:
        from lacuna.experiments.registry import RunRegistry
        from lacuna.experiments.registry_render import write_registry_markdown
        reg_path = PROJECT_ROOT / "experiments" / "registry.json"
        run_registry = RunRegistry(reg_path)
        run_registry.load()
        folder_name = ckpt_path.parent.parent.name
        reg_entry = run_registry.find_by_folder(folder_name)
        if reg_entry is not None:
            new_metrics = {
                "accuracy": report["summary"]["accuracy"],
                "mar_acc": report["summary"].get("mar_acc", 0.0),
                "mnar_acc": report["summary"].get("mnar_acc", 0.0),
            }
            run_registry.update(reg_entry.run_id, status="evaluated", metrics=new_metrics)
            write_registry_markdown(run_registry, reg_path.parent / "REGISTRY.md")
            print(f"Registry updated: {reg_entry.run_id} -> evaluated")
    except Exception as e:
        print(f"Warning: registry update failed: {e}")

    print(f"\nFull report: {json_path}")
    print(f"Raw predictions: {pt_path}")


if __name__ == "__main__":
    main()
