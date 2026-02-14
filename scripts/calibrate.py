#!/usr/bin/env python3
"""
Post-hoc temperature scaling for Lacuna model calibration.

Loads a trained checkpoint, collects gate logits on validation data,
finds the optimal temperature via grid search, and saves a calibrated
checkpoint with the patched temperature.

The calibrated checkpoint can be evaluated with evaluate.py to see
before/after ECE improvement.

Usage:
    python scripts/calibrate.py --checkpoint CKPT --config CONFIG
    python scripts/calibrate.py --checkpoint CKPT --config CONFIG --device cuda
    python scripts/calibrate.py --checkpoint CKPT --config CONFIG --output calibrated.pt
    python scripts/calibrate.py --checkpoint CKPT --config CONFIG --generators lacuna_tabular_110
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
    load_model_weights,
    save_checkpoint,
    CheckpointData,
)
from lacuna.training.calibration import (
    collect_gate_logits,
    find_optimal_temperature,
    apply_temperature_scaling,
    logits_to_class_probs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc temperature scaling for Lacuna calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pt file)",
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
        help="Output checkpoint path (default: {checkpoint_dir}/calibrated.pt)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Override which datasets to use for calibration",
    )
    parser.add_argument(
        "--batches", type=int, default=None,
        help="Number of calibration batches (default: from config val_batches)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--t-min", type=float, default=0.1,
        help="Minimum temperature to search (default: 0.1)",
    )
    parser.add_argument(
        "--t-max", type=float, default=10.0,
        help="Maximum temperature to search (default: 10.0)",
    )
    parser.add_argument(
        "--mnar-variants", type=str, nargs="+", default=None,
        help="MNAR variant names (default: ['self_censoring']). "
             "Must match the architecture used to train the checkpoint.",
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

    # Determine output path
    ckpt_path = Path(args.checkpoint)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ckpt_path.parent / "calibrated.pt"

    print("=" * 60)
    print("LACUNA TEMPERATURE SCALING")
    print("=" * 60)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    # Load generator registry
    generators_name = args.generators or config.generator.config_path or config.generator.config_name
    registry = load_registry_from_config(generators_name)
    prior = GeneratorPrior.uniform(registry)
    print(f"\nGenerators: {registry.K} | Classes: {registry.class_counts()}")

    # Load calibration datasets (use validation datasets)
    catalog = create_default_catalog()
    dataset_names = args.datasets or config.data.val_datasets or ["iris"]
    print(f"Calibration datasets: {dataset_names}")

    cal_raw = load_raw_datasets(catalog, dataset_names, config.data.max_cols)
    if len(cal_raw) == 0:
        print("Error: No datasets loaded. Exiting.")
        return

    print(f"  Loaded {len(cal_raw)} datasets")

    # Create calibration data loader
    n_batches = args.batches or config.training.val_batches or 30
    cal_loader = SemiSyntheticDataLoader(
        raw_datasets=cal_raw,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=n_batches,
        seed=seed + 7777777,  # Different seed from training/eval
    )

    # Create model and load weights
    mnar_variants = args.mnar_variants  # None means use default (1/1/1)
    model = create_lacuna_model(
        hidden_dim=config.model.hidden_dim,
        evidence_dim=config.model.evidence_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        max_cols=config.data.max_cols,
        dropout=config.model.dropout,
        mnar_variants=mnar_variants,
    )
    load_model_weights(model, str(ckpt_path), device=device)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    current_temp = model.moe.gating.temperature.item()
    print(f"Current temperature: {current_temp:.4f}")

    # Collect gate logits
    print(f"\nCollecting gate logits ({n_batches} batches)...")
    start_time = time.time()
    gate_logits, true_class = collect_gate_logits(model, cal_loader, device=device)
    collect_time = time.time() - start_time
    print(f"  Collected {gate_logits.shape[0]} samples in {collect_time:.1f}s")
    print(f"  Gate logits shape: {gate_logits.shape}")

    # Get expert-to-class mapping from model
    expert_to_class = model.moe.expert_to_class.cpu()
    experts_per_class = model.moe.experts_per_class.cpu()
    print(f"  Expert-to-class: {expert_to_class.tolist()}")
    print(f"  Experts-per-class: {experts_per_class.tolist()}")

    # Find optimal temperature
    print(f"\nSearching for optimal temperature (range [{args.t_min}, {args.t_max}])...")
    opt_start = time.time()
    optimal_t, info = find_optimal_temperature(
        gate_logits=gate_logits,
        true_class=true_class,
        expert_to_class=expert_to_class,
        experts_per_class=experts_per_class,
        t_min=args.t_min,
        t_max=args.t_max,
    )
    opt_time = time.time() - opt_start
    print(f"  Search completed in {opt_time:.1f}s")

    # Report results
    print(f"\n{'=' * 60}")
    print("CALIBRATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Optimal temperature: {optimal_t:.4f}")
    print(f"  NLL:  {info['nll_before']:.4f} -> {info['nll_after']:.4f}  (Δ = {info['nll_after'] - info['nll_before']:.4f})")
    print(f"  ECE:  {info['ece_before']:.4f} -> {info['ece_after']:.4f}  (Δ = {info['ece_after'] - info['ece_before']:.4f})")

    # Compute accuracy before/after (temperature doesn't change argmax much,
    # but can shift borderline cases)
    p_before = logits_to_class_probs(gate_logits, 1.0, expert_to_class, experts_per_class)
    p_after = logits_to_class_probs(gate_logits, optimal_t, expert_to_class, experts_per_class)

    acc_before = (p_before.argmax(dim=-1) == true_class).float().mean().item()
    acc_after = (p_after.argmax(dim=-1) == true_class).float().mean().item()
    print(f"  Acc:  {acc_before*100:.1f}% -> {acc_after*100:.1f}%")

    # Per-class accuracy
    class_names = ["MCAR", "MAR", "MNAR"]
    for c_idx, c_name in enumerate(class_names):
        mask = true_class == c_idx
        if mask.sum() > 0:
            acc_b = (p_before.argmax(dim=-1)[mask] == c_idx).float().mean().item()
            acc_a = (p_after.argmax(dim=-1)[mask] == c_idx).float().mean().item()
            print(f"    {c_name}: {acc_b*100:.1f}% -> {acc_a*100:.1f}%  (n={mask.sum().item()})")

    # Apply temperature and save
    print(f"\nApplying temperature T={optimal_t:.4f} to model...")
    apply_temperature_scaling(model, optimal_t)

    # Verify
    new_temp = model.moe.gating.temperature.item()
    print(f"  Verified: model temperature = {new_temp:.4f}")

    # Save calibrated checkpoint
    calibrated_ckpt = CheckpointData(
        model_state=model.state_dict(),
        metrics={
            "calibration": {
                "optimal_temperature": round(optimal_t, 6),
                "nll_before": info["nll_before"],
                "nll_after": info["nll_after"],
                "ece_before": info["ece_before"],
                "ece_after": info["ece_after"],
                "accuracy_before": round(acc_before, 4),
                "accuracy_after": round(acc_after, 4),
                "n_calibration_samples": info["n_samples"],
                "source_checkpoint": str(ckpt_path),
            },
        },
    )
    save_checkpoint(calibrated_ckpt, output_path)
    print(f"\nCalibrated checkpoint saved: {output_path}")

    # Also save calibration info as JSON
    info_path = output_path.with_suffix(".json")
    cal_info = {
        "optimal_temperature": round(optimal_t, 6),
        "source_checkpoint": str(ckpt_path),
        "config": args.config,
        "calibration_datasets": dataset_names,
        "n_calibration_batches": n_batches,
        "n_calibration_samples": info["n_samples"],
        "nll_before": info["nll_before"],
        "nll_after": info["nll_after"],
        "ece_before": info["ece_before"],
        "ece_after": info["ece_after"],
        "accuracy_before": round(acc_before, 4),
        "accuracy_after": round(acc_after, 4),
    }
    with open(info_path, "w") as f:
        json.dump(cal_info, f, indent=2)
    print(f"Calibration info saved: {info_path}")

    print(f"\nNext step: evaluate calibrated model with:")
    print(f"  python scripts/evaluate.py --checkpoint {output_path} --config {args.config}")


if __name__ == "__main__":
    main()
