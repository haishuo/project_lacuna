#!/usr/bin/env python3
"""
Generator Fingerprint Test: Train on one MNAR style, test on another.

This tests whether Lacuna learns abstract MNAR mechanisms or just
memorizes implementation details (functional form fingerprints).

Default: Train on threshold-style MNAR (fingerprint_train), evaluate
on sigmoid-style MNAR (fingerprint_test) — an OOD generalization test.

Usage:
    python scripts/train_fingerprint_test.py
    python scripts/train_fingerprint_test.py --train-config fingerprint_train --test-config fingerprint_test
    python scripts/train_fingerprint_test.py --device cuda --epochs 50
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config.schema import LacunaConfig, DataConfig, ModelConfig, TrainingConfig
from lacuna.generators import load_registry_from_config
from lacuna.generators.priors import GeneratorPrior
from lacuna.data.batching import SyntheticDataLoader, SyntheticDataLoaderConfig
from lacuna.models.assembly import create_lacuna_model
from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.training import save_checkpoint, CheckpointData, create_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fingerprint test: train on one MNAR style, evaluate on another"
    )
    parser.add_argument("--train-config", type=str, default="fingerprint_train",
                        help="Generator config name for training (default: fingerprint_train)")
    parser.add_argument("--test-config", type=str, default="fingerprint_test",
                        help="Generator config name for OOD evaluation (default: fingerprint_test)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--batches-per-epoch", type=int, default=100,
                        help="Batches per epoch")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden dim")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max-cols", type=int, default=16, help="Max columns")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GENERATOR FINGERPRINT TEST")
    print("=" * 70)
    print(f"Train config:  {args.train_config}")
    print(f"Test config:   {args.test_config}")
    print(f"Device:        {device}")
    print(f"Epochs:        {args.epochs}")
    print(f"Seed:          {args.seed}")

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"fingerprint_{timestamp}"
    output_dir = Path("outputs") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # ================================================================
    # Load registries from YAML configs
    # ================================================================
    print(f"\nLoading train registry: {args.train_config}")
    train_registry = load_registry_from_config(args.train_config)
    print(f"  Generators: {train_registry.K}")
    print(f"  Class distribution: {train_registry.class_counts()}")

    print(f"\nLoading test registry: {args.test_config}")
    test_registry = load_registry_from_config(args.test_config)
    print(f"  Generators: {test_registry.K}")
    print(f"  Class distribution: {test_registry.class_counts()}")

    # Print train generators
    print(f"\nTrain generators ({train_registry.K}):")
    for gen in train_registry:
        print(f"  {gen.generator_id:3d}: {gen.name} (class={gen.class_id})")

    # Print test generators
    print(f"\nTest generators ({test_registry.K}):")
    for gen in test_registry:
        print(f"  {gen.generator_id:3d}: {gen.name} (class={gen.class_id})")

    # ================================================================
    # Create data loaders
    # ================================================================
    train_loader_config = SyntheticDataLoaderConfig(
        batch_size=args.batch_size,
        n_range=(50, 200),
        d_range=(5, 12),
        max_cols=args.max_cols,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )

    val_loader_config = SyntheticDataLoaderConfig(
        batch_size=args.batch_size,
        n_range=(50, 200),
        d_range=(5, 12),
        max_cols=args.max_cols,
        batches_per_epoch=args.batches_per_epoch // 5,
        seed=args.seed + 1000000,
    )

    train_loader = SyntheticDataLoader(
        generators=train_registry.generators,
        config=train_loader_config,
    )

    # Validation uses TRAIN registry (in-distribution)
    val_loader = SyntheticDataLoader(
        generators=train_registry.generators,
        config=val_loader_config,
    )

    # Test uses TEST registry (out-of-distribution)
    test_loader_config = SyntheticDataLoaderConfig(
        batch_size=args.batch_size,
        n_range=(50, 200),
        d_range=(5, 12),
        max_cols=args.max_cols,
        batches_per_epoch=args.batches_per_epoch // 2,
        seed=args.seed + 2000000,
    )

    test_loader = SyntheticDataLoader(
        generators=test_registry.generators,
        config=test_loader_config,
    )

    print(f"\n  Train loader: {len(train_loader)} batches/epoch")
    print(f"  Val loader:   {len(val_loader)} batches (in-distribution)")
    print(f"  Test loader:  {len(test_loader)} batches (OOD)")

    # ================================================================
    # Create model
    # ================================================================
    print("\nCreating model...")
    model = create_lacuna_model(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_cols=args.max_cols,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ================================================================
    # Train
    # ================================================================
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-5,
        warmup_steps=100,
        patience=10,
        checkpoint_dir=str(output_dir / "checkpoints"),
        save_best_only=True,
        eval_every_epoch=True,
    )

    logger = create_logger(output_dir)
    trainer = Trainer(model, trainer_config, device=device, log_fn=logger)

    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING (in-distribution)")
    print("=" * 70)

    start_time = time.time()
    result = trainer.fit(train_loader, val_loader)
    training_time = time.time() - start_time

    print("\n" + "-" * 70)
    print("TRAINING RESULTS (in-distribution)")
    print("-" * 70)
    print(f"  Epochs completed: {result['final_epoch'] + 1}")
    print(f"  Total steps:      {result['total_steps']}")
    print(f"  Best val loss:    {result['best_val_loss']:.4f}")
    print(f"  Best val acc:     {result['best_val_acc']*100:.1f}%")
    print(f"  Training time:    {training_time:.1f}s ({training_time/60:.1f}m)")

    # Save checkpoint
    ckpt = CheckpointData(
        model_state=model.state_dict(),
        step=result['total_steps'],
        epoch=result['final_epoch'],
        best_val_loss=result['best_val_loss'],
        best_val_acc=result['best_val_acc'],
        metrics={
            "training_time": training_time,
            "train_config": args.train_config,
            "test_config": args.test_config,
            "phase": "training",
        },
    )
    save_checkpoint(ckpt, output_dir / "checkpoints" / "model_trained.pt")

    # ================================================================
    # Evaluate on OOD test set
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: OOD EVALUATION")
    print(f"  Train MNAR style: {args.train_config}")
    print(f"  Test MNAR style:  {args.test_config}")
    print("=" * 70)

    ood_metrics = trainer.evaluate(test_loader)

    print("\n" + "-" * 70)
    print("OOD TEST RESULTS")
    print("-" * 70)
    for key, val in sorted(ood_metrics.items()):
        if "acc" in key:
            print(f"  {key}: {val*100:.1f}%")
        else:
            print(f"  {key}: {val:.4f}")

    # ================================================================
    # Compare in-distribution vs OOD
    # ================================================================
    print("\n" + "=" * 70)
    print("FINGERPRINT TEST SUMMARY")
    print("=" * 70)

    id_acc = result['best_val_acc']
    ood_acc = ood_metrics.get('test_accuracy', ood_metrics.get('test_acc', 0.0))
    gap = id_acc - ood_acc

    print(f"  In-distribution accuracy:  {id_acc*100:.1f}%")
    print(f"  OOD accuracy:              {ood_acc*100:.1f}%")
    print(f"  Generalization gap:        {gap*100:.1f}pp")
    print()

    if gap < 0.05:
        print("  PASS: Model generalizes well across MNAR functional forms!")
        print("  The model appears to learn abstract mechanism properties,")
        print("  not implementation-specific fingerprints.")
    elif gap < 0.15:
        print("  MARGINAL: Moderate generalization gap.")
        print("  Some fingerprint memorization may be occurring.")
    else:
        print("  FAIL: Large generalization gap — fingerprint memorization detected.")
        print("  The model likely memorizes functional forms rather than learning")
        print("  abstract mechanism properties.")

    # Save final summary
    summary = {
        "train_config": args.train_config,
        "test_config": args.test_config,
        "in_distribution_accuracy": float(id_acc),
        "ood_accuracy": float(ood_acc),
        "generalization_gap": float(gap),
        "training_time_seconds": training_time,
        "epochs_completed": result['final_epoch'] + 1,
        "ood_metrics": {k: float(v) for k, v in ood_metrics.items()},
    }

    import json
    with open(output_dir / "fingerprint_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {output_dir / 'fingerprint_results.json'}")
    print(f"  Model saved to:   {output_dir / 'checkpoints' / 'model_trained.pt'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
