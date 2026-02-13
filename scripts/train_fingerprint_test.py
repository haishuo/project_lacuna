#!/usr/bin/env python3
"""
Generator Fingerprint Test: Train on threshold, test on sigmoid.

This tests whether Lacuna learns abstract MNAR mechanisms or just
memorizes implementation details (functional form fingerprints).
"""

import argparse
from logging import config
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config.schema import LacunaConfig, DataConfig, ModelConfig, TrainingConfig
from lacuna.generators import (
    GeneratorRegistry,
    GeneratorParams,
    GeneratorPrior,
    MCARBernoulli,
    MARLogistic,
    MNARThresholdLeft,
)
from lacuna.data.batching import SyntheticDataLoader, SyntheticDataLoaderConfig
from lacuna.models.assembly import create_lacuna_model
from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.training import save_checkpoint, CheckpointData


def create_threshold_registry():
    """Registry with THRESHOLD MNAR only (no sigmoid!)."""
    return GeneratorRegistry([
        MCARBernoulli(0, "mcar_low", GeneratorParams(miss_rate=0.15)),
        MCARBernoulli(1, "mcar_high", GeneratorParams(miss_rate=0.35)),
        MARLogistic(2, "mar_weak", GeneratorParams(alpha0=-0.5, alpha1=2.0)),
        MARLogistic(3, "mar_strong", GeneratorParams(alpha0=-0.5, alpha1=4.0)),
        MNARThresholdLeft(4, "mnar_thresh_70", GeneratorParams(percentile=70, miss_prob=0.6)),
        MNARThresholdLeft(5, "mnar_thresh_80", GeneratorParams(percentile=80, miss_prob=0.7)),
    ])


def main():
    print("=" * 70)
    print("GENERATOR FINGERPRINT TEST")
    print("=" * 70)
    print("Training on THRESHOLD MNAR only (no sigmoid variants)")
    print("Will test on sigmoid MNAR after training\n")
    
    # Configuration
    config = LacunaConfig(
        data=DataConfig(max_cols=16, n_range=(50, 200), d_range=(5, 12)),
        model=ModelConfig(hidden_dim=128, n_layers=4, n_heads=4),
        training=TrainingConfig(
            batch_size=64,
            lr=1e-4,
            epochs=25,  # Short training for quick test
        ),
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.epochs}")
    
    # Create threshold-only registry
    registry = create_threshold_registry()
    prior = GeneratorPrior.uniform(registry)
    
    print(f"\nRegistry (THRESHOLD MNAR ONLY):")
    for gen in registry.generators:
        print(f"  {gen.generator_id}: {gen.name} (class={gen.class_id})")
    
    # Create model
    model = create_lacuna_model(
        hidden_dim=config.model.hidden_dim,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        max_cols=config.data.max_cols,
        mnar_variants=["threshold"],  # Tell model we only have threshold variant
    ).to(config.device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    loader_config = SyntheticDataLoaderConfig(
        batch_size=16,
        n_range=config.data.n_range,
        d_range=config.data.d_range,
        max_cols=config.data.max_cols,
        seed=config.seed,
        batches_per_epoch=100,
    )

    train_loader = SyntheticDataLoader(
        generators=registry.generators,
        config=loader_config,
    )

    val_loader = SyntheticDataLoader(
        generators=registry.generators,
        config=loader_config,
    )
    
    # Train
    trainer_config = TrainerConfig(
        epochs=config.training.epochs,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        eval_every_epoch=True,
    )
    
    trainer = Trainer(model, trainer_config, device=config.device)
    
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    result = trainer.fit(train_loader, val_loader)
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val acc: {result['best_val_acc']*100:.1f}%")
    print(f"Training time: {training_time/60:.1f} minutes")
    
    # Save model
    output_dir = Path("outputs/fingerprint_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = CheckpointData(
        model_state=model.state_dict(),
        step=result['total_steps'],
        epoch=result['final_epoch'],
        best_val_loss=result['best_val_loss'],
        best_val_acc=result['best_val_acc'],
        metrics={"training_time": training_time, "test_type": "threshold_only"},
    )
    save_checkpoint(ckpt, output_dir / "model_threshold_trained.pt")
    
    print(f"\n✓ Model saved to: {output_dir / 'model_threshold_trained.pt'}")
    print("\nNEXT: Run evaluation script to test on sigmoid MNAR!")


if __name__ == "__main__":
    main()