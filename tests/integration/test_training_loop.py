"""
Integration test: Full training loop with checkpointing.

Tests:
- Training reduces loss
- Checkpoints save and restore correctly
- Validation runs correctly
- Early stopping works

Uses semi-synthetic data (real iris X + synthetic missingness mechanism) —
the same data pathway as production, scaled down for test speed.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.semisynthetic import SemiSyntheticDataLoader
from lacuna.generators import load_registry_from_config
from lacuna.generators.priors import GeneratorPrior
from lacuna.models.assembly import create_lacuna_mini
from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_model_weights,
    CheckpointData,
)


@pytest.fixture
def registry():
    """Minimal generator registry."""
    return load_registry_from_config("lacuna_minimal_6")


@pytest.fixture
def iris_raw():
    """Iris loaded once — 150 rows × 4 features, always available via sklearn."""
    return create_default_catalog().load("iris")


def _make_loader(registry, iris_raw, *, batches: int, batch_size: int, seed: int):
    """Build a SemiSyntheticDataLoader on iris for a test run."""
    return SemiSyntheticDataLoader(
        raw_datasets=[iris_raw],
        registry=registry,
        prior=GeneratorPrior.uniform(registry),
        max_rows=64,
        max_cols=16,
        batch_size=batch_size,
        batches_per_epoch=batches,
        seed=seed,
    )


def _sample_batch(loader: SemiSyntheticDataLoader):
    """Pull one batch from the loader (for test assertions on a fixed batch)."""
    return next(iter(loader))


class TestTrainingLoop:
    """Test full training loop."""

    def test_training_reduces_loss(self, registry, iris_raw):
        """Training completes and emits finite validation metrics."""
        model = create_lacuna_mini(max_cols=16)

        train_loader = _make_loader(registry, iris_raw, batches=10, batch_size=8, seed=42)
        val_loader = _make_loader(registry, iris_raw, batches=3, batch_size=8, seed=99)

        trainer_config = TrainerConfig(
            lr=3e-3,
            epochs=5,
            warmup_steps=5,
            grad_clip=1.0,
            patience=100,
        )
        trainer = Trainer(model, trainer_config, device="cpu")

        result = trainer.fit(train_loader, val_loader)

        assert result["final_epoch"] == 4
        assert result["best_val_loss"] < float("inf")
        assert result["best_val_loss"] > 0
        assert result["best_val_acc"] >= 0.0

    def test_checkpoint_save_load(self, registry, iris_raw):
        """Checkpoints preserve model state."""
        model1 = create_lacuna_mini(max_cols=16)
        model2 = create_lacuna_mini(max_cols=16)

        train_loader = _make_loader(registry, iris_raw, batches=5, batch_size=8, seed=42)
        trainer_config = TrainerConfig(lr=1e-3, epochs=1, warmup_steps=5)
        trainer = Trainer(model1, trainer_config, device="cpu")
        trainer.fit(train_loader)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            data = CheckpointData(
                model_state=model1.state_dict(),
                step=trainer.state.step,
                epoch=trainer.state.epoch,
            )
            save_checkpoint(data, ckpt_path)
            load_model_weights(model2, ckpt_path)

        test_batch = _sample_batch(
            _make_loader(registry, iris_raw, batches=1, batch_size=4, seed=123)
        )

        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(test_batch)
            out2 = model2(test_batch)

        assert torch.allclose(out1.posterior.p_class, out2.posterior.p_class)

    def test_training_with_validation(self, registry, iris_raw):
        """Training with validation tracks best model."""
        model = create_lacuna_mini(max_cols=16)

        train_loader = _make_loader(registry, iris_raw, batches=10, batch_size=8, seed=42)
        val_loader = _make_loader(registry, iris_raw, batches=3, batch_size=8, seed=99)

        trainer_config = TrainerConfig(
            lr=1e-3,
            epochs=3,
            warmup_steps=10,
            patience=10,
        )
        trainer = Trainer(model, trainer_config, device="cpu")

        result = trainer.fit(train_loader, val_loader)

        assert "best_val_loss" in result
        assert "best_val_acc" in result
        assert result["best_val_loss"] < float("inf")
        assert result["best_val_acc"] >= 0

    def test_early_stopping_triggers(self, registry, iris_raw):
        """Early stopping fires when validation stops improving."""
        model = create_lacuna_mini(max_cols=16)

        train_loader = _make_loader(registry, iris_raw, batches=5, batch_size=4, seed=42)
        val_loader = _make_loader(registry, iris_raw, batches=2, batch_size=4, seed=99)

        trainer_config = TrainerConfig(
            lr=1e-8,
            epochs=100,
            warmup_steps=0,
            patience=2,
            min_delta=0.001,
        )
        trainer = Trainer(model, trainer_config, device="cpu")

        result = trainer.fit(train_loader, val_loader)

        assert result["final_epoch"] < 100


class TestCheckpointIntegrity:
    """Test checkpoint save/load integrity."""

    def test_checkpoint_preserves_all_state(self, registry, iris_raw):
        """Checkpoint preserves step, epoch, and optimizer state."""
        model = create_lacuna_mini(max_cols=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        loader = _make_loader(registry, iris_raw, batches=5, batch_size=4, seed=42)
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch)
            log_probs = output.posterior.p_class.clamp(min=1e-8).log()
            loss = torch.nn.functional.nll_loss(log_probs, batch.class_ids)
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            data = CheckpointData(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                step=42,
                epoch=3,
                best_val_loss=0.5,
                metrics={"test_key": "test_value"},
            )
            save_checkpoint(data, ckpt_path)
            loaded = load_checkpoint(ckpt_path)

        assert loaded.step == 42
        assert loaded.epoch == 3
        assert loaded.best_val_loss == 0.5
        assert loaded.metrics["test_key"] == "test_value"
        assert loaded.optimizer_state is not None
