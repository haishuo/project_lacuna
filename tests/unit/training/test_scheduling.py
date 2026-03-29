"""
Tests for lacuna.training.scheduling

Tests the learning rate scheduler with warmup and decay.
"""

import pytest
import torch
import torch.nn as nn
import math

from lacuna.training.scheduling import LRScheduler


# =============================================================================
# Test LRScheduler
# =============================================================================

class TestLRScheduler:
    """Tests for learning rate scheduler."""

    @pytest.fixture
    def optimizer(self):
        """Create dummy optimizer."""
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_warmup_phase(self, optimizer):
        """Test that LR increases during warmup."""
        scheduler = LRScheduler(
            optimizer=optimizer,
            lr=1e-3,
            min_lr=1e-6,
            warmup_steps=100,
            warmup_epochs=0.0,
            lr_schedule="cosine",
            total_steps=1000,
        )

        # LR should increase during warmup
        lr_at_0 = scheduler.get_lr(0)
        lr_at_50 = scheduler.get_lr(50)
        lr_at_100 = scheduler.get_lr(100)

        assert lr_at_0 < lr_at_50 <= lr_at_100
        assert abs(lr_at_100 - 1e-3) < 1e-6

    def test_cosine_decay(self, optimizer):
        """Test cosine learning rate decay."""
        scheduler = LRScheduler(
            optimizer=optimizer,
            lr=1e-3,
            min_lr=1e-5,
            warmup_steps=100,
            warmup_epochs=0.0,
            lr_schedule="cosine",
            total_steps=1000,
        )

        # LR should decay after warmup
        lr_at_warmup_end = scheduler.get_lr(100)
        lr_at_middle = scheduler.get_lr(500)
        lr_at_end = scheduler.get_lr(999)

        assert lr_at_warmup_end >= lr_at_middle >= lr_at_end
        # At end should be close to min_lr
        assert lr_at_end == pytest.approx(1e-5, rel=0.1)

    def test_constant_schedule(self, optimizer):
        """Test constant learning rate (after warmup)."""
        scheduler = LRScheduler(
            optimizer=optimizer,
            lr=1e-3,
            min_lr=1e-6,
            warmup_steps=100,
            warmup_epochs=0.0,
            lr_schedule="constant",
            total_steps=1000,
        )

        # After warmup, should stay at lr
        lr_at_500 = scheduler.get_lr(500)
        lr_at_900 = scheduler.get_lr(900)

        assert lr_at_500 == pytest.approx(1e-3, rel=1e-3)
        assert lr_at_900 == pytest.approx(1e-3, rel=1e-3)

    def test_linear_decay(self, optimizer):
        """Test linear learning rate decay."""
        scheduler = LRScheduler(
            optimizer=optimizer,
            lr=1e-3,
            min_lr=1e-5,
            warmup_steps=0,
            warmup_epochs=0.0,
            lr_schedule="linear",
            total_steps=1000,
        )

        lr_start = scheduler.get_lr(0)
        lr_mid = scheduler.get_lr(500)
        lr_end = scheduler.get_lr(999)

        assert lr_start > lr_mid > lr_end
        assert lr_start == pytest.approx(1e-3, rel=1e-3)
        assert lr_end == pytest.approx(1e-5, rel=0.1)

    def test_step_updates_optimizer(self, optimizer):
        """Test that step() updates optimizer LR."""
        scheduler = LRScheduler(
            optimizer=optimizer,
            lr=1e-3,
            min_lr=1e-6,
            warmup_steps=100,
            warmup_epochs=0.0,
            lr_schedule="cosine",
            total_steps=1000,
        )

        # Step should update optimizer's LR
        scheduler.step(50)

        actual_lr = optimizer.param_groups[0]["lr"]
        expected_lr = scheduler.get_lr(50)

        assert actual_lr == pytest.approx(expected_lr, rel=1e-5)

    def test_update_warmup_steps_from_epochs(self, optimizer):
        """Test warmup_epochs overrides warmup_steps."""
        scheduler = LRScheduler(
            optimizer=optimizer,
            lr=1e-3,
            min_lr=1e-6,
            warmup_steps=100,
            warmup_epochs=0.5,
            lr_schedule="cosine",
            total_steps=1000,
        )

        scheduler.update_warmup_steps(steps_per_epoch=200)
        assert scheduler.warmup_steps == 100  # 0.5 * 200

    def test_invalid_schedule_raises(self, optimizer):
        """Test that invalid schedule type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            LRScheduler(
                optimizer=optimizer,
                lr=1e-3,
                min_lr=1e-6,
                warmup_steps=100,
                warmup_epochs=0.0,
                lr_schedule="invalid",
                total_steps=1000,
            )
