"""
Tests for lacuna.training.early_stopping

Tests the EarlyStopping class independently.
"""

import pytest

from lacuna.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Tests for the EarlyStopping class."""

    def test_default_state(self):
        """Test initial state values."""
        es = EarlyStopping(patience=5, min_delta=1e-4, early_stop_metric="val_loss", early_stop_mode="min")

        assert es.best_val_loss == float("inf")
        assert es.best_val_acc == 0.0
        assert es.patience_counter == 0
        assert es.should_stop is False
        assert es.best_epoch == 0
        assert es.best_step == 0

    def test_improvement_resets_patience(self):
        """Test that improvement resets patience counter."""
        es = EarlyStopping(patience=5, min_delta=1e-4, early_stop_metric="val_loss", early_stop_mode="min")
        es.best_val_loss = 1.0
        es.patience_counter = 3

        should_stop = es.check({"val_loss": 0.5}, current_epoch=2, current_step=100)

        assert should_stop is False
        assert es.patience_counter == 0
        assert es.best_val_loss == 0.5
        assert es.best_epoch == 2
        assert es.best_step == 100

    def test_no_improvement_increments_patience(self):
        """Test that no improvement increments patience counter."""
        es = EarlyStopping(patience=5, min_delta=0.01, early_stop_metric="val_loss", early_stop_mode="min")
        es.best_val_loss = 0.5
        es.patience_counter = 0

        # 0.51 is not < (0.5 + 0.01) = 0.51, so no improvement
        should_stop = es.check({"val_loss": 0.51}, current_epoch=3, current_step=150)

        assert should_stop is False
        assert es.patience_counter == 1

    def test_patience_exhaustion_triggers_stop(self):
        """Test that patience exhaustion triggers stop."""
        es = EarlyStopping(patience=3, min_delta=1e-4, early_stop_metric="val_loss", early_stop_mode="min")
        es.best_val_loss = 0.5
        es.patience_counter = 2

        should_stop = es.check({"val_loss": 0.6}, current_epoch=5, current_step=200)

        assert should_stop is True
        assert es.patience_counter == 3
        assert es.should_stop is True

    def test_max_mode_improvement(self):
        """Test early stopping in max mode (for accuracy)."""
        es = EarlyStopping(patience=5, min_delta=1e-4, early_stop_metric="val_acc", early_stop_mode="max")
        es.best_val_acc = 0.8
        es.best_val_loss = 0.5
        es.patience_counter = 0

        should_stop = es.check({"val_acc": 0.85, "val_loss": 0.4}, current_epoch=1, current_step=50)

        assert should_stop is False
        assert es.patience_counter == 0
        assert es.best_val_acc == 0.85

    def test_max_mode_no_improvement(self):
        """Test max mode with no improvement."""
        es = EarlyStopping(patience=3, min_delta=1e-4, early_stop_metric="val_acc", early_stop_mode="max")
        es.best_val_acc = 0.9
        es.patience_counter = 1

        should_stop = es.check({"val_acc": 0.89, "val_loss": 0.5}, current_epoch=2, current_step=100)

        assert should_stop is False
        assert es.patience_counter == 2

    def test_is_best_min_mode(self):
        """Test is_best in min mode."""
        es = EarlyStopping(patience=5, min_delta=1e-4, early_stop_metric="val_loss", early_stop_mode="min")
        es.best_val_loss = 0.5

        assert es.is_best({"val_loss": 0.4}) is True
        assert es.is_best({"val_loss": 0.6}) is False

    def test_is_best_max_mode(self):
        """Test is_best in max mode."""
        es = EarlyStopping(patience=5, min_delta=1e-4, early_stop_metric="val_acc", early_stop_mode="max")
        es.best_val_acc = 0.8

        assert es.is_best({"val_acc": 0.85}) is True
        assert es.is_best({"val_acc": 0.75}) is False

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown early_stop_mode"):
            EarlyStopping(patience=5, min_delta=1e-4, early_stop_metric="val_loss", early_stop_mode="invalid")

    def test_tracks_both_metrics_independently(self):
        """Test that both val_loss and val_acc are tracked independently."""
        es = EarlyStopping(patience=5, min_delta=1e-4, early_stop_metric="val_loss", early_stop_mode="min")

        # First check: both metrics improve
        es.check({"val_loss": 0.5, "val_acc": 0.7}, current_epoch=0, current_step=10)
        assert es.best_val_loss == 0.5
        assert es.best_val_acc == 0.7

        # Second check: only acc improves
        es.check({"val_loss": 0.6, "val_acc": 0.8}, current_epoch=1, current_step=20)
        assert es.best_val_loss == 0.5  # unchanged
        assert es.best_val_acc == 0.8   # updated
