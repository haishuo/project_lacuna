"""Tests for lacuna.training.logging."""

import json
import pytest
from pathlib import Path

from lacuna.training.logging import TrainingLogger, create_logger


class TestTrainingLogger:
    """Tests for TrainingLogger."""

    def test_log_step_writes_jsonl(self, tmp_path):
        """log_step should write a JSON line with type=step."""
        fake_clock = lambda: 1000.0
        logger = TrainingLogger(log_dir=tmp_path, clock=fake_clock)
        logger.log_step({"loss": 0.5, "lr": 0.001})

        lines = logger.log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "step"
        assert entry["timestamp"] == 1000.0
        assert entry["loss"] == 0.5
        assert entry["lr"] == 0.001

    def test_log_epoch_writes_jsonl(self, tmp_path):
        """log_epoch should write a JSON line with type=epoch."""
        fake_clock = lambda: 2000.0
        logger = TrainingLogger(log_dir=tmp_path, clock=fake_clock)
        logger.log_epoch({"epoch": 1, "loss": 0.3})

        lines = logger.log_file.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert entry["type"] == "epoch"
        assert entry["epoch"] == 1

    def test_log_validation_writes_jsonl(self, tmp_path):
        """log_validation should write type=validation."""
        fake_clock = lambda: 3000.0
        logger = TrainingLogger(log_dir=tmp_path, clock=fake_clock)
        logger.log_validation({"val_loss": 0.2, "val_acc": 0.95})

        lines = logger.log_file.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert entry["type"] == "validation"
        assert entry["val_loss"] == 0.2

    def test_multiple_entries_appended(self, tmp_path):
        """Multiple log calls should append to the same file."""
        call_count = 0
        def incrementing_clock():
            nonlocal call_count
            call_count += 1
            return float(call_count)

        logger = TrainingLogger(log_dir=tmp_path, clock=incrementing_clock)
        logger.log_step({"loss": 0.5})
        logger.log_step({"loss": 0.4})
        logger.log_epoch({"epoch": 1})

        lines = logger.log_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_creates_log_dir(self, tmp_path):
        """Should create log_dir if it doesn't exist."""
        log_dir = tmp_path / "deep" / "nested" / "dir"
        logger = TrainingLogger(log_dir=log_dir, clock=lambda: 0.0)
        logger.log_step({"x": 1})
        assert log_dir.exists()
        assert logger.log_file.exists()

    def test_log_file_path(self, tmp_path):
        """log_file should be log_dir/metrics.jsonl."""
        logger = TrainingLogger(log_dir=tmp_path, clock=lambda: 0.0)
        assert logger.log_file == tmp_path / "metrics.jsonl"

    def test_invalid_metrics_type(self, tmp_path):
        """Should raise ValueError if metrics is not a dict."""
        logger = TrainingLogger(log_dir=tmp_path, clock=lambda: 0.0)
        with pytest.raises(ValueError, match="metrics must be a dict"):
            logger.log_step("not a dict")

    def test_none_log_dir(self):
        """Should raise ValueError if log_dir is None."""
        with pytest.raises(ValueError, match="log_dir must not be None"):
            TrainingLogger(log_dir=None)

    def test_as_callback_returns_callable(self, tmp_path):
        """as_callback should return a callable."""
        logger = TrainingLogger(log_dir=tmp_path, clock=lambda: 0.0)
        cb = logger.as_callback()
        assert callable(cb)

    def test_callback_routes_validation(self, tmp_path):
        """Callback should route val_loss metrics to validation type."""
        logger = TrainingLogger(log_dir=tmp_path, clock=lambda: 0.0)
        cb = logger.as_callback()
        cb({"val_loss": 0.1})

        entry = json.loads(logger.log_file.read_text().strip())
        assert entry["type"] == "validation"

    def test_callback_routes_epoch(self, tmp_path):
        """Callback should route epoch-only metrics to epoch type."""
        logger = TrainingLogger(log_dir=tmp_path, clock=lambda: 0.0)
        cb = logger.as_callback()
        cb({"epoch": 1, "loss": 0.5})

        entry = json.loads(logger.log_file.read_text().strip())
        assert entry["type"] == "epoch"

    def test_callback_routes_step(self, tmp_path):
        """Callback should route step-level metrics to step type."""
        logger = TrainingLogger(log_dir=tmp_path, clock=lambda: 0.0)
        cb = logger.as_callback()
        cb({"step": 10, "loss": 0.5})

        entry = json.loads(logger.log_file.read_text().strip())
        assert entry["type"] == "step"


class TestCreateLogger:
    """Tests for backward-compatible create_logger."""

    def test_returns_callable(self, tmp_path):
        """create_logger should return a callable."""
        log_fn = create_logger(tmp_path)
        assert callable(log_fn)

    def test_writes_to_logs_subdir(self, tmp_path):
        """Should write to output_dir/logs/metrics.jsonl."""
        log_fn = create_logger(tmp_path)
        log_fn({"val_loss": 0.5})
        assert (tmp_path / "logs" / "metrics.jsonl").exists()

    def test_none_output_dir(self):
        """Should raise ValueError for None output_dir."""
        with pytest.raises(ValueError, match="output_dir must not be None"):
            create_logger(None)
