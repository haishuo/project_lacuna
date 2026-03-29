"""
lacuna.training.logging

Structured training logger with JSON Lines output.

Provides a TrainingLogger class that writes structured metrics
to a JSONL file, with injectable clock for deterministic testing.
"""

import json
import time
from pathlib import Path
from typing import Callable


class TrainingLogger:
    """Structured training logger writing JSON Lines format.

    Each log entry is a JSON object on its own line in
    ``{log_dir}/metrics.jsonl``, with a ``type`` field
    (``"step"``, ``"epoch"``, or ``"validation"``) and a
    ``timestamp`` field from the injected clock.

    Args:
        log_dir: Directory for log output. Created if it does not exist.
        clock: Callable returning current time as float.
            # NON-DETERMINISTIC: timestamps depend on system clock
            Defaults to ``time.time``.

    Raises:
        ValueError: If log_dir is None.
    """

    def __init__(
        self,
        log_dir: Path,
        clock: Callable[[], float] = time.time,  # NON-DETERMINISTIC: timestamps depend on system clock
    ) -> None:
        if log_dir is None:
            raise ValueError("log_dir must not be None")
        self._log_dir = Path(log_dir)
        self._clock = clock
        self._log_file = self._log_dir / "metrics.jsonl"

    @property
    def log_file(self) -> Path:
        """Path to the JSONL metrics file."""
        return self._log_file

    def _write_entry(self, entry_type: str, metrics: dict) -> None:
        """Write a single JSON Lines entry.

        Args:
            entry_type: One of "step", "epoch", "validation".
            metrics: Arbitrary metrics dict. Must be JSON-serializable.

        Raises:
            ValueError: If metrics is not a dict.
            TypeError: If metrics contains non-JSON-serializable values.
        """
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics must be a dict, got {type(metrics).__name__}")

        entry = {
            "type": entry_type,
            "timestamp": self._clock(),
            **metrics,
        }
        self._log_dir.mkdir(parents=True, exist_ok=True)
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_step(self, metrics: dict) -> None:
        """Log a training step.

        Args:
            metrics: Step-level metrics (e.g. loss, learning rate).
        """
        self._write_entry("step", metrics)

    def log_epoch(self, metrics: dict) -> None:
        """Log an epoch summary.

        Args:
            metrics: Epoch-level metrics (e.g. avg loss, accuracy).
        """
        self._write_entry("epoch", metrics)

    def log_validation(self, metrics: dict) -> None:
        """Log validation results.

        Args:
            metrics: Validation metrics (e.g. val_loss, val_acc).
        """
        self._write_entry("validation", metrics)

    def as_callback(self) -> Callable[[dict], None]:
        """Return a logging callback compatible with Trainer.log_fn.

        The callback inspects the metrics dict to determine the entry type:
        - Contains ``"val_loss"`` -> validation
        - Contains ``"epoch"`` but not ``"step"`` -> epoch
        - Otherwise -> step

        Returns:
            Callable that accepts a metrics dict and logs it.
        """
        def callback(metrics: dict) -> None:
            if "val_loss" in metrics:
                self.log_validation(metrics)
            elif "epoch" in metrics and "step" not in metrics:
                self.log_epoch(metrics)
            else:
                self.log_step(metrics)
        return callback


def create_logger(output_dir: Path) -> Callable[[dict], None]:
    """Create logging function for training metrics.

    Backward-compatible wrapper that creates a TrainingLogger internally.

    Args:
        output_dir: Experiment output directory. Logs will be written
            to ``{output_dir}/logs/metrics.jsonl``.

    Returns:
        Logging callback function that accepts a metrics dict.

    Raises:
        ValueError: If output_dir is None.
    """
    if output_dir is None:
        raise ValueError("output_dir must not be None")
    logger = TrainingLogger(log_dir=Path(output_dir) / "logs")
    return logger.as_callback()
