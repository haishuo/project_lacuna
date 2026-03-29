"""
lacuna.training.early_stopping

Early stopping logic for training loops.

Monitors a validation metric and stops training when no improvement
is observed for a configurable number of epochs (patience).
"""

from typing import Dict


class EarlyStopping:
    """
    Early stopping tracker.

    Monitors a validation metric and signals when training should stop
    due to lack of improvement.

    Tracks best values for both val_loss and val_acc independently,
    but uses the configured metric for the patience decision.

    Args:
        patience: Number of checks without improvement before stopping.
        min_delta: Minimum change to count as improvement.
        early_stop_metric: Metric key to monitor (e.g. "val_loss", "val_acc").
        early_stop_mode: "min" for loss-like metrics, "max" for accuracy-like.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float,
        early_stop_metric: str,
        early_stop_mode: str,
    ):
        if early_stop_mode not in ("min", "max"):
            raise ValueError(f"Unknown early_stop_mode: {early_stop_mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.early_stop_metric = early_stop_metric
        self.early_stop_mode = early_stop_mode

        # Best metric tracking
        self.best_val_loss: float = float("inf")
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.best_step: int = 0

        # Patience state
        self.patience_counter: int = 0
        self.should_stop: bool = False

    def check(
        self,
        val_metrics: Dict[str, float],
        current_epoch: int,
        current_step: int,
    ) -> bool:
        """
        Check if training should stop early.

        Updates best metrics and patience counter based on validation results.

        Args:
            val_metrics: Validation metrics dict.
            current_epoch: Current training epoch.
            current_step: Current global step.

        Returns:
            True if training should stop, False otherwise.
        """
        val_loss = val_metrics.get("val_loss", float("inf"))
        val_acc = val_metrics.get("val_acc", 0.0)

        # Always track best of both metrics (regardless of which is used for stopping)
        loss_improved = val_loss < (self.best_val_loss - self.min_delta)
        acc_improved = val_acc > (self.best_val_acc + self.min_delta)

        if loss_improved:
            self.best_val_loss = val_loss
        if acc_improved:
            self.best_val_acc = val_acc

        # Determine if the tracked metric improved (for early stopping decision)
        if self.early_stop_mode == "min":
            metric = val_metrics.get(self.early_stop_metric, val_loss)
            improved = metric < (self.best_val_loss + self.min_delta)
        else:  # max
            metric = val_metrics.get(self.early_stop_metric, val_acc)
            improved = metric > (self.best_val_acc - self.min_delta)

        if improved:
            self.patience_counter = 0
            self.best_epoch = current_epoch
            self.best_step = current_step
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                return True
            return False

    def is_best(self, val_metrics: Dict[str, float]) -> bool:
        """Check if current metrics are best so far."""
        if self.early_stop_mode == "min":
            metric = val_metrics.get(self.early_stop_metric, val_metrics["val_loss"])
            return metric < self.best_val_loss
        else:
            metric = val_metrics.get(self.early_stop_metric, val_metrics["val_acc"])
            return metric > self.best_val_acc
