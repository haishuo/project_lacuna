"""
lacuna.training.scheduling

Learning rate scheduling with warmup and decay.

Supports:
    - Linear warmup
    - Cosine annealing
    - Linear decay
    - Constant (after warmup)
"""

import math
import torch


class LRScheduler:
    """
    Learning rate scheduler with warmup and decay.

    Supports:
        - Linear warmup
        - Cosine annealing
        - Linear decay
        - Constant (after warmup)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        min_lr: float,
        warmup_steps: int,
        warmup_epochs: float,
        lr_schedule: str,
        total_steps: int,
    ):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: Optimizer whose LR will be updated.
            lr: Peak (base) learning rate.
            min_lr: Minimum learning rate for scheduling.
            warmup_steps: Number of linear warmup steps.
            warmup_epochs: Alternative warmup as fraction of epoch
                (overrides warmup_steps when > 0 and update_warmup_steps is called).
            lr_schedule: Schedule type — "constant", "cosine", or "linear".
            total_steps: Total training steps for computing decay progress.
        """
        if lr_schedule not in ("constant", "cosine", "linear"):
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

        self.optimizer = optimizer
        self.total_steps = total_steps

        self.base_lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.lr_schedule = lr_schedule

        # If warmup_epochs is set, compute warmup_steps
        # (will be updated when we know steps_per_epoch)
        self.warmup_epochs = warmup_epochs

    def update_warmup_steps(self, steps_per_epoch: int):
        """Update warmup steps based on epochs."""
        if self.warmup_epochs > 0:
            self.warmup_steps = int(self.warmup_epochs * steps_per_epoch)

    def get_lr(self, step: int) -> float:
        """Compute learning rate for given step."""
        # Warmup phase
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        # Post-warmup phase
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)

        if self.lr_schedule == "constant":
            return self.base_lr

        elif self.lr_schedule == "cosine":
            # Cosine annealing to min_lr
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        elif self.lr_schedule == "linear":
            # Linear decay to min_lr
            return self.base_lr - (self.base_lr - self.min_lr) * progress

        return self.base_lr

    def step(self, current_step: int):
        """Update optimizer learning rate."""
        lr = self.get_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
