"""
lacuna.training.trainer

Training loop for Lacuna models.

Supports three training modes:
    1. Pretraining: Self-supervised reconstruction only
    2. Classification: Mechanism classification only  
    3. Joint: Both reconstruction and classification

Features:
    - Learning rate warmup and scheduling
    - Gradient clipping
    - Early stopping with patience
    - Checkpointing (best and periodic)
    - Mixed precision training (optional)
    - Multi-task loss balancing
    - Comprehensive logging
    - Validation during training

Design:
    - Explicit configuration via TrainerConfig dataclass
    - Mutable state isolated in TrainerState dataclass
    - Clean separation between training logic and I/O
    - Callbacks for custom logging/monitoring
"""

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Tuple, Any, Union
from pathlib import Path
import time

from lacuna.core.types import TokenBatch
from lacuna.core.exceptions import CheckpointError
from lacuna.models.assembly import LacunaModel
from lacuna.training.loss import (
    LacunaLoss,
    LossConfig,
)
from lacuna.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointData,
)
from lacuna.training.scheduling import LRScheduler
from lacuna.training.early_stopping import EarlyStopping
from lacuna.training.training_step import TrainingStepExecutor, DetailedValResult


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for the training loop."""
    
    # === Optimization ===
    lr: float = 1e-4                    # Peak learning rate
    min_lr: float = 1e-6                # Minimum learning rate for scheduling
    weight_decay: float = 0.01          # AdamW weight decay
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam betas
    eps: float = 1e-8                   # Adam epsilon
    grad_clip: float = 1.0              # Gradient clipping norm (0 = disabled)
    
    # === Schedule ===
    epochs: int = 20                    # Maximum training epochs
    warmup_steps: int = 100             # Linear warmup steps
    warmup_epochs: float = 0.0          # Alternative: warmup as fraction of epoch
    lr_schedule: str = "cosine"         # "constant", "cosine", "linear"
    
    # === Early Stopping ===
    patience: int = 5                   # Epochs without improvement before stopping
    min_delta: float = 1e-4             # Minimum improvement to count as progress
    early_stop_metric: str = "val_loss" # Metric to monitor ("val_loss" or "val_acc")
    early_stop_mode: str = "min"        # "min" for loss, "max" for accuracy
    
    # === Training Mode ===
    training_mode: str = "joint"        # "pretraining", "classification", "joint"
    
    # === Loss Configuration ===
    mechanism_weight: float = 1.0       # Weight for mechanism classification loss
    reconstruction_weight: float = 0.5  # Weight for reconstruction loss
    class_weight: float = 0.5           # Weight for class-level vs mechanism-level
    mechanism_loss_type: str = "cross_entropy"  # "cross_entropy" or "brier"
    label_smoothing: float = 0.0        # Label smoothing (cross-entropy only)
    load_balance_weight: float = 0.01   # MoE load balancing
    
    # === Mixed Precision ===
    use_amp: bool = False               # Use automatic mixed precision
    
    # === Logging ===
    log_every: int = 50                 # Log training metrics every N steps
    eval_every: int = 500               # Evaluate on validation set every N steps
    eval_every_epoch: bool = True       # Also evaluate at end of each epoch
    
    # === Checkpointing ===
    checkpoint_dir: Optional[str] = None  # Directory for checkpoints
    save_best_only: bool = True         # Only save best model
    save_every_epoch: bool = False      # Save checkpoint every epoch
    keep_last_n: int = 3                # Keep last N checkpoints (0 = keep all)
    
    # === Output Control ===
    quiet: bool = False                 # Suppress step-level logging (epoch summaries only)
                                        # Warnings and early stopping notices are NEVER suppressed

    # === Reproducibility ===
    seed: Optional[int] = None          # Random seed
    deterministic: bool = False         # Use deterministic algorithms
    
    def __post_init__(self):
        if self.training_mode not in ("pretraining", "classification", "joint"):
            raise ValueError(f"Unknown training_mode: {self.training_mode}")
        
        if self.lr_schedule not in ("constant", "cosine", "linear"):
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
        
        if self.early_stop_mode not in ("min", "max"):
            raise ValueError(f"Unknown early_stop_mode: {self.early_stop_mode}")
    
    def get_loss_config(self) -> LossConfig:
        """Create LossConfig from trainer config."""
        # Adjust weights based on training mode
        if self.training_mode == "pretraining":
            mechanism_weight = 0.0
            reconstruction_weight = 1.0
        elif self.training_mode == "classification":
            mechanism_weight = 1.0
            reconstruction_weight = 0.0
        else:  # joint
            mechanism_weight = self.mechanism_weight
            reconstruction_weight = self.reconstruction_weight
        
        return LossConfig(
            mechanism_weight=mechanism_weight,
            reconstruction_weight=reconstruction_weight,
            class_weight=self.class_weight,
            mechanism_loss_type=self.mechanism_loss_type,
            label_smoothing=self.label_smoothing,
            load_balance_weight=self.load_balance_weight,
        )


@dataclass
class TrainerState:
    """Mutable training state."""
    
    # === Progress ===
    step: int = 0                       # Global step counter
    epoch: int = 0                      # Current epoch
    samples_seen: int = 0               # Total samples processed

    # === Epoch Metrics ===
    epoch_loss_sum: float = 0.0
    epoch_samples: int = 0
    epoch_correct: int = 0
    epoch_start_time: float = 0.0
    
    # === Learning Rate ===
    current_lr: float = 0.0
    
    def reset_epoch_metrics(self):
        """Reset metrics at start of epoch."""
        self.epoch_loss_sum = 0.0
        self.epoch_samples = 0
        self.epoch_correct = 0
        self.epoch_start_time = time.time()  # NON-DETERMINISTIC: wall-clock timing for diagnostics only

    def get_epoch_metrics(self) -> Dict[str, float]:
        """Compute epoch-level metrics."""
        n = max(self.epoch_samples, 1)
        return {
            "epoch_loss": self.epoch_loss_sum / n,
            "epoch_acc": self.epoch_correct / n,
            "epoch_time": time.time() - self.epoch_start_time,  # NON-DETERMINISTIC: wall-clock timing
        }


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Training loop manager for Lacuna models.
    
    Handles:
        - Training loop with validation
        - Learning rate scheduling
        - Gradient clipping
        - Early stopping
        - Checkpointing
        - Logging via callbacks
    
    Attributes:
        model: LacunaModel to train.
        config: TrainerConfig with training parameters.
        device: Device for training (cuda/cpu).
        state: TrainerState with current training progress.
        loss_fn: LacunaLoss for computing losses.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision (if enabled).
    
    Example:
        >>> trainer = Trainer(model, config, device="cuda")
        >>> trainer.fit(train_loader, val_loader)
        >>> # Access best metrics
        >>> print(f"Best val acc: {trainer.state.best_val_acc}")
    """
    
    def __init__(
        self,
        model: LacunaModel,
        config: TrainerConfig,
        device: str = "cuda",
        log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: LacunaModel to train.
            config: TrainerConfig with training parameters.
            device: Device string ("cuda", "cpu", "cuda:0", etc.)
            log_fn: Optional callback for logging metrics.
        """
        self.config = config
        self.device = device
        self.log_fn = log_fn
        
        # Set random seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        
        # Set deterministic mode if requested
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Move model to device
        self.model = model.to(device)
        
        # Initialize state
        self.state = TrainerState()

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            early_stop_metric=config.early_stop_metric,
            early_stop_mode=config.early_stop_mode,
        )

        # Create loss function
        self.loss_fn = LacunaLoss(config.get_loss_config())
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
        
        # Scheduler will be initialized in fit() when we know total steps
        self.scheduler: Optional[LRScheduler] = None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Checkpoint directory
        if config.checkpoint_dir is not None:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        # Track saved checkpoints for cleanup
        self.saved_checkpoints: List[Path] = []

        # Initialize step executor for training/validation
        self.step_executor = TrainingStepExecutor(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            use_amp=config.use_amp,
            grad_clip=config.grad_clip,
            device=device,
            scaler=self.scaler,
        )
    
    def _log(self, metrics: Dict[str, Any]):
        """Log metrics via callback if set."""
        if self.log_fn is not None:
            self.log_fn(metrics)
    
    def train_step(self, batch: TokenBatch) -> Dict[str, float]:
        """Execute single training step (delegates to TrainingStepExecutor)."""
        return self.step_executor.train_step(
            batch, scheduler=self.scheduler, state=self.state,
        )

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation on entire validation set."""
        return self.step_executor.validate(val_loader)

    def validate_detailed(self, val_loader: DataLoader) -> DetailedValResult:
        """Run detailed validation for evaluation/reporting."""
        return self.step_executor.validate_detailed(val_loader)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        return self.step_executor.evaluate(test_loader)

    def _save_checkpoint(
        self,
        val_metrics: Dict[str, float],
        is_best: bool = False,
        force: bool = False,
    ):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        # Determine if we should save
        if self.config.save_best_only and not is_best and not force:
            return
        
        # Create checkpoint data
        checkpoint = CheckpointData(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=None,  # We track LR in TrainerState
            step=self.state.step,
            epoch=self.state.epoch,
            best_val_loss=self.early_stopping.best_val_loss,
            best_val_acc=self.early_stopping.best_val_acc,
            config=self.config.__dict__,
            metrics=val_metrics,
        )
        
        # Save
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch{self.state.epoch}_step{self.state.step}.pt"
        
        save_checkpoint(checkpoint, path)
        
        # Track and cleanup old checkpoints
        if not is_best:
            self.saved_checkpoints.append(path)
            
            if self.config.keep_last_n > 0:
                while len(self.saved_checkpoints) > self.config.keep_last_n:
                    old_path = self.saved_checkpoints.pop(0)
                    if old_path.exists():
                        old_path.unlink()
    
    def load_checkpoint(self, path: Union[str, Path]):
        """
        Load model and training state from checkpoint.
        
        Args:
            path: Path to checkpoint file.
        """
        checkpoint = load_checkpoint(path, device=self.device)
        
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        
        self.state.step = checkpoint.step
        self.state.epoch = checkpoint.epoch
        self.early_stopping.best_val_loss = checkpoint.best_val_loss
        self.early_stopping.best_val_acc = checkpoint.best_val_acc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Run full training loop.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            resume_from: Optional path to checkpoint to resume from.
        
        Returns:
            Dict with final/best metrics.
        """
        # Resume from checkpoint if specified
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            print(f"Resumed from checkpoint: step={self.state.step}, epoch={self.state.epoch}")
        
        # Compute total steps for scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config.epochs
        
        # Initialize scheduler
        self.scheduler = LRScheduler(
            optimizer=self.optimizer,
            lr=self.config.lr,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
            warmup_epochs=self.config.warmup_epochs,
            lr_schedule=self.config.lr_schedule,
            total_steps=total_steps,
        )
        self.scheduler.update_warmup_steps(steps_per_epoch)
        
        # Initialize learning rate
        self.state.current_lr = self.scheduler.get_lr(self.state.step)
        
        quiet = self.config.quiet

        if not quiet:
            print(f"Starting training:")
            print(f"  Mode: {self.config.training_mode}")
            print(f"  Epochs: {self.config.epochs}")
            print(f"  Steps/epoch: {steps_per_epoch}")
            print(f"  Total steps: {total_steps}")
            print(f"  Device: {self.device}")
            print(f"  Mixed precision: {self.config.use_amp}")

        # Training loop
        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.epochs):
            self.state.epoch = epoch
            self.state.reset_epoch_metrics()

            # Epoch training loop
            for batch_idx, batch in enumerate(train_loader):
                # Training step
                step_metrics = self.train_step(batch)

                # Periodic logging (always log to callback, only print if not quiet)
                if self.state.step % self.config.log_every == 0:
                    step_metrics["epoch"] = epoch
                    step_metrics["step"] = self.state.step
                    self._log(step_metrics)

                    if not quiet:
                        loss_str = f"loss={step_metrics['total_loss']:.4f}"
                        acc_str = f"acc={step_metrics['acc']:.3f}"
                        lr_str = f"lr={step_metrics['lr']:.2e}"
                        print(f"  Step {self.state.step}: {loss_str}, {acc_str}, {lr_str}")

                # Periodic validation
                if (
                    val_loader is not None
                    and self.config.eval_every > 0
                    and self.state.step % self.config.eval_every == 0
                ):
                    val_metrics = self.validate(val_loader)
                    val_metrics["step"] = self.state.step
                    self._log(val_metrics)

                    if not quiet:
                        print(f"  Validation: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_acc']:.3f}")

                    # Check for improvement and save checkpoint
                    is_best = self.early_stopping.is_best(val_metrics)
                    if is_best:
                        self._save_checkpoint(val_metrics, is_best=True)

                    # Early stopping check — ALWAYS printed
                    if self.early_stopping.check(val_metrics, self.state.epoch, self.state.step):
                        print(f"Early stopping at step {self.state.step}")
                        break

            # End of epoch — always build metrics, print in appropriate format
            epoch_metrics = self.state.get_epoch_metrics()

            # End-of-epoch validation
            val_metrics_epoch = None
            is_best = False
            if val_loader is not None and self.config.eval_every_epoch:
                val_metrics_epoch = self.validate(val_loader)
                val_metrics_epoch["epoch"] = epoch
                self._log(val_metrics_epoch)

                # Check for improvement
                is_best = self.early_stopping.is_best(val_metrics_epoch)
                if is_best:
                    self._save_checkpoint(val_metrics_epoch, is_best=True)

                # Early stopping — ALWAYS printed
                if self.early_stopping.check(val_metrics_epoch, self.state.epoch, self.state.step):
                    print(f"Early stopping at epoch {epoch}")

            # Print epoch summary
            if quiet:
                # Compact single-line format
                parts = [f"Epoch {epoch+1:3d}/{self.config.epochs}"]
                parts.append(f"train_acc={epoch_metrics['epoch_acc']:.3f}")
                if val_metrics_epoch is not None:
                    parts.append(f"val_acc={val_metrics_epoch['val_acc']:.3f}")
                    parts.append(f"val_loss={val_metrics_epoch['val_loss']:.4f}")
                parts.append(f"{epoch_metrics['epoch_time']:.1f}s")
                line = " | ".join(parts)
                if val_metrics_epoch is not None and is_best:
                    line += "  \u2605 best"
                print(line)
            else:
                print(f"Epoch {epoch} complete: loss={epoch_metrics['epoch_loss']:.4f}, "
                      f"acc={epoch_metrics['epoch_acc']:.3f}, time={epoch_metrics['epoch_time']:.1f}s")
                if val_metrics_epoch is not None:
                    print(f"  Epoch validation: loss={val_metrics_epoch['val_loss']:.4f}, acc={val_metrics_epoch['val_acc']:.3f}")
                    if is_best:
                        print(f"  New best model! (val_acc={val_metrics_epoch['val_acc']:.4f})")

            # Periodic checkpoint
            if self.config.save_every_epoch:
                self._save_checkpoint(val_metrics_epoch if val_loader else {}, force=True)

            # Check if should stop
            if self.early_stopping.should_stop:
                break

        # Final summary — ALWAYS printed
        print(f"\nTraining complete!")
        print(f"  Best val_loss: {self.early_stopping.best_val_loss:.4f} (epoch {self.early_stopping.best_epoch})")
        print(f"  Best val_acc: {self.early_stopping.best_val_acc:.4f}")

        return {
            "best_val_loss": self.early_stopping.best_val_loss,
            "best_val_acc": self.early_stopping.best_val_acc,
            "best_epoch": self.early_stopping.best_epoch,
            "final_epoch": self.state.epoch,
            "total_steps": self.state.step,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def train_lacuna(
    model: LacunaModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str = "cuda",
    training_mode: str = "joint",
    checkpoint_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Convenience function to train a Lacuna model.
    
    Args:
        model: LacunaModel to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device string.
        training_mode: "pretraining", "classification", or "joint".
        checkpoint_dir: Optional checkpoint directory.
        **kwargs: Additional TrainerConfig parameters.
    
    Returns:
        trainer: Trained Trainer instance.
        results: Dict with training results.
    
    Example:
        >>> trainer, results = train_lacuna(model, train_loader, val_loader)
        >>> print(f"Best accuracy: {results['best_val_acc']}")
    """
    config = TrainerConfig(
        epochs=epochs,
        lr=lr,
        training_mode=training_mode,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )
    
    trainer = Trainer(model, config, device=device)
    results = trainer.fit(train_loader, val_loader)
    
    return trainer, results


def pretrain_lacuna(
    model: LacunaModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
    **kwargs,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Pretrain Lacuna model with reconstruction objective only.
    
    Args:
        model: LacunaModel to pretrain.
        train_loader: DataLoader for training data (with artificial masking).
        val_loader: Optional DataLoader for validation.
        epochs: Number of pretraining epochs.
        lr: Learning rate.
        device: Device string.
        **kwargs: Additional TrainerConfig parameters.
    
    Returns:
        trainer: Trainer instance.
        results: Dict with pretraining results.
    """
    return train_lacuna(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        training_mode="pretraining",
        **kwargs,
    )


def finetune_lacuna(
    model: LacunaModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 20,
    lr: float = 1e-5,  # Lower LR for finetuning
    device: str = "cuda",
    **kwargs,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Finetune pretrained Lacuna model for classification.
    
    Args:
        model: Pretrained LacunaModel.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation.
        epochs: Number of finetuning epochs.
        lr: Learning rate (typically lower than pretraining).
        device: Device string.
        **kwargs: Additional TrainerConfig parameters.
    
    Returns:
        trainer: Trainer instance.
        results: Dict with finetuning results.
    """
    return train_lacuna(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        training_mode="classification",
        **kwargs,
    )