"""
lacuna.training.training_step

Training step execution, validation, and evaluation logic.

Contains:
    - DetailedValResult: Rich validation results dataclass
    - TrainingStepExecutor: Executes train steps, validation, and evaluation
"""

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from lacuna.core.types import TokenBatch
from lacuna.core.exceptions import NumericalError
from lacuna.training.loss import (
    LacunaLoss,
    compute_class_accuracy,
)


@dataclass
class DetailedValResult:
    """Rich validation results for evaluation and reporting.

    All tensor fields are on CPU to avoid GPU memory accumulation.
    This is evaluation-only and does NOT affect the training loop.
    """
    metrics: Dict[str, float]           # Same as validate() returns
    confusion_matrix: np.ndarray        # [3, 3] - rows=true, cols=pred
    all_p_class: torch.Tensor           # [N_total, 3] - all probability predictions (CPU)
    all_true_class: torch.Tensor        # [N_total] - all ground truth labels (CPU)
    all_generator_ids: torch.Tensor     # [N_total] - generator that produced each sample (CPU)
    per_generator_acc: Dict[int, float] # generator_id -> accuracy
    per_generator_count: Dict[int, int] # generator_id -> sample count
    n_samples: int


class TrainingStepExecutor:
    """
    Executes training steps, validation, and evaluation.

    Separated from Trainer to keep training loop orchestration
    distinct from step-level computation (CLAUDE.md Rule 3).

    Args:
        model: Neural network model.
        loss_fn: Loss function.
        optimizer: Optimizer.
        use_amp: Whether to use automatic mixed precision.
        grad_clip: Gradient clipping norm (0 = disabled).
        device: Device string for batch transfer.
        scaler: GradScaler for mixed precision (if use_amp=True).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: LacunaLoss,
        optimizer: torch.optim.Optimizer,
        use_amp: bool,
        grad_clip: float,
        device: str,
        scaler: Optional[Any] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.device = device
        self.scaler = scaler

    def _to_device(self, batch: TokenBatch) -> TokenBatch:
        """Move batch to training device."""
        return batch.to(self.device)

    def train_step(self, batch: TokenBatch, scheduler=None, state=None) -> Dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: TokenBatch with training data.
            scheduler: Optional LR scheduler to step after backward pass.
            state: Optional TrainerState to update step count and epoch metrics.

        Returns:
            Dict with loss values and metrics.
        """
        self.model.train()
        batch = self._to_device(batch)

        self.optimizer.zero_grad()

        # Forward pass (with optional mixed precision)
        if self.use_amp:
            with autocast():
                output = self.model(batch)
                total_loss, loss_dict = self.loss_fn(output, batch)
        else:
            output = self.model(batch)
            total_loss, loss_dict = self.loss_fn(output, batch)

        # Check for NaN
        if torch.isnan(total_loss):
            raise NumericalError("NaN loss detected during training")

        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )

            self.optimizer.step()

        # Update learning rate
        current_lr = 0.0
        if scheduler is not None and state is not None:
            current_lr = scheduler.step(state.step)
            state.current_lr = current_lr

        # Compute accuracy
        if batch.class_ids is not None:
            acc = compute_class_accuracy(output.posterior.p_class, batch.class_ids)
        else:
            acc = torch.tensor(0.0)

        # Update state
        if state is not None:
            state.step += 1
            state.samples_seen += batch.batch_size
            state.epoch_loss_sum += total_loss.item() * batch.batch_size
            state.epoch_samples += batch.batch_size
            if batch.class_ids is not None:
                preds = output.posterior.p_class.argmax(dim=-1)
                state.epoch_correct += (preds == batch.class_ids).sum().item()

        # Return metrics
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics["acc"] = acc.item()
        metrics["lr"] = current_lr if state is None else state.current_lr

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation on entire validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Dict with validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        # Per-class tracking
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}

        # Loss component tracking
        loss_sums = {}

        for batch in val_loader:
            batch = self._to_device(batch)

            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(batch)
                    loss, loss_dict = self.loss_fn(output, batch)
            else:
                output = self.model(batch)
                loss, loss_dict = self.loss_fn(output, batch)

            B = batch.batch_size
            total_loss += loss.item() * B
            total_samples += B

            # Track loss components
            for k, v in loss_dict.items():
                val = v.item() if torch.is_tensor(v) else v
                loss_sums[k] = loss_sums.get(k, 0.0) + val * B

            # Accuracy
            if batch.class_ids is not None:
                preds = output.posterior.p_class.argmax(dim=-1)
                total_correct += (preds == batch.class_ids).sum().item()

                # Per-class accuracy
                for class_idx in range(3):
                    mask = batch.class_ids == class_idx
                    if mask.sum() > 0:
                        class_total[class_idx] += mask.sum().item()
                        class_correct[class_idx] += (preds[mask] == class_idx).sum().item()

        # Compute averages
        n = max(total_samples, 1)
        metrics = {
            "val_loss": total_loss / n,
            "val_acc": total_correct / n,
        }

        # Per-class accuracy
        class_names = ["mcar", "mar", "mnar"]
        for class_idx, class_name in enumerate(class_names):
            if class_total[class_idx] > 0:
                metrics[f"val_{class_name}_acc"] = class_correct[class_idx] / class_total[class_idx]
            else:
                metrics[f"val_{class_name}_acc"] = 0.0

        # Loss components
        for k, v in loss_sums.items():
            if k != "total_loss":
                metrics[f"val_{k}"] = v / n

        return metrics

    @torch.no_grad()
    def validate_detailed(self, val_loader: DataLoader) -> DetailedValResult:
        """
        Run detailed validation for evaluation/reporting.

        Unlike validate(), this collects full probability predictions,
        confusion matrix, and per-generator accuracy. All tensors are
        moved to CPU immediately to avoid GPU memory accumulation.

        This method is evaluation-only and does NOT affect training state.

        Args:
            val_loader: DataLoader for validation/test data.

        Returns:
            DetailedValResult with comprehensive metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        # Per-class tracking
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}

        # Loss component tracking
        loss_sums = {}

        # Collect all predictions on CPU
        all_p_class_list = []
        all_true_class_list = []
        all_generator_ids_list = []

        for batch in val_loader:
            batch = self._to_device(batch)

            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(batch)
                    loss, loss_dict = self.loss_fn(output, batch)
            else:
                output = self.model(batch)
                loss, loss_dict = self.loss_fn(output, batch)

            B = batch.batch_size
            total_loss += loss.item() * B
            total_samples += B

            # Track loss components
            for k, v in loss_dict.items():
                val = v.item() if torch.is_tensor(v) else v
                loss_sums[k] = loss_sums.get(k, 0.0) + val * B

            # Collect predictions — move to CPU immediately
            p_class = output.posterior.p_class.detach().cpu()  # [B, 3]
            all_p_class_list.append(p_class)

            if batch.class_ids is not None:
                true_class = batch.class_ids.detach().cpu()  # [B]
                all_true_class_list.append(true_class)

                preds = p_class.argmax(dim=-1)
                total_correct += (preds == true_class).sum().item()

                # Per-class accuracy
                for class_idx in range(3):
                    mask = true_class == class_idx
                    if mask.sum() > 0:
                        class_total[class_idx] += mask.sum().item()
                        class_correct[class_idx] += (preds[mask] == class_idx).sum().item()

            if batch.generator_ids is not None:
                gen_ids = batch.generator_ids.detach().cpu()  # [B]
                all_generator_ids_list.append(gen_ids)

        # Concatenate all collected tensors
        all_p_class = torch.cat(all_p_class_list, dim=0)  # [N_total, 3]
        all_true_class = torch.cat(all_true_class_list, dim=0) if all_true_class_list else torch.zeros(total_samples, dtype=torch.long)
        all_generator_ids = torch.cat(all_generator_ids_list, dim=0) if all_generator_ids_list else torch.zeros(total_samples, dtype=torch.long)

        all_preds = all_p_class.argmax(dim=-1)  # [N_total]

        # Compute confusion matrix [3, 3] — rows=true, cols=pred
        confusion = np.zeros((3, 3), dtype=np.int64)
        for true_idx in range(3):
            for pred_idx in range(3):
                confusion[true_idx, pred_idx] = (
                    (all_true_class == true_idx) & (all_preds == pred_idx)
                ).sum().item()

        # Per-generator accuracy
        per_generator_acc = {}
        per_generator_count = {}
        unique_gens = all_generator_ids.unique().tolist()
        for gen_id in unique_gens:
            gen_mask = all_generator_ids == gen_id
            gen_count = gen_mask.sum().item()
            gen_correct = (all_preds[gen_mask] == all_true_class[gen_mask]).sum().item()
            per_generator_count[gen_id] = gen_count
            per_generator_acc[gen_id] = gen_correct / gen_count if gen_count > 0 else 0.0

        # Compute summary metrics (same format as validate())
        n = max(total_samples, 1)
        metrics = {
            "val_loss": total_loss / n,
            "val_acc": total_correct / n,
        }

        class_names = ["mcar", "mar", "mnar"]
        for class_idx, class_name in enumerate(class_names):
            if class_total[class_idx] > 0:
                metrics[f"val_{class_name}_acc"] = class_correct[class_idx] / class_total[class_idx]
            else:
                metrics[f"val_{class_name}_acc"] = 0.0

        for k, v in loss_sums.items():
            if k != "total_loss":
                metrics[f"val_{k}"] = v / n

        return DetailedValResult(
            metrics=metrics,
            confusion_matrix=confusion,
            all_p_class=all_p_class,
            all_true_class=all_true_class,
            all_generator_ids=all_generator_ids,
            per_generator_acc=per_generator_acc,
            per_generator_count=per_generator_count,
            n_samples=total_samples,
        )

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Dict with test metrics.
        """
        metrics = self.validate(test_loader)
        # Rename val_ to test_
        return {k.replace("val_", "test_"): v for k, v in metrics.items()}
