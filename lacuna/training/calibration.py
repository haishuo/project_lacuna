"""
lacuna.training.calibration

Post-hoc temperature scaling for calibration (Guo et al. 2017).

Temperature scaling finds an optimal temperature T that minimizes NLL on a
held-out validation set, then patches the model's gating network buffer.

The Lacuna model has a MoE layer with 5 experts mapped to 3 classes:
    expert_to_class = [MCAR, MAR, MNAR, MNAR, MNAR]
    experts_per_class = [1, 1, 3]

Gate logits are converted to class probabilities via:
    1. gate_probs = softmax(logits / T)
    2. p_class[c] = sum(gate_probs[i] for i where expert_to_class[i] == c)
    3. p_class[c] /= experts_per_class[c]   (mean aggregation)
    4. p_class /= p_class.sum()              (renormalize)

Temperature scaling only adjusts T in step 1 and does not modify any learned
weights. This is the simplest and most effective post-hoc calibration method.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lacuna.models.assembly import LacunaModel
from lacuna.core.types import MoEOutput


def logits_to_class_probs(
    gate_logits: torch.Tensor,
    temperature: float,
    expert_to_class: torch.Tensor,
    experts_per_class: torch.Tensor,
) -> torch.Tensor:
    """
    Convert gate logits to class probabilities at a given temperature.

    Replicates the aggregation logic of MixtureOfExperts.get_class_posterior()
    with "mean" class_aggregation, but allows an arbitrary temperature.

    Args:
        gate_logits: [N, n_experts] pre-softmax gating logits.
        temperature: Scalar temperature for softmax.
        expert_to_class: [n_experts] mapping from expert index to class index.
        experts_per_class: [n_classes] count of experts per class.

    Returns:
        p_class: [N, n_classes] class probabilities.
    """
    # Step 1: temperature-scaled softmax
    gate_probs = F.softmax(gate_logits / temperature, dim=-1)  # [N, 5]

    N = gate_logits.shape[0]
    n_classes = experts_per_class.shape[0]

    # Step 2: aggregate to class level
    p_class = torch.zeros(N, n_classes, device=gate_logits.device)
    for expert_idx in range(gate_logits.shape[1]):
        class_idx = expert_to_class[expert_idx].item()
        p_class[:, class_idx] += gate_probs[:, expert_idx]

    # Step 3: normalize by experts per class (mean aggregation)
    p_class = p_class / experts_per_class.unsqueeze(0).clamp(min=1)

    # Step 4: renormalize
    p_class = p_class / p_class.sum(dim=-1, keepdim=True)

    return p_class


def nll_loss(
    gate_logits: torch.Tensor,
    true_class: torch.Tensor,
    temperature: float,
    expert_to_class: torch.Tensor,
    experts_per_class: torch.Tensor,
) -> float:
    """
    Compute negative log-likelihood at a given temperature.

    Args:
        gate_logits: [N, n_experts] pre-softmax gating logits.
        true_class: [N] true class labels.
        temperature: Scalar temperature.
        expert_to_class: [n_experts] expert-to-class mapping.
        experts_per_class: [n_classes] experts per class.

    Returns:
        Scalar NLL value.
    """
    p_class = logits_to_class_probs(
        gate_logits, temperature, expert_to_class, experts_per_class
    )
    log_probs = torch.log(p_class.clamp(min=1e-8))
    nll = F.nll_loss(log_probs, true_class)
    return nll.item()


def find_optimal_temperature(
    gate_logits: torch.Tensor,
    true_class: torch.Tensor,
    expert_to_class: torch.Tensor,
    experts_per_class: torch.Tensor,
    t_min: float = 0.1,
    t_max: float = 10.0,
    n_coarse: int = 100,
    n_fine: int = 100,
) -> Tuple[float, dict]:
    """
    Find optimal temperature via grid search (coarse + fine).

    Two-phase approach:
        1. Coarse: Log-uniform grid over [t_min, t_max]
        2. Fine: Linear grid around the coarse optimum

    Args:
        gate_logits: [N, n_experts] collected gate logits (CPU).
        true_class: [N] true class labels (CPU).
        expert_to_class: [n_experts] expert-to-class mapping (CPU).
        experts_per_class: [n_classes] experts per class (CPU).
        t_min: Minimum temperature to search.
        t_max: Maximum temperature to search.
        n_coarse: Number of coarse grid points.
        n_fine: Number of fine grid points.

    Returns:
        (optimal_temperature, info_dict) where info_dict contains:
            nll_before: NLL at T=1.0
            nll_after: NLL at optimal T
            ece_before: ECE at T=1.0
            ece_after: ECE at optimal T
            search_range: (t_min, t_max)
    """
    # Coarse search: log-uniform grid
    log_temps = torch.linspace(math.log(t_min), math.log(t_max), n_coarse)
    coarse_temps = torch.exp(log_temps)

    best_nll = float("inf")
    best_t = 1.0

    for t in coarse_temps.tolist():
        current_nll = nll_loss(
            gate_logits, true_class, t, expert_to_class, experts_per_class
        )
        if current_nll < best_nll:
            best_nll = current_nll
            best_t = t

    # Fine search: linear grid around coarse optimum
    fine_min = max(t_min, best_t * 0.5)
    fine_max = min(t_max, best_t * 2.0)
    fine_temps = torch.linspace(fine_min, fine_max, n_fine)

    for t in fine_temps.tolist():
        current_nll = nll_loss(
            gate_logits, true_class, t, expert_to_class, experts_per_class
        )
        if current_nll < best_nll:
            best_nll = current_nll
            best_t = t

    # Compute before/after metrics
    nll_before = nll_loss(
        gate_logits, true_class, 1.0, expert_to_class, experts_per_class
    )
    nll_after = best_nll

    ece_before = _compute_ece(
        gate_logits, true_class, 1.0, expert_to_class, experts_per_class
    )
    ece_after = _compute_ece(
        gate_logits, true_class, best_t, expert_to_class, experts_per_class
    )

    info = {
        "nll_before": round(nll_before, 6),
        "nll_after": round(nll_after, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "search_range": (t_min, t_max),
        "n_samples": gate_logits.shape[0],
    }

    return best_t, info


def _compute_ece(
    gate_logits: torch.Tensor,
    true_class: torch.Tensor,
    temperature: float,
    expert_to_class: torch.Tensor,
    experts_per_class: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error at a given temperature."""
    p_class = logits_to_class_probs(
        gate_logits, temperature, expert_to_class, experts_per_class
    )
    confidences, preds = p_class.max(dim=-1)
    correct = (preds == true_class).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(true_class)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        mask = (confidences > lo) & (confidences <= hi)
        n_bin = mask.sum().item()
        if n_bin > 0:
            avg_conf = confidences[mask].mean().item()
            avg_acc = correct[mask].mean().item()
            ece += (n_bin / n_total) * abs(avg_acc - avg_conf)

    return ece


def apply_temperature_scaling(model: LacunaModel, temperature: float) -> None:
    """
    Patch the model's gating network with the optimal temperature.

    Modifies model.moe.gating.log_temperature in-place.

    Args:
        model: LacunaModel instance.
        temperature: Optimal temperature from find_optimal_temperature().
    """
    log_t = math.log(temperature)
    device = model.moe.gating.log_temperature.device
    model.moe.gating.log_temperature.data = torch.tensor(log_t, device=device)


@torch.no_grad()
def collect_gate_logits(
    model: LacunaModel,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect gate logits and true class labels from a data loader.

    Runs forward passes and collects the raw gate logits (pre-softmax)
    from the MoE layer. All tensors are moved to CPU immediately.

    Args:
        model: LacunaModel instance.
        data_loader: DataLoader yielding TokenBatch instances.
        device: Device to run forward passes on.

    Returns:
        (gate_logits, true_class) both on CPU.
            gate_logits: [N_total, n_experts]
            true_class: [N_total]
    """
    model.eval()

    logits_list = []
    labels_list = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch)

        # Collect gate logits from MoE output
        logits_list.append(output.moe.gate_logits.detach().cpu())

        if batch.class_ids is not None:
            labels_list.append(batch.class_ids.detach().cpu())

    gate_logits = torch.cat(logits_list, dim=0)
    true_class = torch.cat(labels_list, dim=0) if labels_list else torch.zeros(
        gate_logits.shape[0], dtype=torch.long
    )

    return gate_logits, true_class
