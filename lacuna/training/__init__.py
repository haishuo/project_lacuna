"""
lacuna.training

Training infrastructure for Project Lacuna.

Exports:
- Loss functions and configuration
- Trainer and configuration
- Checkpointing
- Logging
"""

from .loss import (
    # Configuration
    LossConfig,
    # Mechanism classification losses
    mechanism_cross_entropy,
    mechanism_cross_entropy_from_probs,
    brier_score,
    class_cross_entropy,
    mechanism_full_cross_entropy,
    # Reconstruction losses
    reconstruction_mse,
    reconstruction_huber,
    multi_head_reconstruction_loss,
    # Auxiliary losses
    kl_divergence_loss,
    entropy_loss,
    load_balance_loss,
    # Combined loss
    LacunaLoss,
    # Factory functions
    create_loss_function,
    create_pretraining_loss,
    create_classification_loss,
    create_joint_loss,
    # Accuracy metrics
    compute_class_accuracy,
    compute_mechanism_accuracy,
    compute_per_class_accuracy,
)

from .trainer import (
    Trainer,
    TrainerConfig,
    TrainerState,
    DetailedValResult,
)

from .checkpoint import (
    CheckpointData,
    save_checkpoint,
    load_checkpoint,
    load_model_weights,
    CheckpointManager,
    validate_checkpoint,
    compare_checkpoints,
    export_for_inference,
    compute_checkpoint_hash,
)

from .logging import (
    create_logger,
)

from .report import (
    generate_eval_report,
    compute_confidence_analysis,
    compute_probability_distributions,
    compute_entropy_stats,
    compute_ece,
    compute_per_generator_accuracy,
    save_raw_predictions,
    load_raw_predictions,
    print_eval_summary,
)

__all__ = [
    # Loss Configuration
    "LossConfig",
    # Mechanism Losses
    "mechanism_cross_entropy",
    "mechanism_cross_entropy_from_probs",
    "brier_score",
    "class_cross_entropy",
    "mechanism_full_cross_entropy",
    # Reconstruction Losses
    "reconstruction_mse",
    "reconstruction_huber",
    "multi_head_reconstruction_loss",
    # Auxiliary Losses
    "kl_divergence_loss",
    "entropy_loss",
    "load_balance_loss",
    # Combined Loss
    "LacunaLoss",
    # Factory Functions
    "create_loss_function",
    "create_pretraining_loss",
    "create_classification_loss",
    "create_joint_loss",
    # Accuracy Metrics
    "compute_class_accuracy",
    "compute_mechanism_accuracy",
    "compute_per_class_accuracy",
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "DetailedValResult",
    # Checkpoint
    "CheckpointData",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_weights",
    "CheckpointManager",
    "validate_checkpoint",
    "compare_checkpoints",
    "export_for_inference",
    "compute_checkpoint_hash",
    # Logging
    "create_logger",
    # Report utilities
    "generate_eval_report",
    "compute_confidence_analysis",
    "compute_probability_distributions",
    "compute_entropy_stats",
    "compute_ece",
    "compute_per_generator_accuracy",
    "save_raw_predictions",
    "load_raw_predictions",
    "print_eval_summary",
]