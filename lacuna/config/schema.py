"""
lacuna.config.schema

Configuration dataclasses with validation.

Simplified for row-level tokenization architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    """Data processing configuration.

    Lacuna trains on semi-synthetic data: real tabular X loaded from the
    catalog (via `train_datasets` / `val_datasets`), with synthetic
    missingness mechanisms applied per batch.
    """
    # Dimension limits (tokeniser capacity)
    max_cols: int = 32
    max_rows: int = 256

    # Catalog dataset names for semi-synthetic training.
    # train_datasets is required; val_datasets is required whenever
    # training runs. Left Optional so partial configs (e.g. eval-only
    # invocations) can omit one.
    train_datasets: Optional[List[str]] = None
    val_datasets: Optional[List[str]] = None

    def __post_init__(self):
        if self.max_cols <= 0:
            raise ValueError("max_cols must be positive")
        if self.max_rows <= 0:
            raise ValueError("max_rows must be positive")


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 128
    evidence_dim: int = 64
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 32
    batches_per_epoch: int = 100
    val_batches: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 200
    patience: int = 10
    min_delta: float = 1e-4
    
    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class GeneratorConfig:
    """Generator configuration.

    config_name: Named config under configs/generators/ (e.g. "lacuna_minimal_6")
    config_path: Absolute path to a custom YAML file (overrides config_name)
    """
    config_name: str = "lacuna_minimal_6"
    config_path: Optional[str] = None


@dataclass
class LacunaConfig:
    """Complete Lacuna configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "/mnt/artifacts/project_lacuna/runs"
    loss_matrix: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.3, 1.0],
        [0.2, 0.0, 0.2],
        [1.0, 0.3, 0.0],
    ])
    
    def get_loss_matrix_tensor(self) -> torch.Tensor:
        """Convert loss matrix to tensor."""
        return torch.tensor(self.loss_matrix, dtype=torch.float32)
    
    @classmethod
    def minimal(cls) -> "LacunaConfig":
        """Create minimal semi-synthetic config for fast testing.

        Uses iris (the smallest sklearn built-in, always available) as
        both train and val source. Paired with `lacuna_minimal_6` for a
        small generator registry. CPU-only. Intended for test suites and
        smoke checks — NOT for dissertation-grade ablation runs.
        """
        return cls(
            data=DataConfig(
                max_cols=16,
                max_rows=64,
                train_datasets=["iris"],
                val_datasets=["iris"],
            ),
            model=ModelConfig(
                hidden_dim=64,
                evidence_dim=32,
                n_layers=4,
                n_heads=4,
                dropout=0.1,
            ),
            training=TrainingConfig(
                epochs=5,
                batch_size=8,
                batches_per_epoch=20,
                val_batches=5,
                lr=1e-3,
                warmup_steps=10,
                patience=3,
            ),
            generator=GeneratorConfig(config_name="lacuna_minimal_6"),
            device="cpu",
        )