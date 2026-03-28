# Lacuna Architecture

## What Lacuna Is

Lacuna is a transformer-based classifier for missing data mechanisms in tabular datasets.
Given data with missing values, it determines whether missingness is MCAR, MAR, or MNAR,
then recommends a statistical action (complete-case analysis, multiple imputation, or
sensitivity analysis) via a Bayes-optimal decision rule.

## System Overview

```
Data Input (CSV with NaN)
         |
         v
+---------------------------------------------+
| DATA PIPELINE                               |
|  ingestion.py     -> RawDataset             |
|  semisynthetic.py -> apply missingness      |
|  tokenization.py  -> [val, obs, mask, fid]  |
|  batching.py      -> TokenBatch [B,R,C,4]   |
+---------------------------------------------+
         |
         v  TokenBatch
+---------------------------------------------+
| ENCODER (encoder.py)                        |
|  TokenEmbedding: 4 -> hidden_dim            |
|  Transformer: row-wise self-attention       |
|  RowPooling: cols -> hidden_dim             |
|  DatasetPooling: rows -> evidence_dim       |
|  Output: evidence [B, 64]                   |
+---------------------------------------------+
         |
         v  evidence
+---------------------------------------------+
| PARALLEL FEATURE EXTRACTION                 |
|  reconstruction/  -> predict masked values  |
|    3 heads (default): MCAR, MAR,            |
|      MNAR-SelfCensoring                     |
|    Output: per-head errors [B, 3]           |
|  missingness_features.py -> 16 statistics   |
|    Output: features [B, 16]                 |
+---------------------------------------------+
         |
         v  evidence + errors + features
+---------------------------------------------+
| MIXTURE OF EXPERTS (moe.py)                 |
|  GatingNetwork -> [B, 3] gate weights       |
|  3 experts: MCAR, MAR, MNAR                |
|  Aggregation -> class_logits [B, 3]         |
+---------------------------------------------+
         |
         v  class_logits
+---------------------------------------------+
| POSTERIOR & DECISION                        |
|  softmax -> p_class [B, 3]                  |
|  Bayes decision + loss matrix               |
|    -> action (Green/Yellow/Red)             |
|    -> expected risk                         |
+---------------------------------------------+
         |
         v
LacunaOutput (posterior, decision, moe, evidence)
```

## Module Map

### `lacuna/core/` - Foundation types and utilities
- `types.py` - Immutable dataclasses: ObservedDataset, TokenBatch, PosteriorResult, Decision, LacunaOutput
- `exceptions.py` - Exception hierarchy: LacunaError -> ValidationError, ConfigError, RegistryError, CheckpointError, NumericalError
- `validation.py` - Boundary validation: shape, dtype, NaN/Inf, probability, range
- `rng.py` - Deterministic RNG management via explicit seed passing (RNGState)
- `logging.py` - Stub (placeholder)

### `lacuna/config/` - Configuration system
- `schema.py` - Dataclass configs: DataConfig, ModelConfig, TrainingConfig, GeneratorConfig, LacunaConfig
- `load.py` - YAML loading/serialization with validation on load
- `hashing.py` - Deterministic config hashing for reproducibility

### `lacuna/data/` - Data pipeline
- `ingestion.py` - Load CSV/Parquet/sklearn datasets -> RawDataset
- `catalog.py` - Dataset catalog with caching (UCI/OpenML/sklearn)
- `normalization.py` - Robust MAD-based normalization
- `observed.py` - ObservedDataset utilities
- `semisynthetic.py` - Apply synthetic missingness to real data
- `tokenization.py` - Row-level tokenization: cell -> [value, observed, mask_type, feature_id]
- `batching.py` - SyntheticDataLoader, collation with padding
- `missingness_features.py` - Extract 16 statistical features for mechanism discrimination

### `lacuna/models/` - Neural architecture
- `encoder.py` - LacunaEncoder: BERT-inspired transformer with row/dataset pooling
- `assembly.py` - LacunaModel: orchestrates encoder + reconstruction + features + MoE + decision
- `moe.py` - MixtureOfExperts: gating network + 3 experts + aggregation
- `decision.py` - Bayes-optimal decision rule with loss matrix
- `aggregator.py` - ClassAggregator: combine expert logits -> class posterior
- `heads.py` - Expert head definitions
- `reconstruction/` - Self-supervised pretraining heads (MCAR, MAR, MNAR-SelfCensoring).
    Note: `MNARThresholdHead` and `MNARLatentHead` exist in code but are unused —
    scheduled for removal (see REFACTORING_CHECKLIST.md).

### `lacuna/generators/` - 110 parametric missingness generators
- `base.py` - Generator abstract base class (immutable)
- `registry.py` - GeneratorRegistry with validation
- `params.py` - GeneratorParams immutable container
- `priors.py` - GeneratorPrior for sampling distributions
- `families/` - Implementations organized by mechanism class
  - `base_data.py` - Synthetic data samplers
  - `registry_builder.py` - Load/build registries from YAML
  - `mcar/` - 32 MCAR generators (bernoulli, multilevel, conditional, blocks, etc.)
  - `mar/` - 36 MAR generators (logistic, structural, complex, multiple, survey, etc.)
  - `mnar/` - 42 MNAR generators (self-censoring, latent, detection, selection, etc.)

### `lacuna/training/` - Training infrastructure
- `trainer.py` - Main training loop: scheduling, clipping, early stopping, checkpointing
- `loss.py` - Multi-task loss: mechanism CE + reconstruction MSE + auxiliary
- `checkpoint.py` - Checkpoint save/load with validation and comparison
- `calibration.py` - Post-hoc temperature scaling
- `report.py` - Evaluation reporting and analysis
- `logging.py` - Structured logging utility

### `scripts/` - Standalone entry points
- `train_semisynthetic.py` - Full training pipeline
- `infer.py` - End-user inference interface
- `evaluate.py` - Model evaluation
- `calibrate.py` - Post-hoc calibration
- `consolidate_results.py` - Gather experiment results
- `generate_dissertation_figures.py` - Publication figures
- `generate_roc_curves.py` - ROC/PR curves
- `diagnose_reconstruction.py` - Reconstruction diagnostics
- `download_datasets.py` - Dataset download/cache
- `train.py` - Simple training entry point

### `tests/` - Test suite (~13,800 lines)
- `unit/` - Per-module unit tests covering normal, edge, and failure cases
- `integration/` - End-to-end pipeline, training loop, determinism, registry loading

## Key Design Decisions

1. **Explicit RNG** - No global seeds. All randomness flows through RNGState with spawn() for child streams.
2. **Immutable data structures** - ObservedDataset, TokenBatch, GeneratorParams, generators are all immutable.
3. **110 generators** - Large parametric generator set ensures training diversity across mechanism subtypes.
4. **Semi-synthetic pipeline** - Real data structure + synthetic missingness = realistic training data.
5. **Symmetric experts** - 1 MCAR + 1 MAR + 1 MNAR expert (not 1/1/3) to prevent gradient asymmetry.
6. **Temperature scaling** - Post-hoc T=1.96 improves both calibration (ECE=0.038) and accuracy (82.6%).

## Execution

```bash
# Training
python scripts/train_semisynthetic.py --config configs/training/semisynthetic_full.yaml

# Inference
python scripts/infer.py --input data.csv --checkpoint model.pt

# Evaluation
python scripts/evaluate.py --checkpoint best.pt --config config.yaml

# Calibration
python scripts/calibrate.py --checkpoint model.pt --output calibrated.pt
```
