# Project Lacuna

**Systematic Missing Data Mechanism Classification via Transformer-Based Deep Learning**

Project Lacuna is a deep learning system that classifies the *missingness mechanism* of a tabular dataset — given a CSV file with missing values (NaN), it determines *why* the data is missing, which is the essential first step for choosing the correct statistical analysis strategy. The model achieves **82.6% accuracy** on held-out synthetic test data with well-calibrated probability estimates (ECE = 0.038).

---

## Table of Contents

1. [Background: The Missing Data Problem](#background-the-missing-data-problem)
2. [What Lacuna Does](#what-lacuna-does)
3. [System Architecture](#system-architecture)
4. [File Structure](#file-structure)
5. [Component Deep-Dives](#component-deep-dives)
   - [Data Tokenization](#data-tokenization)
   - [Transformer Encoder](#transformer-encoder)
   - [Missingness Feature Extractor](#missingness-feature-extractor)
   - [Mixture of Experts](#mixture-of-experts)
   - [Bayes-Optimal Decision Rule](#bayes-optimal-decision-rule)
   - [Generator System (Training Data)](#generator-system-training-data)
   - [Training Pipeline](#training-pipeline)
   - [Post-Hoc Temperature Calibration](#post-hoc-temperature-calibration)
6. [Data Flow: End-to-End](#data-flow-end-to-end)
7. [Key Design Decisions and Rationale](#key-design-decisions-and-rationale)
8. [Experiment History and Ablation Studies](#experiment-history-and-ablation-studies)
9. [Final Model Performance](#final-model-performance)
10. [Configuration System](#configuration-system)
11. [User-Facing Interface](#user-facing-interface)
12. [Hardware and Reproducibility](#hardware-and-reproducibility)
13. [Dependencies](#dependencies)

---

## Background: The Missing Data Problem

Missing data is ubiquitous in scientific research, medical records, survey datasets, and observational studies. The standard statistical machinery for handling missingness (multiple imputation, complete-case analysis, sensitivity analysis) is only valid when applied to data whose *mechanism* matches the method's assumptions. Using the wrong method can produce systematically biased results.

The field classifies missingness mechanisms into three canonical types, introduced by Rubin (1976):

### MCAR — Missing Completely At Random
The probability of a value being missing is entirely independent of both observed and unobserved data. For example, a lab technician drops samples at random, or a sensor fails randomly. Formally: `P(R | X_obs, X_mis) = P(R)`.

**Implication:** Complete-case analysis (simply dropping rows with missing values) is valid and unbiased, though inefficient. Simple imputation with the column mean is also acceptable.

### MAR — Missing At Random
The probability of missingness depends on *observed* variables but not on the missing values themselves. For example, younger patients are less likely to report income (age is observed; missingness depends on age, not on the unreported income). Formally: `P(R | X_obs, X_mis) = P(R | X_obs)`.

**Implication:** Complete-case analysis is biased. Multiple imputation (MICE, Amelia), likelihood-based methods (EM algorithm, Full Information Maximum Likelihood), or similar methods that condition on observed data are required for unbiased estimates.

### MNAR — Missing Not At Random
The probability of missingness depends on the *unobserved* values themselves. For example, high-earners are less likely to report their income (missingness depends on the missing income value). Formally: `P(R | X_obs, X_mis) ≠ P(R | X_obs)`.

**Implication:** Even multiple imputation is insufficient. Sensitivity analyses, selection models (Heckman correction), or pattern-mixture models are needed. MNAR is the most dangerous mechanism because standard methods give biased results.

### Why Automated Classification Matters

Determining the mechanism from the missingness pattern alone is a non-trivial statistical problem. Tests like Little's MCAR test can rule out MCAR, but distinguishing MAR from MNAR is theoretically impossible without strong assumptions — because MNAR, by definition, depends on what you cannot observe. This project learns a classifier from a large, diverse synthetic training distribution that covers plausible real-world patterns, providing a probabilistic recommendation rather than a definitive answer.

---

## What Lacuna Does

Given a CSV file with missing values:

1. **Tokenizes** the dataset into a tensor representation preserving row-column structure
2. **Encodes** the dataset through a transformer into a 64-dimensional evidence vector
3. **Extracts** 16 explicit statistical features from the missingness pattern
4. **Classifies** via a Mixture of Experts network into posterior probabilities: P(MCAR), P(MAR), P(MNAR)
5. **Decides** via a Bayes-optimal rule (using an asymmetric loss matrix) into one of three recommended actions: GREEN, YELLOW, or RED
6. **Reports** human-readable output with probabilities, confidence, and recommended analysis method

---

## System Architecture

The architecture is designed around a core insight: **the missingness pattern in tabular data encodes information about the mechanism**, but this signal is subtle and distributed across rows and columns. A simple summary statistic is insufficient; the model needs to learn cross-column dependencies.

```
Input: CSV with NaN values
          ↓
  ┌────────────────────────────────────────────────────────┐
  │  TOKENIZATION (data/tokenization.py)                   │
  │  Each cell → [value, is_observed, mask_type, feat_id]  │
  │  Output: TokenBatch [B, max_rows, max_cols, 4]         │
  └────────────────┬───────────────────────────────────────┘
                   ↓
  ┌────────────────────────────────────────────────────────┐
  │  TRANSFORMER ENCODER (models/encoder.py)               │
  │  Row-level self-attention (4 layers, 128-dim, 4 heads) │
  │  Two-stage attention pooling: features→rows→dataset    │
  │  Output: Evidence vector [B, 64]                       │
  └──────────────┬───────────────────────────────┬─────────┘
                 │                               │
                 ↓                               ↓
  ┌──────────────────────────┐   ┌───────────────────────────┐
  │ RECONSTRUCTION HEADS (3) │   │  MISSINGNESS FEATURES     │
  │ (models/reconstruction/) │   │  (data/missingness_       │
  │ Predict masked values    │   │   features.py)            │
  │ → per-head MSE errors    │   │  16 statistical features  │
  └──────────┬───────────────┘   └───────────────┬───────────┘
             │                                   │
             └──────────────┬────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────┐
  │  MIXTURE OF EXPERTS (models/moe.py)                     │
  │  Gating: [evidence, recon_errors, miss_features]        │
  │  3 symmetric experts (1 MCAR, 1 MAR, 1 MNAR)           │
  │  Output: PosteriorResult [P(MCAR), P(MAR), P(MNAR)]     │
  └────────────────────────┬────────────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────────────┐
  │  BAYES-OPTIMAL DECISION RULE (models/decision.py)       │
  │  Asymmetric loss matrix: action × true_class costs      │
  │  Output: Decision (GREEN / YELLOW / RED)                │
  └─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
project_lacuna/
│
├── lacuna/                          # Core Python package
│   ├── core/                        # Foundational types and utilities
│   │   ├── types.py                 # Immutable dataclasses: ObservedDataset, TokenBatch,
│   │   │                            #   ReconstructionResult, MoEOutput, PosteriorResult,
│   │   │                            #   Decision, LacunaOutput
│   │   ├── logging.py               # Structured logging utilities
│   │   ├── exceptions.py            # Custom exceptions: ValidationError, CheckpointError,
│   │   │                            #   NumericalError, RegistryError
│   │   ├── validation.py            # Input validation functions
│   │   └── rng.py                   # Reproducible RNG state management
│   │
│   ├── data/                        # Data pipeline
│   │   ├── tokenization.py          # Row-level tokenization (BERT-inspired)
│   │   ├── batching.py              # Synthetic and validation data loaders
│   │   ├── missingness_features.py  # Extract 16 statistical features
│   │   ├── normalization.py         # Quantile-based feature normalization
│   │   ├── ingestion.py             # CSV/dataset loading utilities
│   │   ├── semisynthetic.py         # Real data + synthetic missingness injection
│   │   ├── observed.py              # Processing utilities for observed datasets
│   │   └── catalog.py               # Dataset catalog and metadata management
│   │
│   ├── generators/                  # 110 parameterized missingness generators
│   │   ├── base.py                  # Abstract Generator interface
│   │   ├── registry.py              # GeneratorRegistry — validated finite set
│   │   ├── params.py                # Parameter containers for generators
│   │   ├── priors.py                # Sampling priors (uniform, class-balanced)
│   │   └── families/
│   │       ├── base_data.py         # Base data generation (normal, exponential, etc.)
│   │       ├── mcar/                # 34 MCAR generators
│   │       │   ├── bernoulli.py     # Fixed per-column missingness probability
│   │       │   ├── blocks.py        # Block-structured missingness
│   │       │   ├── column_effects.py
│   │       │   ├── conditional.py
│   │       │   ├── distributional.py
│   │       │   ├── multilevel.py
│   │       │   └── row_effects.py
│   │       ├── mar/                 # 36 MAR generators
│   │       │   ├── simple.py        # Single observed column predicts missingness
│   │       │   ├── complex.py       # Non-linear observed-to-missing relationships
│   │       │   ├── multiple.py      # Multiple observed predictors
│   │       │   ├── structural.py    # Structural patterns (survey skip logic, etc.)
│   │       │   ├── survey.py        # Survey-style patterns
│   │       │   └── strength.py      # Variable signal strength
│   │       └── mnar/                # 38 MNAR generators
│   │           ├── censoring.py     # Left/right censoring at thresholds
│   │           ├── detection.py     # Detection limit mechanisms
│   │           ├── informative.py   # Informative dropout
│   │           ├── latent.py        # Latent variable dependencies
│   │           ├── selection.py     # Selection bias models
│   │           ├── self_censoring.py# High/low values self-censor
│   │           ├── social.py        # Social network effects
│   │           └── strategic.py     # Strategic omission
│   │
│   ├── models/                      # Neural network architecture
│   │   ├── encoder.py               # Transformer encoder (tokenization → evidence)
│   │   ├── assembly.py              # LacunaModel: assembles all components
│   │   ├── moe.py                   # Mixture of Experts gating network
│   │   ├── decision.py              # Bayes-optimal decision rule
│   │   ├── aggregator.py            # Expert output aggregation
│   │   └── reconstruction/
│   │       ├── base.py              # Base reconstruction head architecture
│   │       └── heads.py             # Per-expert reconstruction heads
│   │
│   ├── training/                    # Training infrastructure
│   │   ├── trainer.py               # Main training loop (warmup, scheduling, early stop)
│   │   ├── loss.py                  # Multi-task loss: mechanism + reconstruction + balance
│   │   ├── calibration.py           # Post-hoc temperature scaling (Guo et al. 2017)
│   │   ├── checkpoint.py            # Checkpoint save/load with versioning
│   │   ├── report.py                # Evaluation metrics and reporting
│   │   └── logging.py               # Training metrics logging
│   │
│   ├── metrics/                     # Evaluation metrics
│   │   ├── classification.py        # Accuracy, F1, per-class breakdown
│   │   ├── calibration.py           # ECE, Brier score, reliability diagrams
│   │   └── uncertainty.py           # Entropy, confidence analysis
│   │
│   ├── cli/                         # CLI entry points
│   │   ├── train.py                 # lacuna-train
│   │   ├── infer.py                 # lacuna-infer
│   │   └── eval.py                  # lacuna-eval
│   │
│   ├── config/                      # Configuration system
│   │   ├── schema.py                # Configuration dataclasses with validation
│   │   ├── load.py                  # YAML loading and merging
│   │   └── hashing.py               # Config hashing for caching
│   │
│   └── utils/
│       ├── device.py                # GPU/CPU device management
│       └── io.py                    # File I/O utilities
│
├── scripts/                         # Runnable scripts
│   ├── infer.py                     # ← USER-FACING: classify a CSV's mechanism
│   ├── train.py                     # ← Training entry point (semi-synthetic only)
│   ├── evaluate.py                  # Full evaluation with detailed reporting
│   ├── calibrate.py                 # Post-hoc temperature scaling
│   ├── run_ablation.py              # Missingness-feature ablation sweep
│   ├── validate_generators.py       # Validate generators against Little's MCAR test
│   ├── consolidate_results.py       # Gather all experiment results into JSON
│   ├── generate_dissertation_figures.py  # Generate dissertation figures + LaTeX
│   ├── download_datasets.py         # Download UCI/OpenML real datasets
│   ├── test_missingness_features.py # Diagnostic: verify feature discriminability
│   ├── diagnose_reconstruction.py   # Debug reconstruction errors by mechanism
│   ├── journal_entry.py             # Experiment logging utility
│   └── repo_stats.py                # Repository statistics
│
├── configs/                         # YAML configuration files
│   ├── training/
│   │   ├── semisynthetic.yaml       # Standard semi-synthetic training
│   │   ├── semisynthetic_minimal.yaml   # Minimal for rapid testing
│   │   ├── semisynthetic_balanced.yaml  # Class-balanced variant
│   │   ├── semisynthetic_full.yaml  # ← PRODUCTION: full semi-synthetic training
│   │   └── ablation.yaml            # Canonical ablation sweep config
│   ├── generators/
│   │   ├── lacuna_tabular_110.yaml  # ← PRODUCTION: all 110 generators
│   │   ├── lacuna_minimal_6.yaml    # 6-generator subset for testing
│   │   └── lacuna_minimal_18.yaml   # 18-generator subset
│   └── experiments/                 # Experiment-specific overrides
│
├── tests/
│   ├── unit/                        # Component-level tests
│   └── integration/                 # End-to-end pipeline tests
│
├── docs/
│   ├── DISSERTATION_DATA.md         # Complete data package for dissertation writing
│   ├── sample_output.txt            # Three example inference outputs (MAR, MCAR, MNAR)
│   └── figures/                     # Generated dissertation figures and LaTeX tables
│
├── experiments/
│   └── JOURNAL.md                   # Detailed experiment log with full metrics tables
│
├── progress_reports/                # Per-run evaluation reports
│
└── README.md                        # This file
```

---

## Component Deep-Dives

### Data Tokenization

**File:** `lacuna/data/tokenization.py`

The tokenization step converts a tabular dataset — a matrix of numeric values with some cells missing — into a structured tensor that the transformer can process. This is the most architecturally significant design choice in the entire system.

Each cell in the dataset is encoded as a **4-dimensional token**:

```
token[row, col] = [value, is_observed, mask_type, feature_id_normalized]
```

- **value**: The numeric cell value, normalized per-column to zero mean and unit variance using the observed values. Missing cells get a zero value (the normalization mean).
- **is_observed**: 1.0 if the value is present in the original data, 0.0 if it is missing (NaN). This is the primary signal of the missingness pattern.
- **mask_type**: Distinguishes *natural* missingness (0.0, from the original dataset) from *artificial* missingness (1.0, imposed during training for self-supervised reconstruction). At inference time, all missingness is natural.
- **feature_id_normalized**: The column index divided by `max_cols`. This provides positional information so the model can track which column a token belongs to, supplemented by learnable position embeddings.

The resulting tensor has shape `[B, max_rows, max_cols, 4]` where `B` is batch size, `max_rows` is a padding limit (typically 256), and `max_cols` is a padding limit (typically 32–48). Datasets smaller than these limits are padded; the row and column masks (`row_mask`, `col_mask`) tell the transformer which positions are real versus padding.

**Why row-level tokenization?** The key is that the transformer attends *across columns within a row*, not across the entire flattened sequence. This preserves the two-dimensional structure of the data. For MAR detection specifically, the model must learn that "missingness in column j correlates with the observed value in column k within the same row." Flattening the entire dataset into a sequence would lose this intra-row structure, or require attending over a sequence of length `n × d`, which is infeasible.

---

### Transformer Encoder

**File:** `lacuna/models/encoder.py`

The encoder transforms the `[B, max_rows, max_cols, 4]` token tensor into a single `[B, 64]` evidence vector representing the dataset-level signal about the missingness mechanism.

**Step 1 — Token Embedding:**
The 4-dimensional raw tokens are projected to `hidden_dim` (128 by default) via a learned linear layer. Additionally, learnable embeddings are added for the `is_observed` flag, the `mask_type` flag, and the column position. The combination of normalized feature IDs (in the token) and learnable position embeddings allows the model to both generalize across datasets with different column counts and specialize on particular column positions within a dataset.

**Step 2 — Transformer Layers (4 layers):**
Standard transformer encoder layers, applied *row by row* — each layer performs multi-head self-attention over the `max_cols` tokens *within* each row independently. There is no cross-row attention in the transformer itself; cross-row information is aggregated only in the pooling steps below. This keeps the attention complexity at `O(max_cols²)` per row rather than `O((max_rows × max_cols)²)`, which would be intractable. The `col_mask` is applied to exclude padding columns from attention. Pre-norm architecture (layer norm before attention) is used for training stability.

**Step 3 — Row Pooling (features → rows):**
After the transformer layers, each row is represented by `max_cols` token vectors. These are collapsed to a single per-row vector via **attention-based pooling**: a learned query vector attends over the column tokens, producing a weighted average. This is preferable to simple mean or max pooling because it lets the model learn which columns are most informative about the mechanism. Output shape: `[B, max_rows, 128]`.

**Step 4 — Dataset Pooling (rows → dataset):**
The `max_rows` row vectors are similarly collapsed to a single dataset-level evidence vector via another attention pooling step with a learned query, followed by a linear projection to `evidence_dim` (64). Output shape: `[B, 64]`.

The 64-dimensional evidence vector is the compressed representation of the entire dataset's missingness pattern. It is fed into the MoE gating network alongside the explicit statistical features and reconstruction errors.

**Configuration (`EncoderConfig`):**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `hidden_dim` | 128 | Transformer hidden dimension |
| `evidence_dim` | 64 | Output evidence vector size |
| `n_layers` | 4 | Number of transformer layers |
| `n_heads` | 4 | Attention heads per layer |
| `max_cols` | 32–48 | Maximum columns (position embedding limit) |
| `dropout` | 0.1 | Regularization |
| `row_pooling` | "attention" | Row aggregation strategy |
| `dataset_pooling` | "attention" | Dataset aggregation strategy |

---

### Missingness Feature Extractor

**File:** `lacuna/data/missingness_features.py`

The transformer learns implicit representations of the missingness pattern, but some signals require explicit statistical computation. The missingness feature extractor computes 16 handcrafted features that directly target the distinguishing characteristics of each mechanism:

**Missing rate statistics (4 features):**
- Mean missing rate across columns
- Variance of missing rates across columns — MCAR should be near-zero (uniform), MAR/MNAR can be higher
- Range (max − min) of per-column missing rates
- Max per-column missing rate

**Point-biserial correlations (3 features):**
Point-biserial correlation measures the correlation between the binary missingness indicator of one column and the continuous observed values in another column. High correlations indicate MAR (observed values predict missingness).
- Mean correlation across all column pairs
- Max correlation (strongest observed-to-missingness relationship)
- Standard deviation of correlations

**Cross-column missingness correlations (3 features):**
The Pearson correlation between missingness indicators of different column pairs. High correlations indicate structured missingness (block patterns in MAR, monotone patterns in MNAR).
- Mean correlation
- Max correlation
- Standard deviation

**Distributional statistics (4 features):**
MNAR mechanisms distort the observed value distribution because the probability of observing a value depends on its magnitude.
- Mean skewness of observed value distributions per column
- Mean kurtosis of observed value distributions per column
- Variance of skewness across columns
- Variance of kurtosis across columns

**Little's MCAR test approximation (2 features):**
An approximation of Little's (1988) MCAR test statistic, which tests the null hypothesis of MCAR by comparing observed column means across different missingness patterns.
- Chi-squared test statistic approximation
- P-value proxy

These 16 features are concatenated with the evidence vector and the reconstruction errors to form the input to the MoE gating network. They provide a "cheat sheet" of summary statistics that complement the learned evidence vector.

---

### Mixture of Experts

**File:** `lacuna/models/moe.py`

The Mixture of Experts (MoE) layer takes the combined signal — evidence vector (64-dim), reconstruction errors (3-dim), and missingness features (16-dim) — and produces the final class posteriors.

**Architecture:**

Three symmetric experts, one per class (1 MCAR, 1 MAR, 1 MNAR). Each expert is a small feedforward network that takes the combined input and produces a 3-dimensional class posterior vector `[P(MCAR), P(MAR), P(MNAR)]`.

A **gating network** — a 2-layer MLP with tanh activations — receives the same input and produces 3 logits, one per expert. After temperature-scaled softmax, these become gating probabilities `g = [g₁, g₂, g₃]`.

The final output is the gated mixture:
```
output = g₁ × expert₁(x) + g₂ × expert₂(x) + g₃ × expert₃(x)
```

Each expert specializes in recognizing its mechanism, while the gating network learns to route inputs appropriately based on the available evidence.

**Reconstruction Heads (one per expert):**
Each expert has a paired reconstruction head that tries to predict the values of artificially masked cells from the token representations. The mean squared error of each head's predictions is included as an input to the gating network: if the MCAR reconstruction head does particularly well (or poorly) on a dataset, that's informative about whether the underlying mechanism is MCAR. These heads provide self-supervised learning signal during training even when class labels are noisy.

**Temperature Scaling:**
The gating network includes a learned `log_temperature` parameter. During post-hoc calibration, this is set to `log(1.96)` = 0.673, which softens the softmax output to improve probability calibration.

**Why symmetric 1/1/1 experts?** See the [Design Decisions](#key-design-decisions-and-rationale) section for the critical story of how asymmetric experts (1/1/3) caused 16+ percentage points of MAR underdetection.

**Load Balancing:**
Without intervention, a MoE gating network can collapse: one expert wins all the routing weight and the others degenerate to unused parameters. Two complementary mechanisms prevent this.

*Switch Transformer loss* (used during training, `lacuna/training/loss.py`): For each batch, compute the fraction of samples hard-routed to each expert (`f_i`) and the average soft probability assigned to each expert (`p_i`). The loss is their dot product scaled by the number of experts:

```
load_balance_loss = n_experts × Σᵢ (fᵢ × pᵢ)
```

`fᵢ` counts hard assignments (argmax); `pᵢ` is the mean softmax probability. When routing is uniform, both are `1/n_experts` and the loss equals 1. Imbalanced routing raises the product, penalizing it. This formulation comes directly from the [Switch Transformer paper (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961).

*KL divergence loss* (used in the MoE layer itself, `lacuna/models/moe.py`): Computes the KL divergence of the mean gating probability distribution from a uniform target:

```
avg_probs = gate_probs.mean(dim=0)       # mean probability per expert across batch
target    = uniform distribution (1/n_experts each)
loss      = KL(avg_probs ‖ target)
```

Both losses are weighted by `load_balance_weight` (default `0.01`) and added to the total training loss. With only 3 experts, expert collapse is less catastrophic than in large-scale MoE models, but the regularization still meaningfully stabilizes training.

---

### Bayes-Optimal Decision Rule

**File:** `lacuna/models/decision.py`

The MoE produces posterior probabilities `p = [P(MCAR), P(MAR), P(MNAR)]`. The decision rule converts these into a recommended action that minimizes expected cost under an asymmetric loss matrix.

**Loss Matrix:**
The matrix `L[action, true_class]` encodes the practical cost of each misclassification:

|  | True MCAR | True MAR | True MNAR |
|--|-----------|----------|-----------|
| **Green (assume MCAR)** | 0.0 | 0.3 | 1.0 |
| **Yellow (assume MAR)** | 0.2 | 0.0 | 0.2 |
| **Red (assume MNAR)** | 1.0 | 0.3 | 0.0 |

The asymmetry encodes key domain knowledge:
- Treating MNAR data as MCAR (Green when truly Red): risk = 1.0 — produces biased estimates, invalid inference
- Treating MCAR data as MNAR (Red when truly Green): risk = 1.0 — sensitivity analysis on random noise, wasted effort plus potential false alarms
- Treating MCAR data as MAR (Yellow when truly Green): risk = 0.2 — conservative but unbiased; multiple imputation works fine on MCAR data
- Treating MNAR data as MAR (Yellow when truly Red): risk = 0.2 — multiple imputation is insufficient but less harmful than complete-case analysis

**Decision Computation:**
For each action `a`, compute the expected risk:
```
risk(a) = Σ_k L[a, k] × P(class_k)
```
Select the action with minimum expected risk. This Bayes-optimal rule means that, for example, if P(MNAR) is only moderately elevated, the rule may still recommend Yellow (MAR treatment) because the asymmetric costs favor conservative caution over confident misclassification.

**Actions:**
- **Green**: Assume MCAR → complete-case analysis or simple imputation acceptable
- **Yellow**: Assume MAR → multiple imputation (MICE, Amelia) or EM/FIML required
- **Red**: Assume MNAR → sensitivity analysis, selection models (Heckman), or pattern-mixture models required

---

### Generator System (Training Data)

**Files:** `lacuna/generators/`

Because missingness mechanisms are unobservable in real data, the model is trained on **synthetic data with known labels**. The generator system creates diverse, realistic missingness patterns spanning the space of plausible mechanisms.

**110 Parameterized Generators:**

| Class | Count | Fraction (uniform sampling) |
|-------|-------|-----------------------------|
| MCAR | 34 | ~31% |
| MAR | 36 | ~33% |
| MNAR | 38 | ~35% |

Each generator:
1. Generates base data (from configurable distributions: Gaussian, exponential, skewed, mixed, correlated)
2. Applies a mechanism-specific missingness pattern to produce a mask `R ∈ {0, 1}^{n×d}`
3. Returns `ObservedDataset(data=X * R, mask=R, class_id=class, generator_id=id)`

**MCAR Generators (34):** Missingness with no structure. Examples:
- `BernoulliMCAR`: Each cell missing independently with fixed probability p
- `BlocksMCAR`: Block-structured missingness that remains statistically uniform
- `RowEffectsMCAR`: Row-level random effects that don't depend on values
- `DistributionalMCAR`: Missingness drawn from a continuous distribution

**MAR Generators (36):** Missingness driven by observed column values. Examples:
- `SimpleMARPredictor`: Column j missing when observed column k exceeds a threshold
- `ComplexMAR`: Non-linear (logistic) relationship between observed value and missingness probability
- `MultipleMAR`: Multiple observed columns together predict missingness via logistic regression
- `SurveyMAR`: Skip-logic patterns (question B only appears if question A answered a certain way)
- `WeakMAR`: Very weak signal, borderline MCAR — exists to train the model on ambiguous cases

**MNAR Generators (38):** Missingness driven by the unobserved values themselves. Examples:
- `SelfCensoringMNAR`: High or low values in column j cause column j to be missing (threshold censoring)
- `CensoringMNAR`: Observations below the detection limit are missing (left-censoring)
- `LatentMNAR`: A hidden latent variable drives both the value and its missingness
- `SelectionMNAR`: Selection bias — units with certain characteristics both have higher values and are more likely to be missing
- `StrategicMNAR`: Strategic omission — entities omit data when it is unfavorable

**Registry and Priors (`generators/registry.py`, `generators/priors.py`):**
Generators are organized into a `GeneratorRegistry` from YAML configs (`configs/generators/lacuna_tabular_110.yaml`). At training time, the `GeneratorPrior` determines how generators are sampled:
- **Uniform prior** (production default): Sample uniformly over the 110 generators, giving slightly more MNAR samples by count
- **Class-balanced prior** (abandoned after Exp 8): Force exactly 1/3 per class — found to hurt performance by overweighting ambiguous generators

**Semi-Synthetic Training (`data/semisynthetic.py`):**
For production training, real datasets from UCI and OpenML are used as the base data distribution. The generator's missingness pattern is then *synthetically imposed* on top. This gives the model exposure to realistic covariance structures, marginal distributions, and column count diversity, while maintaining known ground-truth labels. 24 training datasets and 7 validation datasets from UCI/OpenML were used.

---

### Training Pipeline

**Files:** `lacuna/training/trainer.py`, `lacuna/training/loss.py`

**Multi-Task Loss:**

The total training loss combines three terms:

```
total_loss = mechanism_weight × mechanism_loss
           + reconstruction_weight × reconstruction_loss
           + load_balance_weight × moe_load_balance_loss
```

- **mechanism_loss**: Cross-entropy between the MoE's class posteriors and the true class label. This is the primary supervision signal. (Brier score was tried in Experiment 7 but found inferior — see Ablation Studies.)

- **reconstruction_loss**: Mean squared error between each expert's reconstruction head predictions and the true values at artificially masked positions. This is a self-supervised signal that trains the encoder to produce useful representations even without class label supervision.

- **moe_load_balance_loss**: Encourages the gating network to distribute samples across experts rather than collapsing to a single expert. Prevents the pathological case where one expert handles everything and the others degenerate.

**Artificial Masking for Reconstruction:**
During training, 15% of *observed* values are additionally masked as `mask_type=1.0` (artificial). The reconstruction heads must predict these values from the remaining context. This is inspired by BERT's masked language modeling, adapted for tabular data. The artificial mask is separate from the natural missingness mask, so the model simultaneously learns to reconstruct naturally-missing values AND artificially-withheld values.

**Training Configuration:**

| Hyperparameter | Production Value |
|----------------|-----------------|
| Epochs | 100 |
| Batch size | 16 |
| Batches per epoch | 100 |
| Validation batches | 50 |
| Learning rate | 0.0003 |
| Warmup steps | 200 |
| Early stopping patience | 10 |
| LR schedule | Cosine decay |
| Optimizer | AdamW |
| Gradient clipping | Norm-based |
| Hardware | NVIDIA RTX 5070 Ti |

**Trainer Features (`trainer.py`):**
- Three training modes: `pretraining` (reconstruction only), `classification`, `joint`
- Cosine/linear/constant LR schedules with linear warmup
- Early stopping on validation loss or accuracy
- Best-model checkpointing + periodic snapshots
- Optional mixed-precision (AMP) for GPU acceleration
- Each epoch regenerates fresh batches from generators (preventing overfitting to fixed training sets)

---

### Post-Hoc Temperature Calibration

**Files:** `lacuna/training/calibration.py`, `scripts/calibrate.py`

After training, the model's softmax outputs are often poorly calibrated — the model is too confident in its predictions. Post-hoc temperature scaling (Guo et al. 2017) fixes this by finding a single scalar parameter T that, when applied as `softmax(logits / T)`, minimizes the Expected Calibration Error (ECE) on a held-out validation set.

**Two-Phase Grid Search:**
1. Coarse search: 100 log-uniformly spaced values of T in [0.1, 10.0]
2. Fine search: 100 linearly spaced values in a window around the coarse minimum

**Implementation:** The calibration script (`scripts/calibrate.py`) loads a trained checkpoint, runs forward passes on validation data to collect pre-softmax gate logits, runs the grid search to find optimal T, patches `model.moe.gating.log_temperature = log(T)`, and saves a new `calibrated.pt` checkpoint.

**Unexpected Finding:** Temperature scaling with T=1.96 not only improved calibration (ECE: 0.116 → 0.038) but also improved accuracy on all three classes simultaneously (+4.2 points overall). This is theoretically surprising — temperature scaling does not change the argmax of the softmax in general. What happened: the pre-calibration model was so overconfident on borderline cases that softening the probabilities by T=1.96 frequently flipped incorrect high-confidence predictions to the correct class. This implies the model at T=1.0 was not merely miscalibrated but was actively wrong in a consistent direction that T scaling corrected.

---

## Data Flow: End-to-End

### Training Flow

```
1. GENERATOR REGISTRY
   Sample generator g from prior P(g)
   g.sample_observed(n, d) → ObservedDataset(X, R, class_id, generator_id)

2. TOKENIZATION (data/tokenization.py)
   For each row i and column j:
     token[i,j] = [X[i,j] if R[i,j]=1 else 0,   # normalized value
                   R[i,j],                          # is_observed
                   artificial_mask[i,j],            # mask_type
                   j / max_cols]                    # feature_id
   → TokenBatch shape: [B, max_rows, max_cols, 4]
     with row_mask, col_mask, original_values,
          reconstruction_mask, class_ids, generator_ids

3. ENCODER FORWARD (models/encoder.py)
   TokenEmbedding: [B, max_rows, max_cols, 4] → [B, max_rows, max_cols, 128]
   4× TransformerLayer (row-wise attention): same shape
   RowPooling: [B, max_rows, max_cols, 128] → [B, max_rows, 128]
   DatasetPooling: [B, max_rows, 128] → [B, 64]  (evidence vector)

4. RECONSTRUCTION HEADS (3 heads)
   Input: token representations [B, max_rows, max_cols, 128]
   Each head predicts: [B, max_rows, max_cols]  (reconstructed values)
   MSE against original_values at reconstruction_mask positions
   → per_head_errors: [B, 3]

5. MISSINGNESS FEATURES (data/missingness_features.py)
   Compute 16 statistics from the token batch
   → miss_features: [B, 16]

6. MOE GATING (models/moe.py)
   concat([evidence, per_head_errors, miss_features]) → [B, 83]
   Gating network (2-layer MLP): → [B, 3] logits → softmax → [B, 3] gate probs
   Expert₁(input) → [B, 3],  Expert₂(input) → [B, 3],  Expert₃(input) → [B, 3]
   output = Σᵢ gate[i] × expertᵢ(input)  → [B, 3] class posterior

7. LOSS COMPUTATION (training/loss.py)
   mechanism_loss = CrossEntropy(class_posterior, class_ids)
   reconstruction_loss = Σ MSE(reconstructed[head], original_values[reconstruction_mask])
   load_balance_loss = variance(gate_probs.mean(0)) * penalty_weight
   total_loss = λ₁ × mechanism_loss + λ₂ × reconstruction_loss + λ₃ × load_balance_loss

8. BACKWARD PASS
   total_loss.backward()
   optimizer.step()  (AdamW)
```

### Inference Flow

```
1. INPUT: CSV file
   pd.read_csv() → drop non-numeric → convert NaN → X [n, d], R [n, d]
   → ObservedDataset

2. TOKENIZATION: Same as training (no artificial masking)

3. FORWARD PASS: Same as training steps 3–6

4. TEMPERATURE SCALING
   logits = logits / T  (T = 1.96 post-calibration)
   → calibrated_probs = softmax(logits)

5. BAYES DECISION RULE (models/decision.py)
   For each action a ∈ {Green, Yellow, Red}:
     expected_risk(a) = Σ_k L[a,k] × P(class_k)
   recommended_action = argmin_a expected_risk(a)

6. OUTPUT (scripts/infer.py)
   Print classification report:
   - P(MCAR), P(MAR), P(MNAR)
   - Confidence score (max probability)
   - Entropy (uncertainty measure)
   - Recommended action with explanation
   - Bayes risk per action
```

---

## Key Design Decisions and Rationale

### Decision 1: Row-Level Tokenization (Not Full-Dataset Flattening)

**Choice:** Attend across columns *within* rows; then aggregate rows via pooling.

**Why:** MAR detection fundamentally requires comparing "column j is missing" with "column k has value v in the same row." This is a within-row relationship. Full-dataset flattening into a sequence of length `n × d` would either lose row identity (losing intra-row relationships) or be computationally infeasible for the datasets of interest (n up to 500, d up to 20 → sequences of 10,000 tokens per dataset, nested within a batch).

**Alternative considered:** Treating each column's missingness pattern as a time series across rows, attending across rows per column. Rejected because MAR detection requires cross-column relationships, which this per-column approach would not directly capture.

---

### Decision 2: Explicit Missingness Features Alongside Learned Representations

**Choice:** Compute 16 statistical features and concatenate to the evidence vector as input to the gating network.

**Why:** Diagnostic analysis (Experiment 2) confirmed that these features have high effect sizes: point-biserial correlations separate MCAR from MAR with d > 9.0; Little's test approximation separates MAR from MNAR with d > 2.8. Including these as explicit inputs gives the gating network direct access to the strongest discrimination signals rather than forcing the transformer to rediscover them from scratch. The transformer's learned evidence vector is complementary: it can capture subtle pattern-level signals that the handcrafted features miss.

---

### Decision 3: Symmetric 1/1/1 Expert Architecture

**Choice:** One expert per class (MCAR, MAR, MNAR), 1:1 mapping, no class aggregation.

**Original design:** 5 experts mapped `[MCAR, MAR, MNAR, MNAR, MNAR]` with mean aggregation across experts per class. This was designed to give MNAR more representational capacity (since MNAR has more diverse mechanisms).

**Why it failed:** Despite mean aggregation correcting the class prior mathematically, the 3 MNAR experts created asymmetric gradient flow during backpropagation. Each MCAR and MAR expert received gradients through 1 output head; each of 3 MNAR experts received gradients through 1 head but the MNAR *class* received gradient through 3 heads simultaneously. The gating network learned to route borderline MAR cases to MNAR because doing so reduced loss across 3 MNAR expert outputs rather than 1 MAR expert output — a pure artifact of the aggregation geometry.

**Result of fix:** Switching to 1/1/1 improved MAR recall from 52.6% → 69.3% (+16.7 points) while simultaneously improving overall accuracy (77.0% → 78.4%) — a "free lunch" from fixing the architecture.

---

### Decision 4: Uniform Generator Prior (Not Class-Balanced)

**Choice:** Sample uniformly over generators, accepting slight MNAR overrepresentation (~35% vs 33% for MCAR/MAR).

**Alternative tried:** Class-balanced prior forcing exactly 1/3 per class (Experiment 8). This worsened performance: accuracy dropped from 77.0% to 62.6% and MAR recall dropped from 52.6% to 34.8%.

**Why it backfired:** The balanced prior forces the model to see more of the ambiguous, hard-to-detect MAR generators (like MAR-Weak, MAR-Interactive, MAR-MixedPred that produce genuine MAR/MNAR overlap). Under the uniform prior, these difficult generators are a smaller fraction of training, allowing the model to first learn the detectable cases. Forcing them to 1/3 of training "drowns the signal in noise."

**Implication:** The ~35% MNAR overrepresentation under uniform sampling is not a bug — it reflects that there are more distinct MNAR mechanisms in the generator library, and the model should see them more often.

---

### Decision 5: Post-Hoc Temperature Scaling (Guo et al. 2017)

**Choice:** Apply single-parameter temperature scaling after training rather than any in-training calibration approach.

**Why:** Post-hoc methods are simple, proven, and require no retraining. The single scalar T affects all classes equally, making it impossible to overfit. Cross-entropy training naturally pushes probabilities toward 0 and 1, creating overconfidence that temperature scaling reliably corrects.

**Unexpected bonus:** T=1.96 improved not just calibration but accuracy across all three classes. See [Post-Hoc Temperature Calibration](#post-hoc-temperature-calibration) for the mechanism.

---

### Decision 6: Asymmetric Loss Matrix for Decision Making

**Choice:** Use a domain-specific asymmetric loss matrix rather than plain argmax on class probabilities.

**Why:** Plain argmax ignores the stakes. In the missing data context, MCAR → MNAR confusion (treating MNAR data as MCAR) produces biased estimates and invalid confidence intervals — a fundamental statistical error. MCAR → MAR confusion (applying multiple imputation to MCAR data) is wasteful but not harmful. The asymmetric matrix encodes this: L[Green, MNAR] = 1.0 but L[Yellow, MCAR] = 0.2. This means the decision rule tilts toward Yellow (MAR treatment) in cases of uncertainty, which is the statistically conservative choice.

---

### Decision 7: Semi-Synthetic Training Data

**Choice:** Use real datasets (UCI/OpenML) as base data distributions, apply synthetic missingness mechanisms on top.

**Why:** Purely synthetic base data (e.g., Gaussian columns) would not capture the covariance structures, skewness, multimodality, and column count distributions found in real tabular datasets. Semi-synthetic training gives the model exposure to realistic marginal and joint distributions while maintaining known ground-truth labels.

**Datasets used:** 24 training + 7 validation datasets from UCI Machine Learning Repository and OpenML, covering a variety of domains (biological, social, economic, medical).

---

### Decision 8: Cross-Entropy Loss Over Brier Score

**Choice:** Cross-entropy as the mechanism classification loss.

**Alternative tried:** Brier score `(1/K) Σ (pₖ - yₖ)²` combined with label smoothing=0.1 (Experiment 7). This caused catastrophic regression: accuracy dropped to 54.8% and ECE *worsened* to 0.287.

**Why CE is better here:** Cross-entropy's gradient is `∂L/∂p = -1/p` — infinitely sharp as `p → 0`, forcing the model to concentrate probability mass on the correct class. Brier score's gradient is `∂L/∂p = 2(p - y)` — bounded regardless of how wrong the model is. With 110 generators and limited training time, the weaker Brier gradient cannot push the model hard enough to find sharp MAR/MNAR boundaries.

---

## Experiment History and Ablation Studies

The development of Lacuna proceeded through 10 experiments, with detailed logs in `experiments/JOURNAL.md`. The quantitative trajectory is:

| Exp | Name | Key Change | Accuracy | MAR Recall | ECE | Status |
|-----|------|-----------|----------|------------|-----|--------|
| 1 | Core Architecture | Build full pipeline | — | — | — | Infrastructure |
| 2 | Diagnostic Analysis | Verify feature discriminability | — | — | — | Confirmed signals exist |
| 3 | Generator Expansion | 6 → 110 generators | — | — | — | Infrastructure |
| 4 | Test Stabilization | Fix bugs, semi-synthetic | — | — | — | Infrastructure |
| **5** | **Semi-Synthetic Baseline** | **First full training run** | **77.0%** | **52.6%** | **0.134** | ✅ Baseline |
| 6 | Evaluation Tooling | Add confusion matrix, ECE, per-generator metrics | — | — | — | Infrastructure |
| 7 | Brier + Balanced + Smoothing | 3 changes at once | 54.8% | 30.7% | 0.287 | ❌ Failed |
| 8 | Class-Balanced Prior Only | Isolate prior effect | 62.6% | 34.8% | 0.233 | ❌ Regressed |
| **9** | **1/1/1 Symmetric Experts** | **Architecture fix** | **78.4%** | **69.3%** | **0.116** | ✅ Key fix |
| **10** | **Temperature Scaling** | **Post-hoc T=1.96** | **82.6%** | **73.6%** | **0.038** | ✅ Final |

**The MAR Recall Story:**
The most important diagnostic number in this project is MAR recall — the fraction of truly MAR datasets correctly classified. Starting at 52.6% (barely better than chance in a 3-class problem), the following interventions drove it upward:
- Experiment 7 (Brier+balanced): Crashed to 30.7% — wrong direction entirely
- Experiment 8 (balanced prior only): 34.8% — still worse than baseline
- Experiment 9 (1/1/1 experts): **69.3%** — the architecture fix was the key (+16.7 points from baseline)
- Experiment 10 (temperature scaling): **73.6%** — calibration improved accuracy (+4.3 from Exp 9)

**Key Empirical Insight — Bimodal Generator Identifiability (Experiment 8):**
Per-generator analysis revealed that MAR generators split into two distinct populations:
- *Easily detected* (near-100% accuracy): MAR-ColBlocks, MAR-CrossClass, MAR-SkipLogic
- *Near-undetectable* (0–15% accuracy): MAR-Weak, MAR-Interactive, MAR-Section, MAR-MixedPred

Some MAR mechanisms produce patterns genuinely indistinguishable from MNAR given only observed data. MAR-Interactive creates structured patterns that overlap with MNAR self-censoring. MAR-MixedPred (multi-predictor MAR) appears visually similar to MNAR. This is not a failure of the model — it is a fundamental statistical reality: the MAR/MNAR boundary is blurry for certain generating mechanisms. The 26% remaining MAR misclassification in the final model largely reflects this theoretical irreducibility.

---

## Final Model Performance

**Model:** `calibrated.pt` — 1/1/1 symmetric MoE, T=1.96

### Accuracy and Classification

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| MCAR | 79.4% | 94.5% | 86.3% | 163 |
| MAR | 91.8% | 73.6% | 81.7% | 288 |
| MNAR | 78.7% | 84.5% | 81.5% | 349 |
| **Overall** | | **82.6%** | | **800** |

### Confusion Matrix

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 154 | 0 | 9 |
| **True MAR** | 5 | 212 | 71 |
| **True MNAR** | 35 | 19 | 295 |

### Calibration

| Metric | Value |
|--------|-------|
| ECE (Expected Calibration Error) | 0.038 |
| Temperature (post-hoc) | 1.96 |
| Mean confidence (correct) | 0.818 |
| Mean confidence (incorrect) | 0.733 |

### Selective Accuracy (Coverage vs. Accuracy at Confidence Thresholds)

| Confidence threshold | Accuracy | Coverage |
|---------------------|----------|----------|
| 0.50 | 83.2% | 97.0% |
| 0.60 | 84.3% | 89.9% |
| 0.70 | 87.7% | 77.4% |
| 0.80 | 90.3% | 61.8% |
| 0.90 | 91.5% | 30.8% |

The Bayes decision rule is most valuable here: by abstracting from raw probabilities to expected risk, the model can make useful recommendations even for the 30% of cases where the model abstains at τ=0.90.

### Model Size

| Component | Parameters |
|-----------|-----------|
| Transformer encoder | ~580,000 |
| MoE (3 experts + gating) | ~180,000 |
| Reconstruction heads (3) | ~100,000 |
| Missingness feature MLP | ~40,000 |
| **Total** | **~901,130** |

---

## Configuration System

**Files:** `lacuna/config/schema.py`, `lacuna/config/load.py`, `configs/`

All training and model parameters are specified in YAML configuration files. This enables reproducible, version-controlled experiments.

**Production training configuration (`configs/training/semisynthetic_full.yaml`):**

```yaml
seed: 42
device: cuda
output_dir: /mnt/artifacts/project_lacuna/runs

data:
  max_rows: 256          # Tokenization padding limit (row subsampling cap)
  max_cols: 48           # Tokenization padding limit (column cap)
  train_datasets: [...]  # Catalog names of 24 UCI/OpenML/sklearn datasets
  val_datasets:   [...]  # Catalog names of 7 held-out datasets

model:
  hidden_dim: 128
  evidence_dim: 64
  n_layers: 4
  n_heads: 4
  dropout: 0.1
  row_pooling: attention
  dataset_pooling: attention

training:
  epochs: 100
  batch_size: 16
  batches_per_epoch: 100
  val_batches: 50
  lr: 0.0003
  warmup_steps: 200
  patience: 10
  lr_schedule: cosine
  mechanism_loss_type: cross_entropy

generator:
  config_name: lacuna_tabular_110

loss_weights:
  mechanism: 1.0
  reconstruction: 0.1
  load_balance: 0.01

loss_matrix:         # [action, true_class] costs
  - [0.0, 0.3, 1.0]  # Green (assume MCAR)
  - [0.2, 0.0, 0.2]  # Yellow (assume MAR)
  - [1.0, 0.3, 0.0]  # Red (assume MNAR)
```

**Generator configuration (`configs/generators/lacuna_tabular_110.yaml`):**

Specifies all 110 generators by name, class assignment, and parameter ranges. Example entry:

```yaml
generators:
  - name: BernoulliMCAR
    class: mcar
    variant: bernoulli
    params:
      p_range: [0.05, 0.50]  # Missing probability range

  - name: SimpleMAR
    class: mar
    variant: simple
    params:
      strength_range: [0.3, 0.9]  # Signal strength
      threshold_range: [0.1, 0.9]
```

---

## User-Facing Interface

**Script:** `scripts/infer.py`

```bash
python scripts/infer.py \
  --input my_data.csv \
  --checkpoint /path/to/calibrated.pt \
  --device cuda
```

**Input requirements:**
- CSV file with headers
- Numeric columns (non-numeric columns are automatically dropped with a warning)
- NaN or empty cells representing missing values
- At least 5 columns and 20 rows recommended for reliable classification

**Output format:**

```
════════════════════════════════════════════════════════════
LACUNA — Missing Data Mechanism Classifier
════════════════════════════════════════════════════════════
Dataset: patient_records.csv (482 rows × 12 columns)
Missing values: 1,247 / 5,784 cells (21.6%)

Classification:
  MCAR    12.3%
  MAR     73.6%  ← most likely
  MNAR    14.1%

Confidence: 73.6% (moderate)
Entropy: 0.42 / 1.10 (normalized: 0.38)

Recommended action: YELLOW — assume MAR
  → Use multiple imputation (e.g., MICE, Amelia) or
    likelihood-based methods (e.g., EM algorithm, FIML)

Bayes risk analysis:
  Green  (assume MCAR): risk = 0.363
  Yellow (assume MAR):  risk = 0.077  ← minimum
  Red    (assume MNAR): risk = 0.344
════════════════════════════════════════════════════════════
```

See `docs/sample_output.txt` for three complete example outputs (MAR, MCAR, and MNAR cases).

**Interpreting the output:**
- **Classification probabilities**: The model's posterior beliefs about the mechanism
- **Confidence**: The maximum class probability — higher means the model is more certain
- **Entropy**: A summary of uncertainty. Entropy = 0 means perfectly certain; entropy = log(3) ≈ 1.10 means complete uncertainty
- **Recommended action**: The Bayes-optimal choice under the asymmetric loss matrix
- **Bayes risk analysis**: The expected cost of each action given the current probabilities — the recommended action minimizes this

---

## Generating Dissertation Outputs

```bash
# On the Forge GPU server — consolidate all experiment results
python scripts/consolidate_results.py \
  --runs-dir /mnt/artifacts/project_lacuna/runs

# Generate all dissertation figures and LaTeX tables
python scripts/generate_dissertation_figures.py
```

This produces in `docs/figures/`:
- `metric_progression.pdf` — Bar chart of accuracy/recall across experiments
- `mar_journey.pdf` — Line plot of MAR recall/precision/F1 progression
- `calibration_comparison.pdf` — ECE across experiments
- `confusion_matrix_exp5.pdf`, `confusion_matrix_exp9.pdf`, `confusion_matrix_exp10.pdf`
- `tables/main_results.tex` — LaTeX table for main results section
- `tables/confusion_matrix_exp10.tex` — LaTeX confusion matrix

---

## Hardware and Reproducibility

- **Training hardware:** NVIDIA RTX 5070 Ti, CUDA 12.x
- **Training time:** ~28 minutes per experiment (100 epochs, semi-synthetic, Forge server)
- **Calibration time:** ~3 seconds (grid search over 200 temperature values)
- **Evaluation time:** ~2.5 seconds (800 samples, 50 batches)
- **Random seed:** Specified in YAML config (default: 42)
- **Software:** Python 3.x, PyTorch, custom `lacuna` package

To reproduce Experiment 10 (final model):
```bash
# Step 1: Train with 1/1/1 experts
python scripts/train.py \
  --config configs/training/semisynthetic_full.yaml \
  --generators lacuna_tabular_110 \
  --mnar-variants self_censoring \
  --device cuda --quiet --report \
  --name "Experiment 9 — 1/1/1 ablation (uniform prior)"

# Step 2: Calibrate
python scripts/calibrate.py \
  --checkpoint /path/to/best_model.pt \
  --config configs/training/semisynthetic_full.yaml \
  --generators lacuna_tabular_110 \
  --device cuda

# Step 3: Evaluate
python scripts/evaluate.py \
  --checkpoint /path/to/calibrated.pt \
  --config configs/training/semisynthetic_full.yaml \
  --generators lacuna_tabular_110 \
  --device cuda
```

---

## Dependencies

Core requirements:
- Python 3.x
- PyTorch ≥ 1.9 (with CUDA for GPU training)
- NumPy
- PyYAML
- Pandas (CSV loading in inference script)

For downloading training datasets:
- OpenML Python client
- UCI dataset utilities

Full dependency list: see `requirements.txt`
