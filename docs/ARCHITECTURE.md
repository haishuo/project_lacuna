# Lacuna Architecture

## What Lacuna Is

Lacuna is a transformer-based classifier for missing data mechanisms in tabular
datasets. Given a dataset containing missing values, it outputs a calibrated
posterior probability distribution over three mechanism classes:

- **MCAR** — Missing Completely At Random: missingness is independent of all data
- **MAR** — Missing At Random: missingness depends only on observed values
- **MNAR** — Missing Not At Random: missingness depends on the missing values themselves

This is a diagnostic problem, not an imputation problem. The output is a
dataset-level label with calibrated uncertainty, not per-cell predictions.

---

## Data Flow

```
Real Dataset (UCI / OpenML / sklearn)
         |
         v  RawDataset
+---------------------------------------------+
| SEMI-SYNTHETIC PIPELINE                     |
|  catalog.py       -> load real tabular data |
|  normalization.py -> robust MAD scaling     |
|  generator.apply_to(X) -> missingness mask  |
|  tokenization.py  -> TokenBatch [B,R,C,4]  |
+---------------------------------------------+
         |
         v  TokenBatch  [B, max_rows, max_cols, 4]
+---------------------------------------------+
| ENCODER  (models/encoder.py)                |
|  TokenEmbedding: 4 -> hidden_dim            |
|  Row-wise transformer (4 layers, 4 heads)   |
|  Attention row pooling  -> [B, hidden_dim]  |
|  Attention dataset pooling -> [B, 64]       |
+---------------------------------------------+
         |
         v  evidence  [B, 64]
+---------------------------------------------+
| PARALLEL SIGNAL EXTRACTION                  |
|                                             |
|  models/reconstruction/                     |
|    MCARHead  -> reconstruction errors       |
|    MARHead   -> reconstruction errors       |
|    MNARSelfCensoringHead -> errors          |
|    natural errors [B, 3] (on real missing)  |
|                                             |
|  data/missingness_features.py               |
|    missing rate stats (4)                   |
|    point-biserial correlations (3)          |
|    cross-column missingness corr (3)        |
|    distributional stats (4)                 |
|    Little's MCAR approximation (2)          |
|    -> features [B, 16]                      |
+---------------------------------------------+
         |
         v  concat(evidence [64], errors [3], features [16])
+---------------------------------------------+
| MIXTURE OF EXPERTS  (models/moe.py)         |
|  GatingNetwork (MLP, 2 layers)              |
|    -> gate weights  [B, n_experts]          |
|  n_experts = 3: MCAR, MAR, MNAR            |
|  Class aggregation (mean-normalised)        |
|    -> class logits  [B, 3]                  |
+---------------------------------------------+
         |
         v  class logits  [B, 3]
+---------------------------------------------+
| CALIBRATION & OUTPUT                        |
|  Post-hoc temperature scaling (T ≈ 1.96)   |
|  softmax -> P(MCAR), P(MAR), P(MNAR)       |
+---------------------------------------------+
         |
         v  PosteriorResult
+---------------------------------------------+
| BAYES-OPTIMAL DECISION  (models/assembly.py)|
|  Expected risk = P(class) × loss_matrix     |
|  action = argmin expected risk              |
|  Green (MCAR) / Yellow (MAR) / Red (MNAR)  |
+---------------------------------------------+
```

---

## Module Map

### `lacuna/core/` — Foundation types

| File | Contents |
|---|---|
| `types.py` | Immutable dataclasses: `RawDataset`, `TokenBatch`, `ObservedDataset`, `PosteriorResult`, `MoEOutput`, `LacunaOutput`, `ReconstructionResult` |
| `exceptions.py` | `LacunaError` hierarchy: `ValidationError`, `ConfigError`, `RegistryError`, `CheckpointError`, `NumericalError` |
| `validation.py` | Boundary validation: shape, dtype, NaN/Inf, probability, value range |
| `rng.py` | Deterministic RNG via `RNGState`; child streams via `spawn()` |

**Key types:**

`TokenBatch` — the universal batch representation:
```
tokens:              [B, max_rows, max_cols, 4]
row_mask:            [B, max_rows]   bool, padding-aware
col_mask:            [B, max_cols]   bool, padding-aware
generator_ids:       [B]             training labels
class_ids:           [B]             MCAR=0, MAR=1, MNAR=2
original_values:     [B, max_rows, max_cols]
reconstruction_mask: [B, max_rows, max_cols]  artificially masked cells
```

Token dimensions (4 per cell):
```
[0] normalised_value    MAD-robust, clipped to [-3, 3]; 0.0 if missing
[1] is_observed         1.0 = present, 0.0 = missing
[2] mask_type           0.0 = naturally missing, 1.0 = artificially masked
[3] feature_id          j / max_cols  (column positional encoding)
```

---

### `lacuna/config/` — Configuration

| File | Contents |
|---|---|
| `schema.py` | `DataConfig`, `ModelConfig`, `TrainingConfig`, `GeneratorConfig`, `LacunaConfig` |
| `load.py` | `load_config(path)`, `save_config(config, path)` with validation |
| `hashing.py` | Deterministic config hashing for run reproducibility |

Config is YAML-driven. Each run saves its effective config to `{run_dir}/config.yaml`
so evaluation always uses the exact settings the model was trained with.

---

### `lacuna/data/` — Data pipeline

| File | Contents |
|---|---|
| `catalog.py` | Dataset registry with caching; loads UCI / OpenML / sklearn datasets |
| `ingestion.py` | CSV / Parquet / sklearn → `RawDataset` |
| `normalization.py` | Robust MAD-based normalisation |
| `semisynthetic.py` | `SemiSyntheticDataLoader`; applies synthetic missingness to real data. Only data loader path. |
| `tokenization.py` | Row tokenisation; `apply_artificial_masking()` for self-supervised pretraining |
| `batching.py` | `collate_fn` for the PyTorch DataLoader (only contents; pure-synthetic loaders removed) |
| `missingness_features.py` | `MissingnessFeatureExtractor`; computes 16 explicit statistics |

**Semi-synthetic generation:** Each training batch samples a real dataset from the
catalog, selects a generator at random, and calls `generator.apply_to(X)` to compute
the missingness mask from the actual data values. This preserves MAR relationships
that depend on real observed values — a critical correctness property.

**Artificial masking:** During training, 10–15% of observed cells are additionally
masked (mask_type=1.0) to create a self-supervised reconstruction signal. The model
sees two kinds of missingness simultaneously: real (natural) and artificial.

**Missingness features (16 total):**
- Missing rate statistics: mean, variance, range, max across columns (4)
- Point-biserial correlations between missingness indicators and observed values (3)
- Cross-column missingness correlations (3)
- Distributional statistics: skewness and kurtosis of observed values (4)
- Little's MCAR test approximation: χ² statistic and p-value proxy (2)

These explicit features are especially important for MCAR/MAR discrimination,
where the transformer's learned representation alone is insufficient.

---

### `lacuna/generators/` — Parametric missingness generators

| File | Contents |
|---|---|
| `base.py` | `Generator` abstract base class (immutable); `sample()`, `apply_to()` |
| `registry.py` | `GeneratorRegistry`: ID-indexed, validated, class-labelled |
| `priors.py` | `GeneratorPrior`: `uniform()`, `class_balanced()`, `custom()` |
| `families/mcar/` | 22+ MCAR generators |
| `families/mar/` | 37+ MAR generators |
| `families/mnar/` | 46+ MNAR generators |
| `families/registry_builder.py` | `load_registry_from_config(name)` |

**Generator registry configs** (`configs/generators/`):

| Config | Generators | Use |
|---|---|---|
| `lacuna_minimal_6` | 6 (2+2+2) | Fast iteration / testing |
| `lacuna_minimal_18` | 18 (6+6+6) | Development |
| `lacuna_tabular_110` | ~110 | Production training |

The 110-generator registry covers a broad distribution of mechanism subtypes:
MCAR (Bernoulli, block, distributional, multilevel), MAR (logistic, polynomial,
multi-predictor, structural, survey skip-logic), MNAR (self-censoring, threshold,
detection-limit, latent variable, selection bias, social reporting).

---

### `lacuna/models/` — Neural architecture

| File | Contents |
|---|---|
| `encoder.py` | `LacunaEncoder`: token embedding + row-wise transformer + two-stage pooling |
| `reconstruction/base.py` | `BaseReconstructionHead` abstract class; computes both artificial and natural errors |
| `reconstruction/heads.py` | `MCARHead`, `MARHead`, `MNARSelfCensoringHead` |
| `reconstruction/heads_container.py` | `ReconstructionHeads`: manages all three heads; exposes `get_natural_error_tensor()` |
| `moe.py` | `GatingNetwork`, `MixtureOfExperts`, `MoEConfig` |
| `heads.py` | `GeneratorHead`, `ClassHead`: lightweight classification heads |
| `aggregator.py` | Functions: `aggregate_to_class_posterior()`, `compute_entropy()`, `compute_confidence()` |
| `decision.py` | Functions: `bayes_optimal_decision()`, `compute_expected_loss()`, `interpret_decision()` |
| `assembly.py` | `LacunaModel`, `LacunaModelConfig`, `BayesOptimalDecision`, `Decision`; factory functions |

**Encoder internals:**
- `TokenEmbedding`: 4 → hidden_dim (four sub-projections concatenated: value, observed flag, mask type, position)
- `TransformerLayer`: standard multi-head self-attention + feedforward, applied **row-wise** (each row is an independent sequence over its columns)
- `AttentionPooling`: learned attention weights reduce a sequence to a single vector
- Two pooling stages: row pooling (columns → row repr) then dataset pooling (rows → evidence)

Row-wise attention (rather than full-dataset attention) is memory-efficient and
forces the model to learn per-row cross-column patterns — the key signal that
distinguishes MAR (missingness structured by column values) from MCAR (uniform).

**Reconstruction heads:**
Each head produces predictions for artificially masked cells and, separately,
for naturally missing cells. The **natural errors** are what feeds the MoE gating
network — reconstruction quality on real missing data is diagnostic of mechanism type.

- `MCARHead`: Per-token MLP; no cross-column structure (appropriate for independent missingness)
- `MARHead`: Cross-attention over observed cells, using their raw values as keys/values
- `MNARSelfCensoringHead`: Models self-censoring where high or low values trigger missingness

**MoE gating:**
The gating network receives the concatenation of evidence (64), natural reconstruction
errors (3), and missingness features (16) — 83 dimensions total. It outputs soft weights
over the three experts. Expert outputs are aggregated with mean-normalisation per class
to prevent structural bias (one class having more experts than another).

**Bayes-optimal decision:**
`BayesOptimalDecision` takes P(MCAR, MAR, MNAR) and a 3×3 loss matrix and returns
the action (Green / Yellow / Red) that minimises expected loss. This is a thin
wrapper over `decision.py` functions, computed in `assembly.py`.

---

### `lacuna/training/` — Training infrastructure

| File | Contents |
|---|---|
| `trainer.py` | `Trainer`, `TrainerConfig`, `DetailedValResult`; full training loop |
| `loss.py` | `LacunaLoss`; multi-task: mechanism CE + reconstruction MSE + auxiliary |
| `calibration.py` | `find_optimal_temperature()`, `apply_temperature_scaling()` |
| `checkpoint.py` | `CheckpointData`, `save_checkpoint()`, `load_checkpoint()`, `load_model_weights()` |
| `report.py` | `generate_eval_report()`, `save_raw_predictions()`, `print_eval_summary()` |
| `logging.py` | JSONL metric logger |
| `scheduling.py` | `LRScheduler` with linear warmup |
| `early_stopping.py` | `EarlyStopping` with patience and min_delta |

**Loss function:**
```
L_total = mechanism_weight × L_CE  +  reconstruction_weight × L_recon  +  L_aux
```
- `L_CE`: Cross-entropy on class posterior P(MCAR, MAR, MNAR)
- `L_recon`: Mean of per-head MSE on artificially masked cells
- `L_aux`: Load-balancing + optional KL divergence

Default weights: mechanism=1.0, reconstruction=0.5.

**Calibration:**
Post-hoc temperature scaling searches for T that minimises NLL on the validation
set. The temperature is applied to gate logits before the softmax. This is done
after training and does not change model weights.

---

### `lacuna/experiments/` — Experiment tracking

| File | Contents |
|---|---|
| `registry.py` | `RunEntry` dataclass, `RunRegistry` (JSON-backed) |
| `registry_render.py` | Renders `experiments/REGISTRY.md` from registry state |
| `migrate.py` | Backfills registry entries by scanning artifact directories |

`RunEntry` fields: `run_id` (RUN-NNN), `folder_name`, `timestamp`, `config_path`,
`status` (training / evaluated / calibrated), `metrics`, `mnar_variants`, `n_experts`, `tags`.

All pipeline runs are registered automatically. The registry is the authoritative
record of what was run, with what config, and what it achieved.

---

### `scripts/` — Entry points

| Script | Purpose |
|---|---|
| `run_pipeline.py` | **Primary entry point**: train → calibrate → evaluate → figures |
| `train.py` | Training only, with auto eval report |
| `run_ablation.py` | Missingness-feature ablation sweep (7 specs × N seeds) |
| `validate_generators.py` | Validate synthetic generators against Little's MCAR test |
| `evaluate.py` | Evaluation only; `--batches N` controls sample count |
| `calibrate.py` | Post-hoc temperature scaling |
| `generate_run_figures.py` | Full figure suite from a single `eval_report.json` |
| `generate_roc_curves.py` | ROC / PR curves from `predictions.pt` |
| `generate_dissertation_figures.py` | Cross-experiment comparison (requires consolidated JSON) |
| `consolidate_results.py` | Aggregate results across multiple runs |
| `infer.py` | End-user inference on a provided CSV |

---

## Active Configuration (`semisynthetic_full.yaml`)

| Parameter | Value |
|---|---|
| `hidden_dim` | 128 |
| `evidence_dim` | 64 |
| `n_layers` | 4 |
| `n_heads` | 4 |
| `max_cols` | 48 |
| `max_rows` | 128 |
| `batch_size` | 16 |
| `batches_per_epoch` | 200 |
| `val_batches` | 50 |
| `lr` | 1e-4 |
| `warmup_steps` | 400 |
| `patience` | 20 |
| `generator_registry` | `lacuna_tabular_110` |
| `mnar_variants` | `["self_censoring"]` (3 experts total) |
| `train_datasets` | 21 datasets (UCI / sklearn) |
| `val_datasets` | 7 datasets (held-out) |

---

## Current Performance

Best run: RUN-054 (`lacuna_semisyn_20260329_032848`)
Calibrated checkpoint, T ≈ 1.96, 800 eval samples across 7 held-out datasets.

| Metric | Value |
|---|---|
| Overall accuracy | 82.6% |
| MCAR recall | 94.5% |
| MAR recall | 73.6% |
| MNAR recall | 84.5% |
| MCAR precision | 79.4% |
| MAR precision | 91.8% |
| MNAR precision | 78.7% |
| ECE | 0.038 |
| Selective accuracy (τ = 0.9) | 90.9% at 70.3% coverage |

MAR is the hardest class. Nearly all MAR errors (71 / 88) are misclassified as MNAR,
consistent with the theoretical non-identifiability at the MAR/MNAR boundary.
Per-generator MAR accuracy ranges from 0% to 100%, indicating the theoretical
worst case is not uniformly realised in practice.

---

## Key Design Decisions

**Explicit RNG throughout.**
No global seeds. All randomness flows through `RNGState` with `spawn()` for child
streams. Any run is fully reproducible from its seed and config.

**Immutable data structures.**
`TokenBatch`, `RawDataset`, `GeneratorParams`, all generators, and all model outputs
are frozen dataclasses. State is passed explicitly; there is no hidden coupling.

**Semi-synthetic training with `apply_to()`.**
Real data structure + synthetic missingness avoids the need for labelled real-world
datasets. `generator.apply_to(X)` computes the missingness mask from the provided
data values, which is essential for MAR generators whose missingness depends on the
actual observed values.

**Natural errors as the MoE discrimination signal.**
The reconstruction heads compute errors on both artificially masked cells (training
signal) and naturally missing cells (discrimination signal). Only the natural errors
feed the MoE gating network — reconstruction difficulty on real missing data is
diagnostic of mechanism type in a way that artificial errors are not.

**Three complementary input signals to the gating network.**
The transformer evidence, reconstruction natural errors, and explicit missingness
features each capture different aspects of mechanism structure. The explicit features
are critical for MCAR/MAR discrimination where learned representations alone
produce ambiguous gate weights.

**Balanced class aggregation.**
Expert logits are normalised by the number of experts per class before combining.
This prevents structural bias: if MNAR had more experts than MCAR, raw summation
would favour MNAR predictions regardless of the data.

**Symmetric 1/1/1 expert architecture.**
One expert per class. An earlier 1/1/3 design (three MNAR sub-experts) introduced
gradient asymmetry between classes and reduced generalisation. The symmetric design
both simplifies training and improves held-out accuracy.

**Post-hoc calibration.**
Temperature scaling is applied after training, keeping the classification objective
clean during training and making calibration independently adjustable.

---

## Execution

```bash
# Full pipeline (recommended)
python scripts/run_pipeline.py --config configs/training/semisynthetic_full.yaml

# Evaluation only, with high sample count for tight per-generator CIs
python scripts/evaluate.py \
  --checkpoint runs/RUN/checkpoints/calibrated.pt \
  --config runs/RUN/config.yaml \
  --batches 5000 \
  --output runs/RUN/checkpoints/eval_report_highN.json

# Figures for a completed run
python scripts/generate_run_figures.py --run-dir /path/to/run

# Inference on user data
python scripts/infer.py --input data.csv --checkpoint calibrated.pt
```
