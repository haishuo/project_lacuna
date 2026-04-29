# Project Lacuna — Document 2 of 3: Architecture

This document describes the full Lacuna architecture in implementation-faithful detail. It is the second of three companion documents: Document 1 covers motivation and theoretical position; Document 3 covers the experimental history. Where this document references "the model," it refers to Lacuna-Survey v11 except where explicitly noted.

---

## 1. End-to-End Pipeline

At the highest level, Lacuna takes a CSV of tabular data with missing values and produces a probability vector over `[MCAR, MAR, MNAR]` plus a categorical decision (Green/Yellow/Red). Internally the pipeline is:

```
Real dataset (CSV / UCI / OpenML)
   ↓  Catalog loading + MAD-robust normalisation
   ↓  Optional: synthetic missingness applied via a generator (training only)
TokenBatch [B, max_rows, max_cols, 4]
   ↓  Transformer encoder with row-wise attention
Evidence vector [B, 64]
   ↓  In parallel:
   ↓    - 3 reconstruction heads → per-mechanism MSE on naturally-missing cells [B, 3]
   ↓    - 16 explicit missingness features [B, 16]
Concatenated input to gating network [B, 83]
   ↓  Mixture of Experts (one expert per class)
Class logits [B, 3]
   ↓  Post-hoc temperature scaling (T ≈ 1.96)
Posterior P(MCAR), P(MAR), P(MNAR)
   ↓  Bayes-optimal decision rule (asymmetric loss matrix)
Decision: Green / Yellow / Red
```

Each block is described in detail below.

---

## 2. Tokenisation

The tokenisation step is the most architecturally significant design choice in the system. It converts a tabular dataset — a matrix of numeric values with some cells missing — into a structured tensor that the transformer can process while preserving the two-dimensional row-column structure.

### Per-cell Token

Each cell in the dataset is encoded as a 4-dimensional token:

```
token[row, col] = [value, is_observed, mask_type, feature_id_normalised]
```

- **value** — The numeric cell value, normalised per-column using a robust MAD (median absolute deviation) scaling and clipped to `[-3, 3]`. Missing cells are filled with `0.0` (the MAD-normalised mean).
- **is_observed** — `1.0` if the value is present in the original data, `0.0` if it is missing (NaN). This is the primary signal of the missingness pattern.
- **mask_type** — `0.0` for naturally missing cells (from the original dataset), `1.0` for cells that have been *artificially* masked during training to provide a self-supervised reconstruction signal. At inference time, all missingness is natural.
- **feature_id_normalised** — The column index divided by `max_cols`. Provides positional information so the model can track which column a token belongs to. Supplemented by learnable column-position embeddings.

### Tensor Shapes

The resulting tensor has shape `[B, max_rows, max_cols, 4]` where `B` is batch size, `max_rows` is a row-count cap (128 in the active configuration), and `max_cols` is a column-count cap (48). Datasets exceeding these caps are subsampled; smaller datasets are padded. Boolean masks `row_mask: [B, max_rows]` and `col_mask: [B, max_cols]` indicate which positions are real rather than padding.

### Why Row-Level Tokenisation

The transformer attends *across columns within a row*, not across the entire flattened sequence. This preserves the two-dimensional structure of the data. For MAR detection specifically, the model must learn that "missingness in column j correlates with the observed value in column k *within the same row*." Flattening the entire dataset into a sequence of length `n × d` would either lose row identity (losing intra-row relationships) or be computationally infeasible — for a 500×20 dataset, that's a sequence of 10,000 tokens, nested within a batch.

A second alternative considered was treating each column's missingness pattern as a time series across rows and attending across rows per column. This was rejected because MAR detection requires *cross-column* relationships that a per-column scheme cannot directly capture.

---

## 3. Transformer Encoder

The encoder transforms the `[B, max_rows, max_cols, 4]` token tensor into a single `[B, 64]` evidence vector representing the dataset-level signal about the missingness mechanism. It has four stages.

### Stage 1 — Token Embedding

The 4-dimensional raw tokens are projected to `hidden_dim = 128` via a learned linear layer. Learnable embeddings are added for the `is_observed` flag, the `mask_type` flag, and the column position. The combination of normalised feature IDs (in the token) and learnable position embeddings allows the model to both generalise across datasets with different column counts and to specialise on particular column positions within a dataset.

### Stage 2 — Transformer Layers

Four standard transformer encoder layers, each with multi-head self-attention (4 heads) and a feedforward sublayer. Crucially, attention is applied **row by row** — each layer performs self-attention over the `max_cols` tokens *within* each row independently. There is no cross-row attention in the transformer itself; cross-row information is aggregated only in the pooling stages below.

This keeps the attention complexity at `O(max_cols²)` per row rather than `O((max_rows × max_cols)²)`, which would be intractable for the dataset sizes of interest. The `col_mask` is applied to exclude padding columns from attention. A pre-norm architecture (layer norm before attention) is used for training stability.

### Stage 3 — Row Pooling (features → rows)

After the transformer layers, each row is represented by `max_cols` token vectors. These are collapsed to a single per-row vector via **attention-based pooling**: a learned query vector attends over the column tokens, producing a weighted average. This is preferable to mean or max pooling because it lets the model learn which columns are most informative about the mechanism. Output shape: `[B, max_rows, 128]`.

### Stage 4 — Dataset Pooling (rows → dataset)

The `max_rows` row vectors are similarly collapsed to a single dataset-level evidence vector via another attention-pooling step with a learned query, followed by a linear projection to `evidence_dim = 64`. Output shape: `[B, 64]`. This 64-dimensional evidence vector is the compressed representation of the entire dataset's missingness pattern. It feeds the MoE gating network alongside the explicit statistical features and reconstruction errors.

### Encoder Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `hidden_dim` | 128 | Transformer hidden dimension |
| `evidence_dim` | 64 | Output evidence vector size |
| `n_layers` | 4 | Number of transformer layers |
| `n_heads` | 4 | Attention heads per layer |
| `max_cols` | 48 | Maximum columns (positional embedding limit) |
| `max_rows` | 128 | Maximum rows (subsampling cap) |
| `dropout` | 0.1 | Regularisation |
| `row_pooling` | "attention" | Row aggregation strategy |
| `dataset_pooling` | "attention" | Dataset aggregation strategy |

---

## 4. Missingness Feature Extractor

The transformer learns implicit representations of the missingness pattern, but some signals require explicit statistical computation. The missingness feature extractor computes ~16 hand-crafted features that directly target distinguishing characteristics of each mechanism. Diagnostic analysis (Experiment 2) confirmed these features have very high effect sizes — point-biserial correlations separate MCAR from MAR with Cohen's d > 9.0; Little's-test approximation separates MAR from MNAR with d > 2.8.

The feature groups are:

### Missing-rate Statistics (4 features)

- Mean missing rate across columns
- Variance of missing rates across columns — should be near zero under MCAR (uniform), higher under MAR/MNAR
- Range (max − min) of per-column missing rates
- Maximum per-column missing rate

### Point-Biserial Correlations (3 features)

Point-biserial correlation measures the correlation between the binary missingness indicator of one column and the continuous observed values in another column. High correlations indicate MAR (observed values predict missingness).

- Mean correlation across all column pairs
- Maximum correlation (strongest observed-to-missingness relationship)
- Standard deviation of correlations

### Cross-column Missingness Correlations (3 features)

Pearson correlation between missingness indicators of different column pairs. High correlations indicate structured missingness — block patterns in MAR, monotone patterns in MNAR.

- Mean correlation
- Maximum correlation
- Standard deviation

### Distributional Statistics (4 features)

MNAR mechanisms distort the observed value distribution because the probability of observing a value depends on its magnitude.

- Mean skewness of observed-value distributions per column
- Mean kurtosis
- Variance of skewness across columns
- Variance of kurtosis across columns

### Little's MCAR Test Approximation (2 features)

An approximation of Little's (1988) MCAR test statistic, which tests the null hypothesis of MCAR by comparing observed column means across different missingness patterns.

- Chi-squared statistic
- p-value proxy

These ~16 features are concatenated with the evidence vector and the reconstruction errors to form the input to the MoE gating network. They provide a "cheat sheet" of summary statistics that complement the learned evidence vector. The Lacuna-Survey variant uses 10 missingness features specifically (the Little's-test slot was removed — see ADRs 0001 and 0004 in `docs/decisions/`).

---

## 5. Reconstruction Heads

Each Lacuna model includes one reconstruction head per mechanism class. Their primary purpose is twofold: (a) to provide a self-supervised training signal that improves the encoder's representations even when the class label is noisy, and (b) to produce *per-mechanism reconstruction errors on naturally-missing cells* that feed into the MoE gating network as a discriminative signal.

### Head Designs

- **MCARHead** — Per-token MLP, no cross-column structure. Appropriate for independent missingness.
- **MARHead** — Cross-attention over observed cells, using their raw values as keys/values. Models the conditioning of missingness on observed values.
- **MNARSelfCensoringHead** — Models self-censoring in which high or low values trigger missingness.

### Artificial vs Natural Errors

During training, 10–15% of *observed* cells are additionally masked (with `mask_type = 1.0`) and the reconstruction heads must predict those values from the rest of the context. This is inspired by BERT's masked language modelling, adapted for tabular data. The MSE on those artificially-masked cells is what backpropagates as the reconstruction loss.

In addition to artificial masking, each head produces predictions for *naturally* missing cells — the cells that were missing in the source data. These predictions cannot be checked against ground truth, but the head's *uncertainty* on those cells is informative: a head conditioned on the wrong mechanism produces systematically different reconstruction patterns than one conditioned on the right mechanism. The natural reconstruction errors `[B, 3]` (one per head) are concatenated into the gating network's input.

### Why Per-Mechanism Heads

The motivation is direct likelihood-of-data evidence. If the MAR-conditioned reconstruction head fits the observed cells substantially better than the MNAR-conditioned head on a particular dataset, that is evidence (not proof, given Molenberghs) that the mechanism is closer to MAR. The signal is exposed not just to the gating network but also surfaced in the user-facing demo, so an analyst can see the underlying likelihood evidence rather than only the final posterior.

### Reconstruction Evidence and Identifiability

A subtle point that recurs in the experimental record: under MNAR-by-self-censoring or MNAR-by-module-refusal, the unobserved values that drive the missingness are *by construction* absent from the reconstruction target. A model fitting only on observed cells therefore necessarily scores the MAR explanation as well-fitting, because the MAR explanation is consistent with the observed-cell distribution. This is why the natural reconstruction errors alone cannot adjudicate between MAR and MNAR — and it is an empirical instantiation of the Molenberghs identifiability result. The MoE combines reconstruction evidence with explicit pattern features and learned domain priors to produce posteriors that elevate MNAR mass on real MNAR-consensus anchors *without* forcing categorical MNAR labels where the observable likelihood remains MAR-compatible.

---

## 6. Mixture of Experts

The MoE layer takes the combined signal — evidence vector (64-dim), per-mechanism reconstruction errors (3-dim), and missingness features (16-dim, or 10-dim in Lacuna-Survey) — and produces the final class posteriors.

### Architecture

Three symmetric experts, one per class (1 MCAR, 1 MAR, 1 MNAR). Each expert is a small feedforward network that takes the combined input and produces a 3-dimensional class posterior vector `[P(MCAR), P(MAR), P(MNAR)]`.

A **gating network** — a 2-layer MLP with tanh activations — receives the same input and produces 3 logits, one per expert. After temperature-scaled softmax, these become gating probabilities `g = [g₁, g₂, g₃]`. The final output is the gated mixture:

```
output = g₁ × expert₁(x) + g₂ × expert₂(x) + g₃ × expert₃(x)
```

Each expert specialises in recognising its mechanism, while the gating network learns to route inputs appropriately based on the available evidence.

### Why Symmetric 1/1/1

The original design used 5 experts mapped `[MCAR, MAR, MNAR, MNAR, MNAR]` with mean aggregation across experts per class — the intent was to give MNAR more representational capacity, since MNAR has more diverse mechanisms. This failed for a non-obvious reason. Despite mean aggregation correcting the class prior mathematically, the 3 MNAR experts created *asymmetric gradient flow* during backpropagation. Each MCAR and MAR expert received gradients through one output head; each of the three MNAR experts received gradients through one head, but the MNAR *class* received gradient through three heads simultaneously. The gating network learned to route borderline MAR cases to MNAR because doing so reduced loss across three MNAR expert outputs rather than one MAR expert output — a pure artifact of the aggregation geometry.

Switching to 1/1/1 improved MAR recall from 52.6% to 69.3% (+16.7 points) while simultaneously improving overall accuracy from 77.0% to 78.4% — a "free lunch" from fixing the architecture. The 1/1/1 design is therefore an empirically validated constraint, not a default.

### Load Balancing

Without intervention, an MoE gating network can collapse — one expert wins all the routing weight and the others degenerate to unused parameters. Two complementary mechanisms prevent this.

**Switch Transformer loss** (training, in `lacuna/training/loss.py`): for each batch, compute the fraction of samples hard-routed to each expert (`fᵢ`, by argmax) and the average soft probability assigned to each expert (`pᵢ`, by mean). The loss is `n_experts × Σᵢ fᵢ pᵢ`. Uniform routing gives both as `1/n_experts`, with the loss equal to 1; imbalanced routing raises the product. From Fedus et al., 2021.

**KL-divergence loss** (in the MoE layer itself, `lacuna/models/moe.py`): the KL divergence of the mean gating probability distribution from a uniform target.

Both losses are weighted by `load_balance_weight = 0.01` and added to the total training loss. With only three experts, expert collapse is less catastrophic than at large MoE scale, but the regularisation still meaningfully stabilises training.

---

## 7. Post-Hoc Temperature Calibration

After training, the model's softmax outputs are typically poorly calibrated — the model is too confident. Post-hoc temperature scaling (Guo et al. 2017) corrects this by finding a single scalar `T` that, when applied as `softmax(logits / T)`, minimises Expected Calibration Error (ECE) on a held-out validation set.

### Implementation

A two-phase grid search: a coarse log-uniform sweep over `T ∈ [0.1, 10.0]` (100 values), then a fine linear sweep in a window around the coarse minimum (100 values). The calibration script loads a trained checkpoint, runs forward passes on validation data to collect pre-softmax gate logits, runs the grid search to find optimal `T`, patches `model.moe.gating.log_temperature = log(T)`, and saves a new `calibrated.pt` checkpoint.

The optimal value found in production is `T ≈ 1.96`.

### The Unexpected Bonus

Temperature scaling at `T = 1.96` not only improved calibration (ECE: 0.116 → 0.038) but also improved accuracy on all three classes simultaneously, by +4.2 points overall. This is theoretically surprising because temperature scaling does not change the argmax of the softmax in general. What happened: the pre-calibration model was so overconfident on borderline cases that softening the probabilities *did* shift the argmax in a non-trivial fraction of cases — specifically, cases where the model had been incorrectly confident in one direction but the second-place class had been close behind. The implication is that the model at `T = 1.0` was not merely miscalibrated but actively wrong in a consistent direction that temperature scaling corrected.

For Lacuna-Survey, an analogous vector-scaling layer (`(T, bias)`) is fit on real-survey anchors to align the posterior to literature-consensus mechanism readings. This is the only component of the Lacuna-Survey pipeline that touches real-world data; the model itself trains entirely on semi-synthetic data.

---

## 8. Bayes-Optimal Decision Rule

The MoE produces posterior probabilities `p = [P(MCAR), P(MAR), P(MNAR)]`. The decision rule converts these into a recommended action that minimises expected cost under an asymmetric loss matrix.

### Loss Matrix

|  | True MCAR | True MAR | True MNAR |
|--|-----------|----------|-----------|
| **Green (assume MCAR)** | 0.0 | 0.3 | 1.0 |
| **Yellow (assume MAR)** | 0.2 | 0.0 | 0.2 |
| **Red (assume MNAR)** | 1.0 | 0.3 | 0.0 |

The asymmetry encodes domain knowledge:

- Treating MNAR as MCAR (Green when truly Red): risk = 1.0 — biased estimates, invalid inference.
- Treating MCAR as MNAR (Red when truly Green): risk = 1.0 — sensitivity analysis on noise; wasted effort plus false alarms.
- Treating MCAR as MAR (Yellow when truly Green): risk = 0.2 — conservative but unbiased; multiple imputation works fine on MCAR.
- Treating MNAR as MAR (Yellow when truly Red): risk = 0.2 — multiple imputation is insufficient but less harmful than complete-case analysis.

### Decision Computation

For each action `a`, compute the expected risk:

```
risk(a) = Σ_k L[a, k] × P(class_k)
```

Select the action with minimum expected risk. The Bayes-optimal rule means that, for example, if `P(MNAR)` is only moderately elevated, the rule may still recommend Yellow because the asymmetric costs favour conservative caution over confident misclassification.

### Action Interpretation

- **Green** — Assume MCAR. Complete-case analysis or simple imputation acceptable.
- **Yellow** — Assume MAR. Use multiple imputation (MICE, Amelia) or likelihood-based methods (EM, FIML).
- **Red** — Assume MNAR. Use sensitivity analysis, selection models (Heckman), or pattern-mixture models.

---

## 9. Generator System (Training Data)

Because missingness mechanisms are unobservable in real data, the model is trained on **synthetic data with known labels**. The generator system creates diverse, realistic missingness patterns spanning the space of plausible mechanisms.

### The Original 110 Generators

The base Lacuna build (the `lacuna_tabular_110` registry) has 110 parameterised generators: 34 MCAR, 36 MAR, and 38 MNAR. Each generator:

1. Optionally generates base data (Gaussian, exponential, skewed, mixed, correlated) — though in production the base data comes from real datasets via the semi-synthetic pipeline.
2. Applies a mechanism-specific missingness pattern to produce a mask `R ∈ {0,1}^{n×d}`.
3. Returns an `ObservedDataset(data = X * R, mask = R, class_id = class, generator_id = id)`.

Examples of mechanisms covered:

- **MCAR** — `BernoulliMCAR`, `BlocksMCAR`, `RowEffectsMCAR`, `DistributionalMCAR`, `MultilevelMCAR`
- **MAR** — `SimpleMARPredictor`, `ComplexMAR`, `MultipleMAR`, `SurveyMAR` (skip-logic patterns), `WeakMAR` (borderline-MCAR ambiguous cases)
- **MNAR** — `SelfCensoringMNAR`, `CensoringMNAR` (left-censoring at detection limits), `LatentMNAR`, `SelectionMNAR`, `StrategicMNAR`

### The Lacuna-Survey 70-Generator Registry

Lacuna-Survey replaces the generic 110 generators with a curated 70-generator registry covering only mechanisms plausible for self- or interviewer-administered survey questionnaires (12 MCAR, 36 MAR, 22 MNAR):

- **MCAR** — Bernoulli, ColumnMixture, ColumnOrdered, ColumnClustered, RandomBlocks, SubgroupSpecific, **RotatedBooklet** (planned-missing designs as in PISA, NAEP, ECLS-K).
- **MAR** — Logistic, Probit, Threshold, Polynomial, Spline, DemographicGated, PartialResponse, SkipLogic, Branching, SectionLevel, RequiredOptional, QuotaBased, **ModuleSkip** (the MAR counterpart to module-refusal MNAR).
- **MNAR** — SelfCensor variants, Threshold variants, Quantile variants, social-desirability mechanisms (UnderReport, OverReport, NonLinearSocial, Gaming, Volunteer), **ModuleRefusal** (row-aligned battery refusal driven by unobserved value).

The `RotatedBooklet`, `ModuleSkip`, and `ModuleRefusal` generators were added during the Lacuna-Survey iteration arc to close diagnosed feature-space coverage gaps — see Document 3 and ADR 0005.

### Sampling Priors

Generators are sampled at training time via a `GeneratorPrior`:

- **Uniform prior** (production default) — sample uniformly over the registry, accepting slight class imbalance from the count differences (e.g. ~35% MNAR under the 110-generator setup because there are 38 MNAR generators).
- **Class-balanced prior** (abandoned after Experiment 8) — force exactly 1/3 per class. Found to *hurt* performance because it overweights ambiguous MAR generators and drowns the signal in noise.

### Semi-Synthetic Training

For production training, real datasets from UCI, OpenML, and sklearn are used as the base data distribution. The generator's missingness pattern is then *synthetically imposed* on top via `generator.apply_to(X)`, which computes the missingness mask from the actual data values. This is essential for MAR generators whose missingness depends on observed values: synthesising X from a parametric distribution and then applying a MAR mask on top would create artificial structure that does not generalise to real data.

For base Lacuna, 24 training and 7 validation datasets from UCI/OpenML cover biological, social, economic, and medical domains. For Lacuna-Survey, 12 `survey_*` X-bases (10 train + 2 val) provide survey-domain covariate distributions specifically. **No cross-domain (medical, sensor, financial) data enters the main Lacuna-Survey classifier.**

---

## 10. Training Pipeline

### Multi-Task Loss

The total training loss combines three terms:

```
total_loss = mechanism_weight × mechanism_loss
           + reconstruction_weight × reconstruction_loss
           + load_balance_weight × moe_load_balance_loss
```

- **mechanism_loss** — Cross-entropy between the MoE's class posteriors and the true class label. (Brier score was tried in Experiment 7 and found inferior — see Document 3.)
- **reconstruction_loss** — Mean squared error between each expert's reconstruction predictions and the true values at artificially-masked positions.
- **moe_load_balance_loss** — Switch-Transformer + KL terms preventing expert collapse.

Default weights: `mechanism = 1.0`, `reconstruction = 0.1` to `0.5` depending on phase, `load_balance = 0.01`.

### Training Configuration

| Hyperparameter | Production Value |
|----------------|-----------------|
| Epochs | 100 |
| Batch size | 16 |
| Batches per epoch | 100–200 |
| Validation batches | 50 |
| Learning rate | 1e-4 to 3e-4 |
| Warmup steps | 200–400 |
| Early stopping patience | 10–20 |
| LR schedule | Cosine decay |
| Optimiser | AdamW |
| Gradient clipping | Norm-based |
| Hardware | NVIDIA RTX 5070 Ti |
| Training time | ~28 minutes per run |

### Trainer Features

- Three training modes: `pretraining` (reconstruction only), `classification`, `joint`.
- Cosine / linear / constant LR schedules with linear warmup.
- Early stopping on validation loss or accuracy.
- Best-model checkpointing plus periodic snapshots.
- Optional mixed-precision (AMP) for GPU acceleration.
- Each epoch regenerates fresh batches from generators, preventing overfitting to fixed training sets.

---

## 11. Module Map (Implementation Reference)

| Module | Contents |
|--------|----------|
| `lacuna/core/types.py` | Immutable dataclasses: `RawDataset`, `TokenBatch`, `ObservedDataset`, `PosteriorResult`, `MoEOutput`, `LacunaOutput`, `ReconstructionResult`, `Decision` |
| `lacuna/core/exceptions.py` | `LacunaError` hierarchy: `ValidationError`, `ConfigError`, `RegistryError`, `CheckpointError`, `NumericalError` |
| `lacuna/core/rng.py` | `RNGState` with `spawn()` for child streams; no global seeds anywhere |
| `lacuna/data/catalog.py` | Dataset registry, caches UCI/OpenML/sklearn loads |
| `lacuna/data/normalization.py` | Robust MAD normalisation |
| `lacuna/data/semisynthetic.py` | Sole training data loader path; applies synthetic missingness to real data |
| `lacuna/data/tokenization.py` | Row tokenisation; `apply_artificial_masking()` |
| `lacuna/data/missingness_features.py` | 16-feature extractor |
| `lacuna/generators/base.py` | `Generator` abstract base |
| `lacuna/generators/registry.py` | `GeneratorRegistry`, ID-indexed and validated |
| `lacuna/generators/priors.py` | `uniform()`, `class_balanced()`, `custom()` |
| `lacuna/generators/families/` | All MCAR/MAR/MNAR generator families |
| `lacuna/models/encoder.py` | `LacunaEncoder` |
| `lacuna/models/reconstruction/` | `MCARHead`, `MARHead`, `MNARSelfCensoringHead` |
| `lacuna/models/moe.py` | `GatingNetwork`, `MixtureOfExperts`, `MoEConfig` |
| `lacuna/models/aggregator.py` | `aggregate_to_class_posterior`, `compute_entropy`, `compute_confidence` |
| `lacuna/models/decision.py` | `bayes_optimal_decision`, `compute_expected_loss` |
| `lacuna/models/assembly.py` | `LacunaModel` — top-level assembly |
| `lacuna/training/trainer.py` | `Trainer`, `TrainerConfig` |
| `lacuna/training/loss.py` | `LacunaLoss` |
| `lacuna/training/calibration.py` | `find_optimal_temperature`, `apply_temperature_scaling` |
| `lacuna/training/checkpoint.py` | Versioned save/load |
| `lacuna_survey/anchors.py` | Real-survey anchor definitions and citations |
| `lacuna_survey/diagnostic.py` | Real-anchor validation harness |
| `lacuna_survey/probabilistic_diagnostic.py` | Per-anchor posteriors + per-mechanism reconstruction errors |
| `lacuna_survey/mnar_validation.py` | Synth-mechanism-on-real-X harness |
| `lacuna_survey/ood.py` | Out-of-distribution detector (internal validation only — removed from demo) |
| `scripts/run_pipeline.py` | Primary entry point: train → calibrate → evaluate → figures |
| `scripts/infer.py` | End-user CSV inference |
| `demo/app.py` | Streamlit demo |

---

## 12. Model Size and Performance

| Component | Parameters |
|-----------|-----------|
| Transformer encoder | ~580,000 |
| MoE (3 experts + gating) | ~180,000 |
| Reconstruction heads (3) | ~100,000 |
| Missingness-feature MLP | ~40,000 |
| **Total** | **~901,000** |

For headline performance numbers — base Lacuna's 82.6% synthetic accuracy and Lacuna-Survey's posterior elevation on NHANES anchors — see Document 3.

---

## 13. Reproducibility

- **Explicit RNG throughout.** No global seeds. All randomness flows through `RNGState` with `spawn()` for child streams. Any run is fully reproducible from its seed and config.
- **Immutable data structures.** `TokenBatch`, `RawDataset`, generator parameter containers, and all model outputs are frozen dataclasses. State is passed explicitly; there is no hidden coupling.
- **YAML-driven configuration.** Every run saves its effective config to `{run_dir}/config.yaml`, so evaluation always uses the exact settings the model was trained with.
- **Test suite.** 1,096 unit tests covering generators, encoder, MoE, training loops, calibration, and end-to-end inference. The full suite runs in ~47 seconds on CPU.

For the full development trail — the experimental record, the ablations, the negative results, the Lacuna-Survey iteration arc — see Document 3.
