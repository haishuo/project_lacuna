# Lacuna Experiment Journal

Research log for Project Lacuna: a transformer-based classifier for missing data mechanisms (MCAR / MAR / MNAR).

**Objective:** Given a dataset with missing values, classify the generating mechanism to guide downstream statistical analysis (complete-case, multiple imputation, or sensitivity analysis).

**Architecture:** Tokenized tabular data → Transformer encoder → evidence vector → Mixture of Experts (5 experts mapped to 3 classes via mean aggregation) → Bayes-optimal decision rule.

---

## Experiment 1 — Core Architecture Validation

**Date:** 2026-01-08 to 2026-01-10
**Commits:** `b406312` → `a3ef302` (26 commits)

### Motivation
Build and validate the entire pipeline: MNAR mechanisms, tokenization, encoder, reconstruction heads, MoE gating, and training loop. Establish that the model can learn *something* on synthetic data before scaling.

### Changes
- Implemented 6 base generators (2 MCAR, 2 MAR, 2 MNAR) with synthetic data
- Row-level tokenization: each cell → [value, observed_indicator, mask_indicator]
- Transformer encoder with attention-based row and dataset pooling
- Reconstruction heads (5 heads: MCAR, MAR, self_censoring, threshold, latent) for self-supervised pretraining
- MoE with 5 experts → 3 classes via `expert_to_class = [0, 1, 2, 2, 2]`
- Mean class aggregation (divide by experts_per_class, renormalize)
- Bayes-optimal decision rule with asymmetric loss matrix
- Missingness feature extractor (16 statistical features: missing rate variance, cross-column correlations, Little's test approximation)
- Full test suite passing (types, encoder, MoE, assembly, loss, trainer, checkpoint, batching, MNAR, pipeline, semisynthetic)

### Configuration
- `configs/training/minimal.yaml`: 5 epochs, hidden=64, batch=8, 6 generators
- `configs/training/fast_gpu.yaml`: 20 epochs, hidden=128, batch=64, 6 generators

### Results
- All unit and integration tests passing
- Model trains and converges on synthetic data with 6 generators
- No quantitative eval metrics recorded (diagnostic phase)

### Interpretation
Architecture is sound. Ready to scale generator diversity and move to semi-synthetic (real datasets + synthetic missingness).

---

## Experiment 2 — Diagnostic Analysis

**Date:** 2026-01-10
**Commits:** `5d604fd` → `4149935`

### Motivation
Before scaling generators, verify that the discrimination signals are present in the data: reconstruction errors should differ by mechanism, missingness features should separate MCAR from MAR.

### Changes
- Added diagnostic scripts for reconstruction error analysis
- Enhanced MAR generator configurations for stronger signal patterns
- Verified missingness feature discriminability

### Results
- Missingness features show strong MCAR vs MAR separation (d > 9.0 in effect size)
- Little's test approximation separates MAR vs MNAR (d > 2.8)
- Reconstruction errors differ by mechanism as expected

### Interpretation
The input signals to the MoE are informative. If the model can't classify well, the issue is in the gating network or training, not the features.

---

## Experiment 3 — Generator Expansion (6 → 110)

**Date:** 2026-02-13
**Commits:** `bdbd727` → `3911479`

### Motivation
6 generators is too few for a dissertation. Real-world missingness is diverse. Expand to cover the space of plausible mechanisms with parameterized generators.

### Hypothesis
More generators → more diverse training signal → better generalization to unseen missingness patterns.

### Changes
- Refactored from 6 hardcoded generators to 110 parameterized generators
- Organized into `mcar/` (34), `mar/` (36), `mnar/` (38) subdirectories with YAML configs
- YAML-driven registry builder (`lacuna_tabular_110`, `lacuna_minimal_6`, `lacuna_minimal_18`)
- Each generator config specifies class, variant, parameter ranges
- Added fingerprint test script (train on one MNAR variant, test on another)

### Configuration
- `configs/generators/lacuna_tabular_110.yaml`: Full 110-generator config
- 34 MCAR + 36 MAR + 38 MNAR generators

### Controlled Variables
- Model architecture unchanged (hidden=128, layers=4, heads=4)
- Same tokenization and feature extraction

### Results
- 73 test failures from API changes (fixed in follow-up commits)
- Generator registry loads and produces valid training data
- No training metrics yet (infrastructure commit)

### Interpretation
Generator diversity dramatically increases. Class distribution is slightly unbalanced: MNAR has 38/110 generators vs 34 MCAR and 36 MAR. Under uniform sampling, this gives ~35% MNAR vs ~31% MCAR and ~33% MAR.

---

## Experiment 4 — Test Stabilization & Training Robustness

**Date:** 2026-02-13
**Commits:** `d0de8e1` → `7c0a7a4`

### Motivation
Fix all test failures from the generator expansion, then get semi-synthetic training working reliably on the GPU server.

### Changes
1. Fixed 73 test failures (API migrations, import updates)
2. Wired training scripts to YAML-driven generator configs
3. Bumped minimum `d_range` from 3 to 5 (fixes `MARThreePredictor requires d >= 4` crash)
4. Attempted MoE class_aggregation change from "mean" to "sum" (reverted — mean is correct for semi-synthetic)
5. Fixed semisynthetic overfitting: vary data each epoch instead of reusing same batches
6. Handle generator-dataset dimension incompatibility in semisynthetic loader

### Key Bug: MoE Aggregation
- Switching to "sum" aggregation gave MNAR ~60% of probability mass regardless of input (3 experts summed vs 1 each for MCAR/MAR)
- "Mean" aggregation (divide by experts_per_class) is correct: gives each class equal prior weight
- Reverted within 3 minutes after seeing 0% MNAR accuracy flip to 0% MCAR/MAR accuracy

### Results
- All tests passing
- Training runs successfully on GPU (CUDA 12.x, 5070Ti)
- Semi-synthetic data loader handles dimension mismatches gracefully

---

## Experiment 5 — Semi-Synthetic Baseline (110 Generators)

**Date:** 2026-02-13
**Commit:** Training run on Forge (no dedicated commit — uses code from `7c0a7a4`)

### Motivation
First full training run with 110 generators on real datasets. Establish baseline metrics.

### Hypothesis
110 diverse generators + real datasets should give reasonable mechanism classification. MAR/MNAR confusion is expected to be the hardest boundary.

### Configuration
- Config: `configs/training/semisynthetic_full.yaml`
- Generators: `lacuna_tabular_110` (34 MCAR, 36 MAR, 38 MNAR)
- Prior: `GeneratorPrior.uniform(registry)` — uniform over generators
- Loss: Cross-entropy
- Label smoothing: 0.0
- 24 training datasets, 7 validation datasets
- hidden=128, layers=4, heads=4, epochs=100, batch=16, lr=0.0003

### Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | 77.0% (eval) / 83.1% (best val during training) |
| **MCAR recall** | 93.4% |
| **MCAR precision** | 70.3% |
| **MAR recall** | 52.6% |
| **MAR precision** | 94.3% |
| **MNAR recall** | 85.3% |
| **MNAR precision** | 72.3% |
| **ECE** | 0.1338 |
| **Mean confidence (correct)** | 0.865 |
| **Mean confidence (incorrect)** | 0.818 |

**Confusion matrix** (rows = true, cols = predicted):

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 255 | 2 | 16 |
| **True MAR** | 3 | 132 | 116 |
| **True MNAR** | 105 | 6 | 645 |

**Mean predicted probabilities by true class:**

| True Class | P(MCAR) | P(MAR) | P(MNAR) |
|------------|---------|--------|---------|
| MCAR | 0.775 | 0.028 | 0.197 |
| MAR | 0.029 | 0.522 | 0.449 |
| MNAR | 0.173 | 0.017 | 0.810 |

### Interpretation

**MAR is severely underdetected.** Of 251 true MAR samples, 116 (46.2%) are misclassified as MNAR. The model's mean probability on true MAR is [0.029, 0.522, 0.449] — nearly a coin flip between MAR and MNAR, and it defaults to MNAR.

**Root causes identified:**
1. **Class imbalance in sampling:** `GeneratorPrior.uniform()` samples uniformly over 110 generators, but MNAR has 38 generators → ~35% of training data, while MAR has 36 → ~33%. Under uncertainty, the model learned "when in doubt, guess MNAR."
2. **Overconfidence:** ECE = 0.1338. Mean confidence on incorrect predictions is 0.818. The model *thinks* it knows when it doesn't. Cross-entropy loss pushes probabilities toward 0/1.
3. **No per-class weighting:** Hard MAR examples get the same loss weight as easy MCAR.

### Next Decision
Calibrate first, rebalance second, architecture changes only if those fail. "We don't want to open the hood and start ripping out wires unless we're *sure* things are broken."

---

## Experiment 6 — Evaluation Tooling

**Date:** 2026-02-13
**Commits:** `58b2a20`, `e1f5113`

### Motivation
Can't improve what you can't measure. The training script only reported argmax accuracy — no confusion matrix, no calibration metrics, no per-class breakdown.

### Changes
1. Added `DetailedValResult` dataclass and `validate_detailed()` method to Trainer
2. Created `lacuna/training/report.py` with comprehensive reporting:
   - Confusion matrix, precision/recall/F1 per class
   - Expected Calibration Error (ECE) with bin-level data
   - Confidence analysis (correct vs incorrect, by confidence bucket)
   - Probability distribution analysis per true class
   - Entropy statistics per class
   - Selective accuracy (accuracy vs coverage at confidence thresholds)
   - Per-generator accuracy breakdown
3. Rewrote `scripts/evaluate.py` from 2-line stub to full evaluation script
4. Added `--quiet` and `--report` flags to training script
5. Deleted unused script stubs (`generate_data.py`, `eval_fingerprint_test.py`)
6. All predictions saved as `.pt` files for downstream analysis

### Results
Not a training experiment — infrastructure improvement. Metrics from Experiment 5 were produced using this tooling.

---

## Experiment 7 — Calibration & Rebalancing

**Date:** 2026-02-13
**Commit:** `e6aaaf6`

### Motivation
Address the two compounding problems from Experiment 5:
1. MAR underdetection (52.6% recall) from class-imbalanced sampling
2. Overconfidence (ECE 0.1338) from cross-entropy loss

### Hypothesis
- Balanced class prior eliminates the "default to MNAR" bias
- Brier score loss penalizes overconfidence quadratically
- Post-hoc temperature scaling fixes residual miscalibration

### Changes

**7a. Class-balanced prior + label smoothing**
- `GeneratorPrior.uniform(registry)` → `GeneratorPrior.class_balanced(registry)` in train and eval scripts
- Gives exactly 1/3 per class instead of ~35% MNAR
- Added `label_smoothing=0.1` to soften targets

**7b. Brier score loss**
- Switched from cross-entropy to Brier score: `BS = (1/K) * Σ(p_k - y_k)²`
- Fixed bug: `TrainerConfig.get_loss_config()` did not forward `mechanism_loss_type` to `LossConfig`, making Brier score unreachable
- Brier score is a proper scoring rule that rewards honest probabilities

**7c. Post-hoc temperature scaling (Guo et al. 2017)**
- New `lacuna/training/calibration.py`:
  - `collect_gate_logits()` — runs forward passes, collects pre-softmax gate logits
  - `logits_to_class_probs()` — replicates MoE mean aggregation at arbitrary temperature
  - `find_optimal_temperature()` — two-phase grid search (log-uniform coarse + linear fine)
  - `apply_temperature_scaling()` — patches `model.moe.gating.log_temperature`
- New `scripts/calibrate.py` CLI:
  - Loads checkpoint, collects logits, optimizes T, saves calibrated checkpoint + JSON info

### Configuration
Same as Experiment 5 except:
- Prior: `GeneratorPrior.class_balanced(registry)` (was `.uniform()`)
- Loss: Brier score (was cross-entropy)
- Label smoothing: 0.1 (was 0.0)
- Temperature scaling applied post-hoc

### Controlled Variables
- Same model architecture, same datasets, same training hyperparameters
- Same generator set (lacuna_tabular_110)

### Results

**Run:** `lacuna_semisyn_20260214_012227` on Forge (5070Ti, CUDA 12.x)
**Training time:** 730.9s (12.2 min), best val_loss=0.0845 at epoch 4

| Metric | Exp 5 (baseline) | Exp 7 (this) | Δ |
|--------|-----------------|-------------|---|
| **Overall accuracy** | 77.0% | **54.8%** | **−22.2%** |
| **MCAR recall** | 93.4% | 65.1% | −28.3% |
| **MAR recall** | 52.6% | **30.7%** | **−21.9%** |
| **MNAR recall** | 85.3% | 83.1% | −2.2% |
| **MAR precision** | 94.3% | 97.2% | +2.9% |
| **ECE** | 0.1338 | **0.2867** | **+0.153** |
| **Confidence (incorrect)** | 0.818 | 0.789 | −0.029 |

**Confusion matrix** (rows = true, cols = predicted):

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 177 | 0 | 95 |
| **True MAR** | 0 | 104 | 235 |
| **True MNAR** | 29 | 3 | 157 |

**Mean predicted probabilities by true class:**

| True Class | P(MCAR) | P(MAR) | P(MNAR) |
|------------|---------|--------|---------|
| MCAR | 0.643 | 0.012 | 0.345 |
| MAR | 0.019 | 0.406 | 0.575 |
| MNAR | 0.158 | 0.107 | 0.735 |

**Selective accuracy:** 90% accuracy only reached at τ=0.95 (coverage=16.5%)

**Note on generator counts:** Forge reports `{0: 32, 1: 36, 2: 42}` — MNAR has 42 generators at runtime (not 38 as in local registry), likely due to dimension-dependent filtering creating different effective sets.

### Interpretation

**This experiment failed.** All three targets missed badly. Every metric regressed except MAR precision (trivially — fewer MAR predictions made, so the few that survive are more confident).

**Root cause analysis — too many variables changed at once:**
1. **Brier score loss is likely the culprit.** Cross-entropy has infinitely sharp gradient as p→0, which drives the model hard toward correct answers. Brier's quadratic penalty is gentler — with 110 generators and limited training data, the weaker gradient may not push the model hard enough to learn the MAR/MNAR boundary.
2. **Label smoothing compounded the problem.** Smoothing softens targets (1.0→0.9, 0.0→0.033), and Brier already penalizes less than CE. Together they made the loss landscape too flat for the model to find sharp class boundaries.
3. **Class-balanced prior may actually help**, but its effect is masked by the Brier score regression. Need to isolate this variable.
4. **ECE doubled (0.13→0.29)** despite Brier being a "proper scoring rule" — the model is *less* calibrated, not more, because it never learned to classify well in the first place.

**Key lesson:** Change one variable at a time. Always have a control.

### Next Decision
Revert to cross-entropy loss with no label smoothing. Keep class-balanced prior as the single intervention. This isolates the prior effect from the loss function change.

---

## Experiment 8 — Class-Balanced Prior Only (Isolating Variables)

**Date:** 2026-02-14
**Commit:** *(pending)*

### Motivation
Experiment 7 changed three things at once (loss function, label smoothing, class prior) and everything got worse. We need to isolate: does class-balanced prior help when everything else stays constant?

### Hypothesis
Class-balanced prior alone (1/3 per class instead of ~38% MNAR under uniform) will reduce MAR→MNAR misclassification without harming overall accuracy.

### Changes
- Keep `GeneratorPrior.class_balanced(registry)` (from Exp 7)
- Revert to cross-entropy loss (from Exp 5)
- Remove label smoothing (back to 0.0)
- No temperature scaling applied

### Configuration
Same as Experiment 5 except:
- Prior: `GeneratorPrior.class_balanced(registry)` (was `.uniform()`)

### Controlled Variables
- Loss: cross-entropy (same as Exp 5)
- Label smoothing: 0.0 (same as Exp 5)
- Same model architecture, same datasets, same hyperparameters
- Same generator set (lacuna_tabular_110)

### Targets
| Metric | Exp 5 (CE, uniform) | Exp 7 (Brier, balanced) | Target |
|--------|--------------------|-----------------------|--------|
| Overall accuracy | 77.0% | 54.8% | >77% |
| MAR recall | 52.6% | 30.7% | >55% |
| ECE | 0.1338 | 0.2867 | <0.15 |

### Evaluation Workflow
```bash
# Train
python scripts/train_semisynthetic.py \
  --config configs/training/semisynthetic_full.yaml \
  --generators lacuna_tabular_110 --device cuda --quiet --report
```

### Results
**⏳ PENDING — awaiting training run on Forge**

### Next Decision
- If accuracy ≥ Exp 5 and MAR recall improves: class-balanced prior is beneficial, keep it
- If accuracy regresses: revert prior, investigate MAR/MNAR confusion from architecture angle
- Either way: attempt post-hoc temperature scaling on the best model

---

## Planned Experiments

### Experiment 9 — Post-Hoc Temperature Scaling
**Trigger:** After Experiment 8 establishes best pre-calibration baseline
**Change:** Apply `scripts/calibrate.py` to best checkpoint
**Target:** ECE < 0.05 without accuracy loss

### Experiment 10 — Per-Class Loss Weights (Conditional)
**Trigger:** MAR recall < 65% after Experiments 8-9
**Change:** Add `class_weights=[1.0, 1.5, 1.0]` to cross-entropy — upweight MAR 50%
**Target:** MAR recall > 65%, overall accuracy > 77%

### Experiment 11 — Dataset Expansion
**Change:** Increase `max_cols` from 48 to 64, add more datasets from OpenML
**Hypothesis:** More diverse training data improves generalization

### Experiment 12 — Architecture Exploration
**Trigger:** Rebalancing fails to solve MAR/MNAR confusion
**Options:**
- Explicit cross-column dependency module
- Deeper gating network
- Learnable class aggregation (instead of mean)
- Per-column attention in evidence extraction

---

## Appendix A: Generator Distribution

**Local (from YAML config):**

| Class | Count | % (uniform) | % (balanced) |
|-------|-------|-------------|-------------|
| MCAR | 34 | 30.9% | 33.3% |
| MAR | 36 | 32.7% | 33.3% |
| MNAR | 38 | 34.5% | 33.3% |

**Forge runtime (after dimension filtering):**

| Class | Count | % (uniform) | % (balanced) |
|-------|-------|-------------|-------------|
| MCAR | 32 | 29.1% | 33.3% |
| MAR | 36 | 32.7% | 33.3% |
| MNAR | 42 | 38.2% | 33.3% |

*Note: The Forge runtime shows 110 total generators but different per-class counts ({0: 32, 1: 36, 2: 42}) than the local YAML (34/36/38). This may be due to dimension-dependent generator filtering or config differences. The effective MNAR bias under uniform sampling is 38.2%, making class-balanced prior even more important.*

## Appendix B: Model Architecture Summary

```
LacunaModel
├── Encoder (Transformer)
│   ├── Token embedding: [value, observed, mask] → hidden_dim
│   ├── 4 transformer layers (128-dim, 4 heads)
│   ├── Row pooling (attention) → per-row vectors
│   └── Dataset pooling (attention) → evidence vector [64-dim]
├── Reconstruction Heads (5 heads)
│   ├── MCAR, MAR, self_censoring, threshold, latent
│   └── Each: evidence → hidden → predicted values
├── Missingness Feature Extractor
│   └── 16 statistical features from missingness patterns
├── Mixture of Experts
│   ├── Gating Network: [evidence, recon_errors, miss_features] → 5 logits
│   ├── Temperature-scaled softmax → 5 expert probs
│   ├── Mean class aggregation: 5 experts → 3 classes
│   └── expert_to_class = [MCAR, MAR, MNAR, MNAR, MNAR]
└── Bayes Decision Rule
    └── Asymmetric loss matrix → Green/Yellow/Red action
```

## Appendix C: Evaluation Metrics Glossary

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Accuracy** | % correct predictions | Basic performance |
| **Recall** | TP / (TP + FN) per class | Does the model *find* MAR when it's there? |
| **Precision** | TP / (TP + FP) per class | When it says MAR, is it right? |
| **ECE** | Expected Calibration Error | Do predicted probabilities match observed frequencies? |
| **Brier Score** | (1/K)Σ(p_k - y_k)² | Proper scoring rule; rewards honest probabilities |
| **Selective Accuracy** | Accuracy on samples above confidence threshold τ | Can we trust high-confidence predictions? |
| **Coverage** | Fraction of samples above threshold τ | How many samples can we make decisions on? |
| **Entropy** | -Σ p_k log(p_k) | How uncertain is the model? |
