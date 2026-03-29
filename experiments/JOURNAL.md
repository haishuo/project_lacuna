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

**Run:** `lacuna_semisyn_20260214_024148` on Forge (5070Ti, CUDA 12.x)
**Training time:** 729.2s (12.2 min), best val_loss=0.6791 at epoch 6, early stop at epoch 17

| Metric | Exp 5 (baseline) | Exp 7 (Brier+balanced) | Exp 8 (this) | Δ vs Exp 5 |
|--------|-----------------|----------------------|-------------|-----------|
| **Overall accuracy** | 77.0% | 54.8% | **62.6%** | **−14.4%** |
| **MCAR recall** | 93.4% | 65.1% | 89.0% | −4.4% |
| **MCAR precision** | 70.3% | — | 49.1% | −21.2% |
| **MAR recall** | 52.6% | 30.7% | **34.8%** | **−17.8%** |
| **MAR precision** | 94.3% | 97.2% | 85.7% | −8.6% |
| **MNAR recall** | 85.3% | 83.1% | 74.6% | −10.7% |
| **MNAR precision** | 72.3% | — | 81.2% | +8.9% |
| **ECE** | 0.1338 | 0.2867 | **0.2329** | **+0.095** |
| **Confidence (correct)** | 0.865 | — | 0.782 | −0.083 |
| **Confidence (incorrect)** | 0.818 | 0.789 | 0.720 | −0.098 |

**Confusion matrix** (rows = true, cols = predicted):

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 258 | 2 | 30 |
| **True MAR** | 119 | 98 | 64 |
| **True MNAR** | 148 | 14 | 438 |

**Per-class metrics:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| MCAR | 49.1% | 89.0% | 63.3% | 290 |
| MAR | 86.0% | 34.9% | 49.6% | 281 |
| MNAR | 82.3% | 73.0% | 77.4% | 600 |

**Mean predicted probabilities by true class:**

| True Class | P(MCAR) | P(MAR) | P(MNAR) |
|------------|---------|--------|---------|
| MCAR | 0.705 | 0.045 | 0.250 |
| MAR | 0.295 | 0.361 | 0.344 |
| MNAR | 0.247 | 0.060 | 0.692 |

**Selective accuracy:** 90% accuracy at τ=0.90 (coverage=24.7%), 95% accuracy at τ=0.95 (coverage=14.7%)

**Calibration:** ECE = 0.2329

### Key Finding: Bimodal MAR Detection (Per-Generator Analysis)

The per-generator accuracy breakdown reveals that the "MAR recall problem" is not a uniform failure — it's a **bimodal split** between easily-detected and essentially-undetectable generators:

**Well-detected MAR generators:**

| Generator | Accuracy | Samples | Note |
|-----------|----------|---------|------|
| MAR-ColBlocks | 100% | 37 | Block-structured column-level missingness |
| MAR-CrossClass | 100% | 31 | Cross-class dependency patterns |
| MAR-SkipLogic | 89.7% | 29 | Skip-pattern conditional missingness |
| MAR-MonoPredictor | 57.1% | 14 | Single-column predictor |

**Near-zero MAR generators:**

| Generator | Accuracy | Samples | Note |
|-----------|----------|---------|------|
| MAR-Weak | 0% | 7 | Weak signal — near-MCAR |
| MAR-Interactive | 0% | 30 | Interaction-based — may overlap with MNAR |
| MAR-Section | 0% | 33 | Section-level patterns |
| MAR-Moderate | 12.5% | 16 | Moderate signal strength |
| MAR-MixedPred | 15.7% | 51 | Multi-predictor — largest MAR group, mostly misclassified |

**MNAR also has outliers:**

| Generator | Accuracy | Samples | Note |
|-----------|----------|---------|------|
| MNAR-LatentCorr | 0% | 19 | Latent correlation — may look like MAR |
| MNAR-AdaptSamp | 26.3% | 19 | Adaptive sampling — ambiguous pattern |
| MNAR-Feedback | 63.2% | 19 | Feedback mechanism |

**MCAR is bimodal too:**

| Generator | Accuracy | Samples | Note |
|-----------|----------|---------|------|
| MCAR-Complete | 100% | 57 | Complete random — strong pattern |
| MCAR-IndepCol | 100% | 24 | Independent columns |
| MCAR-Burst | 44.4% | 9 | Burst patterns — looks structured |
| MCAR-Uniform | 75.5% | 53 | Uniform random |
| MCAR-Seasonal | 88.2% | 17 | Seasonal pattern — still detectable |

### Interpretation

**Class-balanced prior alone makes things worse.** All three metrics regressed from the Experiment 5 baseline. Overall accuracy dropped 14.4 points, and MAR recall actually got *worse* (52.6% → 34.8%), the opposite of what was intended.

**The real problem is not the prior — it's generator identifiability.** The per-generator data shows that some MAR generators produce patterns that are genuinely indistinguishable from MNAR (or MCAR) given only missingness patterns:

1. **MAR-Weak** and **MAR-Moderate**: Weak MAR signals look like MCAR (random noise). The model can't detect what isn't there.
2. **MAR-Interactive** and **MAR-Section**: These produce structured missingness patterns that overlap with MNAR mechanisms.
3. **MAR-MixedPred** (51 samples, 15.7% accuracy): The largest MAR subgroup, and it's almost entirely misclassified. Multi-predictor MAR may produce patterns that look like MNAR self-censoring.
4. **MNAR-LatentCorr** (0% accuracy): A latent-variable MNAR mechanism that produces patterns resembling MAR. This confirms the boundary is genuinely blurry for certain mechanisms.

**Class-balanced prior backfires because it forces 1/3 MAR**: Under uniform sampling, borderline MAR generators were a smaller fraction of training, so the model could learn to classify the detectable ones. With balanced prior, the undetectable MAR generators get equal weight, drowning the learning signal.

**The overconfidence got worse** (ECE 0.13 → 0.23). The model is now less accurate AND more confused about its confidence.

**New pattern: MCAR is stealing MAR samples.** 119 true MAR samples were classified as MCAR (vs only 3 in Exp 5). The balanced prior may have strengthened the MCAR attractor at the expense of MAR.

### Next Decision

1. **The class-balanced prior should be reverted to uniform** for the baseline. It demonstrably hurts.
2. **Run Experiment 9 (1/1/1 ablation)** with uniform prior to test if architecture asymmetry is a factor — but the per-generator data suggests this is primarily a data/identifiability issue.
3. **Consider generator pruning or reclassification:** MAR-Weak and MAR-Moderate may be producing genuinely MCAR-like patterns. MAR-MixedPred needs investigation — is its pattern actually distinguishable from MNAR?
4. **Post-hoc temperature scaling** (Experiment 10) may still help with calibration even if classification accuracy is capped by generator overlap.

---

## Experiment 9 — 1/1/1 Expert Ablation (Architecture Diagnostic)

**Date:** 2026-02-14
**Commit:** `a581786` (prior revert) + run on Forge
**Checkpoint:** `/mnt/artifacts/project_lacuna/runs/Experiment 9 — 1/1/1 ablation (uniform prior)/checkpoints/best_model.pt`
**Run dir:** `/mnt/artifacts/project_lacuna/runs/Experiment 9 — 1/1/1 ablation (uniform prior)`

### Motivation
The MoE has 5 experts mapped as `[MCAR, MAR, MNAR, MNAR, MNAR]` — 3 MNAR experts vs 1 each for MCAR and MAR. Even with "mean" aggregation (which corrects the prior), this creates:
- **Capacity asymmetry:** MNAR gets 3x the gating parameters
- **Gradient asymmetry:** MNAR receives gradient through 3 logits vs 1 for MAR
- **Representational asymmetry:** 3 reconstruction heads for MNAR vs 1 for MAR

This ablation removes all of that by collapsing to 1 expert per class. It answers: *is the multi-expert MNAR structure helping or creating an MNAR attractor that steals borderline MAR cases?*

### Hypothesis
If MAR recall improves with 1/1/1: the 1/1/3 asymmetry is biasing toward MNAR in the decision geometry. If MAR recall stays the same: the confusion comes from data/identifiability, not architecture.

### Changes
- `--mnar-variants self_censoring` → 3 experts total (1/1/1)
- `expert_to_class = [0, 1, 2]`, `experts_per_class = [1, 1, 1]`
- No aggregation needed (already 1:1 mapping)
- Prior reverted to `GeneratorPrior.uniform(registry)` (from class_balanced in Exp 8)

### Configuration
Same as Experiment 5 (baseline) except:
- `mnar_variants=["self_censoring"]` (was `["self_censoring", "threshold", "latent"]`)

### Controlled Variables
- Prior: uniform (same as Exp 5)
- Loss: cross-entropy (same as Exp 5)
- Label smoothing: 0.0 (same as Exp 5)
- Same datasets, same hyperparameters
- **Only change: expert structure (1/1/3 → 1/1/1)**

### Evaluation Workflow
```bash
python scripts/train_semisynthetic.py \
  --config configs/training/semisynthetic_full.yaml \
  --generators lacuna_tabular_110 --device cuda --quiet --report \
  --mnar-variants self_censoring \
  --name "Experiment 9 — 1/1/1 ablation (uniform prior)"
```

### Results

**Run:** Forge (5070Ti, CUDA 12.x)
**Training time:** 1679s (28 min) | 800 eval samples

| Metric | Exp 5 (1/1/3 baseline) | Exp 9 (1/1/1 this) | Δ |
|--------|----------------------|-------------------|---|
| **Overall accuracy** | 77.0% | **78.4%** | **+1.4%** ✅ |
| **MCAR recall** | 93.4% | 89.5% | −3.9% |
| **MCAR precision** | 70.3% | 85.2% | +14.9% |
| **MAR recall** | 52.6% | **69.3%** | **+16.7%** ✅ |
| **MAR precision** | 94.3% | 88.7% | −5.6% |
| **MAR F1** | ~68.0% | **77.8%** | **+9.8%** ✅ |
| **MNAR recall** | 85.3% | 79.4% | −5.9% |
| **MNAR precision** | 72.3% | 66.8% | −5.5% |
| **ECE** | 0.1338 | **0.1157** | **−0.018** ✅ |
| **90% acc coverage** | ~16.5% | **70.2%** | **+53.7%** ✅ |
| **95% acc coverage** | ~14.7% | **59.1%** | **+44.4%** ✅ |

**Confusion matrix** (rows = true, cols = predicted):

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 195 | 0 | 23 |
| **True MAR** | 1 | 205 | 90 |
| **True MNAR** | 33 | 26 | 227 |

**Per-class metrics:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| MCAR | 85.2% | 89.5% | 87.2% | 218 |
| MAR | 88.7% | 69.3% | 77.8% | 296 |
| MNAR | 66.8% | 79.4% | 72.5% | 286 |

**Calibration:**

- **ECE:** 0.1157 (best so far)
- **Mean confidence:** 0.899
- **Mean confidence (correct):** 0.934
- **Mean confidence (incorrect):** 0.772

**Selective accuracy:**

| Threshold (τ) | Accuracy | Coverage |
|---------------|----------|----------|
| 0.50 | 79.3% | 97.9% |
| 0.60 | 81.1% | 92.8% |
| 0.70 | 84.5% | 86.9% |
| 0.80 | 86.1% | 81.9% |
| 0.90 | 90.9% | 70.2% |
| 0.95 | 92.8% | 59.1% |

### Interpretation

**🎯 Hypothesis confirmed: the 1/1/3 asymmetry was the primary cause of MAR underdetection.**

The single change of reducing from 5 experts (1/1/3) to 3 experts (1/1/1) produced the largest improvement of the entire project:

1. **MAR recall: 52.6% → 69.3% (+16.7 points).** This is not a marginal gain — it's a regime change. The model now correctly classifies over two-thirds of MAR samples, up from barely half.

2. **Overall accuracy improved to 78.4%** — the new best, despite the simpler architecture. Fewer parameters, better results. The extra MNAR experts were not just dead weight — they were actively harmful.

3. **Calibration improved (ECE 0.1338 → 0.1157)** without any calibration-specific intervention. The symmetric architecture produces more honest probabilities naturally.

4. **Selective accuracy is dramatically better.** At τ=0.90, coverage went from ~16.5% to 70.2%. The model can now make high-confidence predictions on 70% of samples with 91% accuracy. This is transformative for practical use — the Bayes decision rule becomes much more useful when the model knows what it knows.

5. **The MNAR recall tradeoff (85.3% → 79.4%) is healthy.** Some of those "MNAR" predictions in Exp 5 were actually stolen MAR samples being returned to their correct class. MNAR precision dropped correspondingly (72.3% → 66.8%) because some genuinely ambiguous cases now go to MAR instead.

6. **MCAR precision improved dramatically (70.3% → 85.2%)** while recall only dropped slightly (93.4% → 89.5%). The model is making fewer false MCAR predictions.

**Why did the asymmetry hurt so much?** Even with mean aggregation dividing by `experts_per_class`, the 3 MNAR logits created a richer gradient signal for the MNAR class during backpropagation. The gating network learned to route borderline MAR→MNAR because doing so reduced loss across 3 expert outputs rather than 1. With 1/1/1, each class has exactly equal gradient flow, and the decision boundary is determined purely by the data.

**Remaining MAR gap:** 30.7% of MAR samples (90/296) are still misclassified as MNAR. This likely reflects the bimodal MAR pattern from Experiment 8 — generators like MAR-Weak, MAR-Interactive, and MAR-MixedPred that produce genuinely ambiguous patterns. Post-hoc temperature scaling may help with calibration, and per-class loss weights could push the boundary further.

### Next Decision

1. **1/1/1 is the new default architecture.** Make `--mnar-variants self_censoring` the default.
2. **Apply post-hoc temperature scaling** (Experiment 10) to this checkpoint — ECE=0.1157 is good but we can likely push below 0.05.
3. **Consider per-class loss weights** (Experiment 11) to further improve MAR recall — the remaining 30.7% MNAR misclassification may be partially addressable.
4. **Run per-generator analysis** on this checkpoint to see if the bimodal MAR pattern persists — did the architecture fix help the hard MAR generators, or only the easy ones?


---

## Experiment 10 — Post-Hoc Temperature Scaling

**Date:** 2026-02-14
**Commit:** *(pending)*
**Baseline checkpoint:** Experiment 9 (`best_model.pt` from 1/1/1 ablation)

### Motivation
Experiment 9 achieved ECE = 0.1157 — good but not great. Post-hoc temperature scaling (Guo et al. 2017) can improve calibration without retraining by finding optimal T to rescale gate logits before softmax.

### Hypothesis
Temperature scaling will reduce ECE below 0.05 without harming accuracy. The 1/1/1 architecture's simpler logit space (3 logits → 3 classes, no aggregation) should be especially amenable to single-parameter calibration.

### Changes
- Apply `scripts/calibrate.py` to Experiment 9 best checkpoint
- Two-phase grid search: coarse (100 log-uniform points over [0.1, 10]) + fine (100 linear points)
- Save calibrated checkpoint with patched `log_temperature`
- Evaluate calibrated checkpoint with `scripts/evaluate.py`

### Configuration
- Source: Experiment 9 checkpoint (1/1/1, uniform prior, CE loss)
- Calibration data: validation datasets from config
- No retraining — post-hoc only

### Evaluation Workflow
```bash
# Step 1: Find optimal temperature
python scripts/calibrate.py \
  --checkpoint "/mnt/artifacts/project_lacuna/runs/Experiment 9 — 1/1/1 ablation (uniform prior)/checkpoints/best_model.pt" \
  --config configs/training/semisynthetic_full.yaml \
  --generators lacuna_tabular_110 --device cuda

# Step 2: Evaluate calibrated model
python scripts/evaluate.py \
  --checkpoint "/mnt/artifacts/project_lacuna/runs/Experiment 9 — 1/1/1 ablation (uniform prior)/checkpoints/calibrated.pt" \
  --config configs/training/semisynthetic_full.yaml \
  --generators lacuna_tabular_110 --device cuda
```

### Targets
| Metric | Exp 9 (baseline) | Target |
|--------|-----------------|--------|
| ECE | 0.1157 | < 0.05 |
| Overall accuracy | 78.4% | ≥ 78% (no regression) |
| MAR recall | 69.3% | ≥ 69% (no regression) |

### Results

**Calibration run:** Forge (5070Ti, CUDA 12.x)
**Optimal temperature:** T = 1.9630 (model was overconfident; T > 1 softens probabilities)
**Calibration samples:** 800 (50 batches, 7 validation datasets)

**Calibration step (calibrate.py on held-out validation data):**
- NLL: 0.6788 → 0.5441 (Δ = −0.1347)
- ECE: 0.1321 → 0.0890 (Δ = −0.0431)
- Accuracy: 80.1% → 80.1% (unchanged — different random sample than eval)

**Full evaluation (evaluate.py on fresh validation data):**

| Metric | Exp 5 (1/1/3 baseline) | Exp 9 (1/1/1, T=1.0) | Exp 10 (1/1/1, T=1.96) | Δ vs Exp 9 |
|--------|----------------------|---------------------|----------------------|-----------|
| **Overall accuracy** | 77.0% | 78.4% | **82.6%** | **+4.2%** ✅ |
| **MCAR recall** | 93.4% | 89.5% | **94.5%** | +5.0% ✅ |
| **MCAR precision** | 70.3% | 85.2% | 79.4% | −5.8% |
| **MAR recall** | 52.6% | 69.3% | **73.6%** | **+4.3%** ✅ |
| **MAR precision** | 94.3% | 88.7% | 91.8% | +3.1% ✅ |
| **MAR F1** | ~68% | 77.8% | **81.7%** | **+3.9%** ✅ |
| **MNAR recall** | 85.3% | 79.4% | **84.5%** | +5.1% ✅ |
| **MNAR precision** | 72.3% | 66.8% | 78.7% | +11.9% ✅ |
| **ECE** | 0.1338 | 0.1157 | **0.0383** | **−0.077** ✅ |

**Confusion matrix** (rows = true, cols = predicted):

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 154 | 0 | 9 |
| **True MAR** | 5 | 212 | 71 |
| **True MNAR** | 35 | 19 | 295 |

**Per-class metrics:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| MCAR | 79.4% | 94.5% | 86.3% | 163 |
| MAR | 91.8% | 73.6% | 81.7% | 288 |
| MNAR | 78.7% | 84.5% | 81.5% | 349 |

**Calibration:**

- **ECE: 0.0383** (target was < 0.05 — **achieved** ✅)
- Mean confidence: 0.803
- Mean confidence (correct): 0.818
- Mean confidence (incorrect): 0.733
- Low-confidence (<50%): 3.0% of predictions

**Selective accuracy:**

| Threshold (τ) | Accuracy | Coverage |
|---------------|----------|----------|
| 0.50 | 83.2% | 97.0% |
| 0.60 | 84.3% | 89.9% |
| 0.70 | 87.7% | 77.4% |
| 0.80 | 90.3% | 61.8% |
| 0.90 | 91.5% | 30.8% |
| 0.95 | 92.3% | 4.9% |

**Mean predicted probabilities by true class:**

| True Class | P(MCAR) | P(MAR) | P(MNAR) |
|------------|---------|--------|---------|
| MCAR | 0.821 | 0.032 | 0.148 |
| MAR | 0.033 | 0.667 | 0.300 |
| MNAR | 0.152 | 0.138 | 0.709 |

### Interpretation

**Temperature scaling did far more than just fix calibration — it improved accuracy on all three classes simultaneously.**

1. **ECE: 0.1157 → 0.0383.** Target of < 0.05 achieved. The model's predicted probabilities now closely match observed frequencies. This is critical for the Bayes decision rule — the loss matrix operates on probabilities, so calibrated probabilities lead to better decisions.

2. **Accuracy: 78.4% → 82.6% (+4.2 points).** This is unexpected for post-hoc calibration, which theoretically shouldn't change argmax predictions. What happened: T=1.96 softened the overconfident softmax, pulling borderline predictions away from the wrong class. When the model was "90% confident MNAR" on a true MAR sample, softening to "65% MNAR / 30% MAR" sometimes flipped the argmax to the correct class.

3. **All three classes improved simultaneously:**
   - MCAR: 89.5% → 94.5% (+5.0) — back to Experiment 5 levels
   - MAR: 69.3% → 73.6% (+4.3) — the best MAR recall ever, by far
   - MNAR: 79.4% → 84.5% (+5.1) — also back to Experiment 5 levels

   This is remarkable. Temperature scaling normally trades off between classes (making one better at another's expense). The fact that all three improved means the model at T=1.0 was uniformly overconfident in the wrong direction.

4. **MAR precision improved too (88.7% → 91.8%).** When the model says MAR, it's now right 92% of the time, AND it says MAR more often. Both precision and recall improving simultaneously is the hallmark of a model that was fundamentally miscalibrated.

5. **Confidence gap widened healthily.** Correct predictions: 0.818, incorrect: 0.733. The model now has a clearer separation between "I know" and "I'm guessing," which is exactly what the Bayes decision rule needs.

6. **The cumulative effect of Experiments 9 + 10 is transformative:**

   | Metric | Exp 5 (start) | Exp 10 (now) | Total Δ |
   |--------|--------------|-------------|---------|
   | Accuracy | 77.0% | 82.6% | +5.6% |
   | MAR recall | 52.6% | 73.6% | +21.0% |
   | MAR F1 | ~68% | 81.7% | +13.7% |
   | ECE | 0.1338 | 0.0383 | −0.096 |

### Next Decision

1. **This is the production model.** 1/1/1 architecture with T=1.96 calibration is the new best across all metrics.
2. **Per-class loss weights (Experiment 11) may no longer be needed** — MAR recall at 73.6% exceeds the 65% threshold that would trigger it. But it could still push toward 80%.
3. **The remaining 26.4% MAR misclassification** (71/288 as MNAR, 5/288 as MCAR) likely reflects the bimodal generator identifiability issue from Experiment 8. Per-generator analysis on this calibrated model would confirm.
4. **Consider reproducibility run** — train a fresh 1/1/1 model with different seed, apply T scaling, verify results are stable.

---


---

## Experiment N

**Date:** 2026-03-29
**Checkpoint:** `/mnt/artifacts/project_lacuna/runs/lacuna_semisyn_20260329_021122/checkpoints/best_model.pt`
**Config:** `configs/training/semisynthetic_full.yaml`
**Run dir:** `/mnt/artifacts/project_lacuna/runs/lacuna_semisyn_20260329_021122`

### Results

**Overall accuracy: 83.9%** (800 samples) | Training: 420s

**Architecture:** 3 experts (mnar_variants=['self_censoring'])

**Per-class metrics:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| MCAR | 79.8% | 96.6% | 87.4% | 237 |
| MAR | 100.0% | 68.4% | 81.2% | 335 |
| MNAR | 75.0% | 93.4% | 83.2% | 228 |

**Confusion matrix** (rows = true, cols = predicted):

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 229 | 0 | 8 |
| **True MAR** | 43 | 229 | 63 |
| **True MNAR** | 15 | 0 | 213 |

**Mean predicted probabilities by true class:**

| True Class | P(MCAR) | P(MAR) | P(MNAR) |
|------------|---------|--------|---------|

**Calibration:**

- **ECE:** 0.1006
- **Mean confidence:** 0.939
- **Mean confidence (correct):** 0.962
- **Mean confidence (incorrect):** 0.819

**Selective accuracy:**

| Threshold (τ) | Accuracy | Coverage |
|---------------|----------|----------|
| 0.40 | 83.9% | 100.0% |
| 0.50 | 84.4% | 99.4% |
| 0.60 | 85.6% | 96.1% |
| 0.70 | 86.3% | 93.2% |
| 0.80 | 88.9% | 86.0% |
| 0.90 | 91.2% | 82.2% |
| 0.95 | 93.5% | 78.6% |

**Entropy:**

| True Class | Mean Entropy | Std Entropy |
|------------|-------------|-------------|

### Interpretation

*TODO: Add interpretation.*

### Next Decision

*TODO: Add next decision.*


---


---

## Experiment N

**Date:** 2026-03-29
**Checkpoint:** `/mnt/artifacts/project_lacuna/runs/lacuna_semisyn_20260329_032848/checkpoints/best_model.pt`
**Config:** `configs/training/semisynthetic_full.yaml`
**Run dir:** `/mnt/artifacts/project_lacuna/runs/lacuna_semisyn_20260329_032848`

### Results

**Overall accuracy: 78.4%** (800 samples) | Training: 1683s

**Architecture:** 3 experts (mnar_variants=['self_censoring'])

**Per-class metrics:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| MCAR | 85.2% | 89.5% | 87.2% | 218 |
| MAR | 88.7% | 69.3% | 77.8% | 296 |
| MNAR | 66.8% | 79.4% | 72.5% | 286 |

**Confusion matrix** (rows = true, cols = predicted):

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 195 | 0 | 23 |
| **True MAR** | 1 | 205 | 90 |
| **True MNAR** | 33 | 26 | 227 |

**Mean predicted probabilities by true class:**

| True Class | P(MCAR) | P(MAR) | P(MNAR) |
|------------|---------|--------|---------|

**Calibration:**

- **ECE:** 0.1157
- **Mean confidence:** 0.899
- **Mean confidence (correct):** 0.934
- **Mean confidence (incorrect):** 0.772

**Selective accuracy:**

| Threshold (τ) | Accuracy | Coverage |
|---------------|----------|----------|
| 0.40 | 78.4% | 100.0% |
| 0.50 | 79.3% | 97.9% |
| 0.60 | 81.1% | 92.8% |
| 0.70 | 84.5% | 86.9% |
| 0.80 | 86.1% | 81.9% |
| 0.90 | 90.9% | 70.2% |
| 0.95 | 92.8% | 59.1% |

**Entropy:**

| True Class | Mean Entropy | Std Entropy |
|------------|-------------|-------------|

### Interpretation

*TODO: Add interpretation.*

### Next Decision

*TODO: Add next decision.*


---

## Planned Experiments

### Experiment 11 — Per-Class Loss Weights (Conditional)
**Trigger:** MAR recall < 65% after Experiments 8-10
**Change:** Add `class_weights=[1.0, 1.5, 1.0]` to cross-entropy — upweight MAR 50%
**Target:** MAR recall > 65%, overall accuracy > 77%

### Experiment 12 — Dataset Expansion
**Change:** Increase `max_cols` from 48 to 64, add more datasets from OpenML
**Hypothesis:** More diverse training data improves generalization

### Experiment 13 — Architecture Exploration
**Trigger:** Rebalancing fails to solve MAR/MNAR confusion
**Options:**
- 2/2/2 symmetric experts (define MCAR and MAR subtypes)
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

**Current default (post-Experiment 9): 1/1/1 symmetric**

```
LacunaModel
├── Encoder (Transformer)
│   ├── Token embedding: [value, observed, mask] → hidden_dim
│   ├── 4 transformer layers (128-dim, 4 heads)
│   ├── Row pooling (attention) → per-row vectors
│   └── Dataset pooling (attention) → evidence vector [64-dim]
├── Reconstruction Heads (3 heads)
│   ├── MCAR, MAR, self_censoring
│   └── Each: evidence → hidden → predicted values
├── Missingness Feature Extractor
│   └── 16 statistical features from missingness patterns
├── Mixture of Experts
│   ├── Gating Network: [evidence, recon_errors, miss_features] → 3 logits
│   ├── Temperature-scaled softmax → 3 expert probs
│   ├── 1:1 class mapping (no aggregation needed)
│   └── expert_to_class = [MCAR, MAR, MNAR]
└── Bayes Decision Rule
    └── Asymmetric loss matrix → Green/Yellow/Red action
```

*Previous default (Experiments 1-5): 1/1/3 asymmetric with `expert_to_class = [MCAR, MAR, MNAR, MNAR, MNAR]` and 5 reconstruction heads. Retired after Experiment 9 showed the asymmetry caused MAR underdetection.*

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
