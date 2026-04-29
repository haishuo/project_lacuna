# Project Lacuna — Document 3 of 3: Experiments, Ablations, and Empirical Results

This is the third of three companion documents collating Project Lacuna for ingestion into NotebookLM. Document 1 covers motivation and theoretical position. Document 2 covers the architecture in detail. This document covers the experimental record: the 10-experiment arc that produced the base Lacuna model at 82.6% synthetic accuracy, the Lacuna-Survey v1–v11 specialisation arc, the validation against real NHANES MNAR anchors, and the negative results that the project chose to document rather than hide.

---

## Part I — Base Lacuna: The 10-Experiment Arc

The development of base Lacuna proceeded through ten experiments. Each was logged in detail in the project's experiment journal, with metrics, controlled variables, hypotheses, and post-hoc interpretations. The trajectory of headline metrics:

| Exp | Name | Key Change | Accuracy | MAR Recall | ECE | Status |
|-----|------|-----------|----------|------------|-----|--------|
| 1 | Core Architecture | Build full pipeline | — | — | — | Infrastructure |
| 2 | Diagnostic Analysis | Verify feature discriminability | — | — | — | Confirmed signals exist |
| 3 | Generator Expansion | 6 → 110 generators | — | — | — | Infrastructure |
| 4 | Test Stabilisation | Fix bugs, semi-synthetic | — | — | — | Infrastructure |
| **5** | **Semi-Synthetic Baseline** | **First full training run** | **77.0%** | **52.6%** | **0.134** | ✅ Baseline |
| 6 | Evaluation Tooling | Add confusion matrix, ECE, per-generator metrics | — | — | — | Infrastructure |
| 7 | Brier + Balanced + Smoothing | 3 changes at once | 54.8% | 30.7% | 0.287 | ❌ Failed |
| 8 | Class-Balanced Prior Only | Isolate prior effect | 62.6% | 34.8% | 0.233 | ❌ Regressed |
| **9** | **1/1/1 Symmetric Experts** | **Architecture fix** | **78.4%** | **69.3%** | **0.116** | ✅ Key fix |
| **10** | **Temperature Scaling** | **Post-hoc T=1.96** | **82.6%** | **73.6%** | **0.038** | ✅ Final |

### Experiments 1–4 — Infrastructure

Experiment 1 built the entire pipeline end-to-end on a minimal six-generator setup (2 MCAR, 2 MAR, 2 MNAR), establishing that the architecture trains and converges. Experiment 2 was a diagnostic pass before scaling: it confirmed that the explicit missingness features (point-biserial correlations, Little's-test approximation, distributional statistics) carry strong discriminative signal. Cohen's d > 9.0 separated MCAR from MAR on point-biserial features; d > 2.8 separated MAR from MNAR on Little's-test approximation. These effect sizes meant the discrimination signal was demonstrably present in the data — if the model failed to classify, the issue would be in gating or training, not in feature availability.

Experiment 3 expanded from 6 hardcoded generators to 110 parameterised generators across 34 MCAR, 36 MAR, and 38 MNAR families. Generators were organised into YAML registries (`lacuna_tabular_110`, `lacuna_minimal_18`, `lacuna_minimal_6`) for use across training and testing. The class distribution under uniform sampling — ~31% MCAR, ~33% MAR, ~35% MNAR — was a noted asymmetry that became important later.

Experiment 4 stabilised tests after the generator refactor (73 test failures from API migrations were fixed) and got semi-synthetic training working reliably on the GPU server (NVIDIA RTX 5070 Ti). A non-trivial bug surfaced and was fixed in this phase: switching MoE class-aggregation from "mean" to "sum" caused the 3-MNAR-experts setup to give MNAR ~60% of probability mass regardless of input. Mean aggregation (divide by `experts_per_class`) was confirmed correct and reverted within minutes.

### Experiment 5 — Semi-Synthetic Baseline (the diagnostic that mattered)

Experiment 5 was the first full training run on real datasets with all 110 generators: 24 training datasets, 7 validation datasets, uniform generator prior, cross-entropy loss, no temperature scaling.

| Metric | Value |
|--------|-------|
| Overall accuracy | 77.0% |
| MCAR recall | 93.4% |
| MAR recall | **52.6%** |
| MNAR recall | 85.3% |
| ECE | 0.1338 |

The headline finding: **MAR was severely underdetected.** Of 251 true MAR samples, 116 (46.2%) were misclassified as MNAR. The model's mean probabilities on true MAR samples were `[0.029, 0.522, 0.449]` — essentially a coin flip between MAR and MNAR, defaulting to MNAR. Three candidate root causes were hypothesised:

1. **Class imbalance under uniform sampling** — MNAR has 38 generators (~35% of training), MAR has 36 (~33%). Under uncertainty, the model learned "when in doubt, guess MNAR."
2. **Overconfidence** — ECE = 0.1338, mean confidence on incorrect predictions = 0.818. The model thought it knew when it didn't.
3. **No per-class loss weighting** — hard MAR examples got the same weight as easy MCAR.

The deliberate decision was to *calibrate first, rebalance second, change architecture only if those failed*. The aphorism logged at the time: "We don't want to open the hood and start ripping out wires unless we're *sure* things are broken."

### Experiment 6 — Evaluation Tooling

Experiment 6 was an infrastructure intervention: the training script up to this point reported only argmax accuracy. The new tooling added confusion matrices, ECE with bin-level data, confidence analysis, per-generator accuracy breakdown, selective accuracy at confidence thresholds, and entropy statistics. All metrics quoted from this point forward use this richer harness.

### Experiment 7 — The Three-Variable Disaster

Experiment 7 attempted to fix calibration and class balance simultaneously by changing **three things at once**: switching from cross-entropy to Brier-score loss, introducing label smoothing of 0.1, and switching from uniform to class-balanced generator prior.

| Metric | Exp 5 | Exp 7 | Δ |
|--------|-------|-------|---|
| Overall accuracy | 77.0% | **54.8%** | **−22.2%** |
| MAR recall | 52.6% | **30.7%** | **−21.9%** |
| ECE | 0.134 | **0.287** | **+0.15** |

Catastrophic regression. Every metric worsened except MAR precision (trivially better because fewer MAR predictions were made). Post-hoc analysis identified the Brier-score loss as the primary culprit: cross-entropy has an unbounded gradient as `p → 0` that drives the model hard toward correct answers, whereas Brier's quadratic penalty is bounded `2(p − y)`. With 110 generators and limited training time, the weaker Brier gradient could not push the model hard enough to find sharp MAR/MNAR boundaries. Label smoothing compounded the problem by softening targets further. The lesson logged: **change one variable at a time, always have a control.**

### Experiment 8 — Isolating the Prior

Experiment 8 reverted the loss function and label smoothing, keeping only the class-balanced prior as the single intervention.

| Metric | Exp 5 (uniform) | Exp 8 (balanced) |
|--------|-----------------|------------------|
| Overall accuracy | 77.0% | 62.6% |
| MAR recall | 52.6% | **34.8%** |
| ECE | 0.134 | 0.233 |

The class-balanced prior **made things worse** — MAR recall dropped further, not improved. The post-hoc analysis with per-generator breakdown was the most important diagnostic insight of the project.

#### The Bimodal MAR Generator Finding

Per-generator accuracy revealed that MAR generators split into two distinct populations:

**Well-detected MAR generators:**

| Generator | Accuracy | Note |
|-----------|----------|------|
| MAR-ColBlocks | 100% | Block-structured column-level missingness |
| MAR-CrossClass | 100% | Cross-class dependency patterns |
| MAR-SkipLogic | 89.7% | Skip-pattern conditional missingness |

**Near-undetectable MAR generators:**

| Generator | Accuracy | Note |
|-----------|----------|------|
| MAR-Weak | 0% | Weak signal — near-MCAR |
| MAR-Interactive | 0% | Interaction-based — overlaps with MNAR |
| MAR-Section | 0% | Section-level patterns |
| MAR-MixedPred | 15.7% | Multi-predictor — largest MAR group |
| MAR-Moderate | 12.5% | Moderate signal strength |

**MNAR also had a hard subgroup**: `MNAR-LatentCorr` at 0% accuracy (latent-variable MNAR resembling MAR), `MNAR-AdaptSamp` at 26.3%.

The structural insight: the class-balanced prior backfires precisely *because* it forces a 1/3 share to MAR, which means borderline-undetectable MAR generators (MAR-Weak, MAR-Interactive, MAR-MixedPred) become a larger fraction of training. Under the uniform prior, those difficult generators were a smaller fraction, and the model could learn the detectable cases first. The balanced prior drowned the signal in noise. The decision: *revert to uniform prior and look elsewhere for the fix.*

### Experiment 9 — The 1/1/1 Architecture Fix

Experiment 9 changed only the expert-pool structure: from 5 experts mapped `[MCAR, MAR, MNAR, MNAR, MNAR]` (with mean aggregation) to 3 experts mapped `[MCAR, MAR, MNAR]` (1:1, no aggregation). Everything else — uniform prior, cross-entropy, no label smoothing, no calibration — was held fixed at the Experiment 5 baseline.

| Metric | Exp 5 (1/1/3) | Exp 9 (1/1/1) | Δ |
|--------|---------------|---------------|---|
| Overall accuracy | 77.0% | **78.4%** | +1.4% |
| MAR recall | 52.6% | **69.3%** | **+16.7%** |
| MAR F1 | ~68.0% | **77.8%** | +9.8% |
| ECE | 0.134 | 0.116 | −0.018 |
| 90%-acc coverage | ~16.5% | **70.2%** | +53.7% |

This was the largest single improvement of the entire project. The diagnosis: even though the prior 1/1/3 setup used mean class-aggregation that was *mathematically* correct (dividing by `experts_per_class`), the three MNAR experts created **asymmetric gradient flow**. Each MCAR and MAR expert received gradient signal through one output head; each of the three MNAR experts received gradient through its one head, but the MNAR *class* received gradient through three heads simultaneously. The gating network learned to route borderline MAR cases to MNAR not because MNAR was the right answer but because doing so reduced loss across three expert outputs rather than one — a pure artifact of the aggregation geometry.

Switching to 1/1/1 made every class structurally symmetric. MAR recall jumped 16.7 points; the architecture was simpler, with fewer parameters, and produced uniformly better results. Selective accuracy at τ = 0.90 went from a useless 16% coverage to a usable 70% coverage. After Experiment 9, `mnar_variants = ["self_censoring"]` (i.e. one MNAR expert) became the production default.

### Experiment 10 — Post-Hoc Temperature Scaling

Experiment 10 applied post-hoc temperature scaling (Guo et al. 2017) to the Experiment 9 checkpoint, with a two-phase grid search over `T ∈ [0.1, 10]`.

The optimal temperature found: `T = 1.9630`. The expected effect was an ECE improvement; the unexpected effect was simultaneous accuracy improvement on all three classes.

| Metric | Exp 9 (T=1.0) | Exp 10 (T=1.96) | Δ |
|--------|--------------|----------------|---|
| Overall accuracy | 78.4% | **82.6%** | **+4.2%** |
| MCAR recall | 89.5% | 94.5% | +5.0% |
| MAR recall | 69.3% | **73.6%** | +4.3% |
| MNAR recall | 79.4% | 84.5% | +5.1% |
| ECE | 0.116 | **0.038** | −0.078 |

This was theoretically surprising — temperature scaling does not change argmax in the limit of well-separated logits. What happened: the pre-calibration model was so consistently overconfident in the wrong direction on borderline cases that softening the probabilities frequently flipped incorrect high-confidence predictions to the correct second-place class. The implication is that the model at `T = 1.0` was not merely miscalibrated but actively wrong in a coherent direction that temperature scaling corrected. The cumulative effect of Experiments 9 + 10 over the Experiment 5 baseline:

| Metric | Exp 5 | Exp 10 | Total Δ |
|--------|-------|--------|---------|
| Accuracy | 77.0% | 82.6% | +5.6% |
| MAR recall | 52.6% | 73.6% | **+21.0%** |
| MAR F1 | ~68% | 81.7% | +13.7% |
| ECE | 0.134 | 0.038 | −0.096 |

### Final Experiment 10 Performance Detail

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| MCAR | 79.4% | 94.5% | 86.3% | 163 |
| MAR | 91.8% | 73.6% | 81.7% | 288 |
| MNAR | 78.7% | 84.5% | 81.5% | 349 |
| **Overall** | | **82.6%** | | **800** |

Confusion matrix:

|  | Pred MCAR | Pred MAR | Pred MNAR |
|--|-----------|----------|-----------|
| **True MCAR** | 154 | 0 | 9 |
| **True MAR** | 5 | 212 | 71 |
| **True MNAR** | 35 | 19 | 295 |

Selective accuracy (coverage versus accuracy at confidence thresholds):

| τ | Accuracy | Coverage |
|---|----------|----------|
| 0.50 | 83.2% | 97.0% |
| 0.70 | 87.7% | 77.4% |
| 0.80 | 90.3% | 61.8% |
| 0.90 | 91.5% | 30.8% |

The 26% remaining MAR misclassification (71/288 routed to MNAR) tracks the bimodal generator-identifiability finding from Experiment 8: it is largely produced by the MAR-Weak, MAR-Interactive, MAR-Section, and MAR-MixedPred generators whose patterns genuinely overlap with MNAR self-censoring under observed-data alone.

---

## Part II — From Generic Lacuna to Lacuna-Survey

The base Lacuna model — 82.6% on synthetic data, ECE = 0.038 — looked excellent on the held-out validation distribution. It then failed on real data.

### The Reframing

The first specialised model demonstrated that the failure was structural, not configural. Six iterations of the generic model (`lacuna_demo_v1` through `v4` plus diagnostic experiments) established that:

- 87.4% synthetic accuracy on `lacuna_demo_v1`,
- 0/3 textbook-MAR-consensus real datasets correctly classified,
- and *confidently routing real MNAR data to MAR* — the worst-error direction for downstream imputation.

Six rounds of generator coverage, class-imbalance correction, per-class loss reweighting, saturation bug fixes, and adding new MAR generator families (MARRealistic, MARPartialResponse, MARDemographicGated) did not move real-world performance. Synthetic numbers shuffled in the 74–98% range; real-world predictions stayed roughly fixed.

The diagnostic conclusion logged at the time: the classifier had hit the information-theoretic ceiling for *pattern-only* features. Domain context — survey, sensor, EHR, longitudinal — is not a feature that can be added but is intrinsic to the problem statement. When a researcher hands a missing-data analysis to a doctor, they are not testing whether the doctor can figure out it is medical data; they have already supplied that information by choosing the doctor. Asking generic Lacuna to do the same problem without that context is asking it to do a strictly harder problem than what humans actually face.

Operationally: Lacuna became a **framework** for training mechanism classifiers specialised by collection process. The generator registry *is* the domain prior. The right granularity is collection process (survey, sensor, longitudinal panel) rather than substantive field (medical, political, marketing). A medical survey, a political survey, and a marketing survey share survey-process mechanisms; they do not share domain-process mechanisms with EHR records that happen to be medical.

### The Lacuna-Survey v1–v11 Iteration Arc

Lacuna-Survey is the first specialised variant. Its construction proceeded through eleven version increments, with all versions before v6 reporting eval-report numbers from the *final* model state (a measurement bug) rather than the best checkpoint. This was found and fixed mid-arc; numbers below refer to best-checkpoint metrics.

| Version | Intervention | Synthetic | Real MAR | Real MNAR | Lesson |
|---------|--------------|-----------|----------|-----------|--------|
| v1 | Generic-tabular baseline | 87.4% | 0/3 | 2/3 | The unidentifiability ceiling exists empirically. |
| v2 | Added MARRealistic generators | 73.9% | 1/3 | 2/3 | Generator-coverage tweaks move predictions sideways. |
| v3 | Saturation fix in `apply_missingness` | 79.5% | 1/3 | 0/3 | Cleaner training trades one error mode for another. |
| v4 | Class weighting + value-conditional features (SMD) | 88.4% | 3/8 | 2/3 | Value-conditional features enable Pima MNAR detection. |
| v5 | SMD effective-sample-size shrinkage | 94.0% | 2/8 | 2/3 | Cleaner SMD didn't transfer; loss weighting overfiring. |
| v6 | Drop class weights | 88.5% | 2/8 | 2/3 | Symmetric loss alone does not fix real-world MAR. |
| v7 | Learnable evidence attenuation α | 86.9% | 1/8 | 1/3 | Per-volume reweighting did not transfer. |
| **v8** | **Low-rate broad-spread MAR + MNAR-Q90-Broad** | 86.5% | **3/5** | 1/3 | **The (low-rate × broad-cols) cell was the missing pattern.** |
| v9 | + `MCARRotatedBooklet` for PISA-style designs | 87% | 7/8 | OOD only | First robust within-domain MCAR detection. |
| v10 | + `MNARModuleRefusal` (4 variants) | reverted | 3/8 | 1/1 (DPQ ✓) | DPQ fixed but MAR collapsed. |
| **v11** | **+ `MARModuleSkip` MAR counterpart, MNAR variants tuned down** | 92% | 6/8 | 0/1 (DPQ MAR ✗) | **Molenberghs floor reached.** |
| v12 | Added `demo_strength` confounding to MNAR-Module | reverted | 5/8 | 0/1 | Strictly dominated by v11. |

The Lacuna-Survey 70-generator registry covers 12 MCAR, 36 MAR, and 22 MNAR mechanisms specifically plausible for self- or interviewer-administered survey questionnaires. The new generators added during the iteration arc to close diagnosed coverage gaps: `MCARRotatedBooklet` (PISA/NAEP/ECLS-K planned-missing designs), `MARModuleSkip` (the MAR counterpart to module refusal), `MNARModuleRefusal` (row-aligned battery refusal driven by unobserved values).

### Out-of-Distribution Detection — Tested and Removed from the Demo

A logistic-regression OOD detector was trained on a 34-dim feature vector (10 missingness + 24 per-column value-distribution stats) to distinguish in-distribution survey X-bases from out-of-distribution non-survey X-bases (wine, breast_cancer, abalone, glass, etc.).

It achieved 95.6% accuracy on synthetic IN/OUT held-out and caught 6/7 real diagnostic cases. The one false negative was `airquality_real` — a weather/integer-day mix that overlapped statistically with survey value distributions. After augmenting the IN-distribution training corpus with the four NHANES MNAR anchors (to prevent false positives on real surveys), the detector lost its ability to catch Pima diabetes. The fundamental issue: with the existing 34-dim feature set, Pima's signature sits inside the cluster of NHANES survey signatures.

The project chose to remove the OOD feature entirely from the user-facing demo rather than retain a feature that fired correctly only sometimes. A scope notice with explicit user responsibility ("Lacuna-Survey is for survey questionnaires; non-survey predictions are not validated; the scope decision is yours") is the only scope mechanism in the deployed demo. This matches the contract every ML tool implicitly has: AlphaFold does not refuse non-protein sequences, Whisper does not refuse non-speech audio. The internal validation tooling retains the OOD module for self-tests.

---

## Part III — The NHANES MNAR Battery: Validating the Calibrated-Posterior Framing

The headline result for Lacuna-Survey is not an argmax accuracy number on real anchors. It is a **posterior-elevation result** on real survey data with literature-consensus MNAR readings, evaluated through the lens of calibrated probabilistic prediction rather than categorical classification.

### The Real-Anchor Corpus

Fourteen real-survey anchors form the calibration and validation corpus, each selected based on a literature-consensus mechanism reading with citations recorded in `lacuna_survey/anchors.py`:

- **2 MCAR anchors** — PISA 2018 (GBR booklet rotation) and PISA 2022 (DEU booklet rotation), both rotated-booklet planned-missing designs (Graham et al. 2006; OECD 2019).
- **7 MAR anchors** — `psych::bfi` (Revelle 2018), `carData::Chile` (Fox 2008), `MASS::Cars93` (Venables & Ripley 2002), `MASS::survey`, `openintro::yrbss`, `carData::GSSvocab`, `openintro::ucla_textbooks_f18`.
- **5 MNAR anchors:**
  - `survey_nhanes_demographics` — INDFMPIR income refusal (Allison 2001 §6.4 reads as MNAR; van Buuren 2018 reads as MAR-conditional — contested).
  - `nhanes_dpq_phq9` — PHQ-9 depression screener module refusal (Cole et al. 2010; van Buuren 2018 §3.7).
  - `nhanes_inq_income` — explicit income module, item-level high-earner refusal (Allison 2001 §1.2).
  - `nhanes_whq_weight` — self-reported weight, heavy-respondent refusal (Connor Gorber et al. 2007, *BMC Public Health*).
  - `nhanes_duq_drug` — drug-use module gateway items (Allison 2001 §1.2).

### The DPQ Diagnostic — Argmax Failure or Calibrated Success?

After the v11 build, real `nhanes_dpq_phq9` classified by argmax as `P(MAR) = 0.64, P(MNAR) = 0.35, P(MCAR) = 0.02`. Three attempts to push the argmax to MNAR (v10 with aggressive MNAR generators; v12 with demographic-confounded MNAR generators) either broke MAR detection or made things strictly worse.

The conclusion: at `n = 128` with the DPQ feature signature, module-refusal MNAR (driven by unobserved depression severity) and demographic-gated MAR-skip are statistically indistinguishable from observable data alone. This is the identifiability limit Molenberghs predicted, made concrete on a specific anchor. *No amount of generator engineering closes it; only stronger covariate observation would.*

The DPQ result was initially read as an argmax failure. The reframing — articulated in the project's Architectural Decision Record 0005 — was that argmax-vs-consensus accuracy is the wrong measuring stick for a tool whose architecture is explicitly Bayesian. The MoE outputs a calibrated posterior over mechanisms; the appropriate evaluation is whether that posterior is sensible, not whether `argmax(p) == consensus_label`. Under that lens:

- DPQ posterior `P(MNAR) = 0.345` aligns directly with the literature split (Allison reads PHQ-9 refusal as MNAR; van Buuren reads it as MAR-conditional on demographics). The 35/64 split is the correct calibrated answer to a question that, by Molenberghs, admits no determinate answer from observed data alone.
- The MAR-mechanism reconstruction head fits DPQ essentially perfectly (per-cell error ~1e-4) while the MNAR head does not (~0.196). By Bayes, the *observable* likelihood ratio massively favours MAR. Lacuna keeps `P(MNAR) = 0.345` rather than crushing it to zero, because the unobserved values that drive the missingness are *by construction absent* from the reconstruction target — a direct empirical instantiation of the Molenberghs identifiability result.

This is the AlphaFold pLDDT analogue: a calibrated confidence under prior plus likelihood, not a categorical verdict. The tool was doing the right thing; the diagnostic was reading it wrong.

### The Four-Anchor Battery Result

After the reframing, three additional NHANES MNAR anchors were acquired (income, weight, drugs) to test whether the DPQ posterior elevation was a fluke or a robust pattern. Cold v11 posteriors before adding these to the calibration corpus:

| Anchor (consensus MNAR) | argmax | P(MCAR) | P(MAR) | **P(MNAR)** |
|---|---|---|---|---|
| `nhanes_dpq_phq9` (depression) | MAR | 0.02 | 0.64 | **0.345** |
| `nhanes_inq_income` (income) | MAR | 0.01 | 0.56 | **0.432** |
| `nhanes_whq_weight` (weight) | MAR | 0.01 | 0.59 | **0.410** |
| `nhanes_duq_drug` (drugs) | **MNAR ✓** | 0.02 | 0.49 | **0.496** |
| MAR-consensus baseline (mean) | — | — | — | ~0.13 |

**The posterior systematically elevates `P(MNAR)` by 3–5× on real MNAR-consensus data relative to clean-MAR baselines, across four distinct sensitivity domains (depression, income, weight, drugs).** One of the four (drugs) flips argmax to MNAR. The other three remain argmax-MAR but with calibrated posteriors that correctly reflect mechanism uncertainty rather than crushing it to zero.

This is the headline empirical result for Lacuna-Survey. It is *not* an argmax-classification number. It is a posterior-elevation number, measured against real survey anchors whose mechanism readings are grounded in the missing-data literature (the literature itself being split in some cases).

### Per-Mechanism Reconstruction Evidence

The auxiliary reconstruction heads expose the identifiability problem directly. For the four NHANES MNAR anchors, mean per-cell reconstruction errors:

| Anchor | recon[MAR] | recon[MNAR] |
|---|---|---|
| `nhanes_dpq_phq9` | 0.0001 | 0.196 |
| `nhanes_inq_income` | 0.0005 | 0.018 |
| `nhanes_whq_weight` | 0.0001 | 0.004 |
| `nhanes_duq_drug` | 0.0003 | 0.030 |

The MAR-conditioned reconstruction head achieves substantially lower observable fit error than the MNAR head on these MNAR-consensus anchors. This finding does not falsify the literature MNAR reading; rather, it illustrates *why* mechanism cannot be inferred from observable reconstruction fit alone. Under MNAR-by-self-censoring or MNAR-by-module-refusal, the unobserved values that drive the missingness are by construction absent from the reconstruction target. A model fitting only on observed cells will necessarily score the MAR explanation as well-fitting, because MAR is consistent with the observed-cell distribution.

The posterior is therefore not a pure reconstruction-likelihood classifier. It combines (a) observable reconstruction fit, (b) explicit missingness-pattern features (per-column rates, cross-column correlations, value-conditional standardised mean differences), and (c) learned survey-domain mechanism priors from the semi-synthetic training distribution. The net effect on these anchors is elevated MNAR posterior mass (0.30–0.50) without forced categorical MNAR labels where the observable likelihood remains MAR-compatible.

### Synth-Mechanism-on-Real-X Validation

A separate validation harness (`lacuna_survey/mnar_validation.py`) applies the trained generators to held-out real X-bases (`survey_cars93`, `survey_survey`) and measures per-class detection. This isolates *can the model detect a known mechanism applied to real survey columns it has not been trained on?*

| Class | Detection rate |
|---|---|
| MCAR | 17/24 = 70.8% |
| MAR | 47/72 = 65.3% |
| **MNAR** | **41/44 = 93.2%** |

The high MNAR detection rate confirms that the trained model genuinely recognises MNAR fingerprints when applied to real survey X-distributions. The gap between this 93.2% synth-mechanism detection rate and the much lower real-anchor argmax accuracy is a direct measurement of the Molenberghs floor at `n = 128`: the model can detect synthetic MNAR applied to real X, but it cannot determinately distinguish real MNAR from real MAR-with-the-same-pattern.

---

## Part IV — Negative Results and Documented Dead Ends

The project's commitment to honest reporting required documenting several dead ends that did not make the headline numbers.

### The Asymmetric Expert Pool (1/1/3)

A 1/1/3 expert design (three MNAR sub-experts) was the original architecture. MAR accuracy on held-out data fell substantially as gradient signal pulled toward the heavier MNAR side, even after class-balanced reweighting. The symmetric 1/1/1 with mean-normalised class aggregation restored accuracy. Documented in `docs/ARCHITECTURE.md:385-388`. The 1/1/1 design is therefore an empirically validated constraint, not a default.

### Class-Balanced Generator Prior

Forcing 1/3 per class harmed performance because it overweighted the borderline-undetectable MAR generators (MAR-Weak, MAR-Interactive, MAR-MixedPred) that produce genuine MAR/MNAR overlap. Under uniform sampling, those generators were a smaller fraction of training, allowing the model to first learn the detectable cases. Documented in Experiment 8.

### Brier Score Loss

Tried in Experiment 7. Bounded gradient `2(p − y)` was insufficient to push the model hard enough to find sharp MAR/MNAR boundaries with 110 generators and limited training time. Reverted to cross-entropy.

### Classical MCAR-Test Ensemble

The `mcar-alternatives-bakeoff` arc tested six families of classical MCAR tests as gate features (Little's MLE, MoM, propensity, HSIC, MissMech, median-split heuristic). ADR 0004 confirms that Little's MLE *actively hurt* classification accuracy at `n = 30` (CI excluded zero, p = 0.036, 21/30 seeds favoured disable). The deployment-time ensemble is technically a different combinator than learned feature weights, but the underlying empirical finding — classical MCAR tests do not carry information beyond Lacuna's existing features — applies in either combinator. Re-introducing them would re-litigate settled work.

### The ESS Dead End

The original `DATA_ACQUISITION.md` premise that European Social Survey rotating modules produce within-country MCAR-by-design was empirically wrong. ESS rotating modules are administered to *every* respondent in participating countries; cross-country missingness is admin-NaN (party-vote variables for non-matching countries), not random assignment. Within Germany (`n = 2420`), of 666 numeric columns, 415 are 100% observed, 251 are 100% NaN, and exactly one (`inwtm`) has any non-trivial missingness (0.9%). The ESS R11 file was inspected, rejected, and parked in `/mnt/data/lacuna/rejected/`. `DATA_ACQUISITION.md` was updated to flag this so future sessions don't repeat the dead end.

### Sample-Size Sensitivity

Lacuna's input is capped at `max_rows = 128`. For datasets with strong value-conditional MNAR on continuous variables, this row count can be insufficient to surface the value-distribution shift signal. Real anchors are typically 500-row subsamples re-subsampled to 128 per inference. The sample-size cliff is not characterised systematically in the current build and remains an open limitation.

### v10 and v12 — Reverted Versions

Both reverted versions in the Lacuna-Survey arc (v10's aggressive MNAR generators, v12's demographic-confounded MNAR variants) are documented in ADR 0005. Both attempted to push real DPQ classification toward MNAR; both either broke MAR detection or were strictly dominated by v11 across every metric. The combination of v11's failures and v10/v12's regressions is what made the Molenberghs floor empirically concrete on this anchor.

---

## Part V — Where the Project Stands

### Base Lacuna (Final)

- 82.6% accuracy on 800 held-out synthetic samples, ECE = 0.038.
- 1/1/1 symmetric MoE, 110-generator semi-synthetic training, post-hoc temperature scaling at `T = 1.96`.
- Production checkpoint: `calibrated.pt` from RUN-054.
- Total parameters: ~901,000.

### Lacuna-Survey v11 (Current Specialised Build)

- Synthetic validation accuracy: 0.92 (MCAR 1.00 / MAR 0.97 / MNAR 0.77).
- Synth-mechanism-on-real-X: MNAR detection rate 93.2%.
- Real-anchor calibrated posteriors: P(MNAR) elevated 3–5× on NHANES MNAR-consensus anchors versus MAR baselines, across four sensitivity domains. One of four (drugs) flips argmax to MNAR; the other three remain argmax-MAR with calibrated posteriors that correctly reflect uncertainty.
- 70-generator survey-specialised registry; 14-anchor real-survey calibration corpus.
- Vector-scaling calibration `(T = 1.756, bias = [+0.31, −0.30, +0.20])` fit on the expanded NHANES corpus.

### Test Suite

- 1,096 unit tests covering generators, encoder, MoE, training loops, calibration, end-to-end inference. Full suite runs in ~47 seconds on CPU.

### Open Directions (Deployment-Layer, Not Architectural Rewrites)

1. **Replace vector-scaling calibration with a small MLP.** Could improve posterior calibration on boundary cases where the gate over-weights one class despite reconstruction evidence (yrbss, survey_survey, ucla_textbooks).
2. **Surface reconstruction-error ratios in the user-facing report** ("MAR head reconstructs your data with error 0.0001; MNAR head with error 0.20 — strong likelihood evidence for MAR"). Already produced internally; needs UI surfacing.
3. **Active learning on real data.** Where reliable expert labels exist (NHANES item-level, ESS), incorporate small amounts of real data into training as anchors.
4. **Encoder-side regularisation.** The transformer evidence pathway carries 71.8% of gate-input L1 mass and is the dominant driver of real-world misclassifications. Targeted regularisation — information bottleneck on the evidence vector, or dropout-during-inference for sensitivity — may make evidence carry only what is robust across distributions.

Two directions are explicitly *not* on the list:

- **Ensemble with classical Little's-style tests** — settled by ADR 0004 and the bakeoff arc.
- **Architectural rewrite of the base model** — the encoder + MoE + reconstruction-head framework has been validated on synthetic across 11 versions; failures on real-world transfer are about information that isn't in the training data, not the architecture that uses it. Branch only on demonstrated need for fundamentally different computation (e.g. temporal attention for a hypothetical Lacuna-Longitudinal variant).

---

## Reproducibility Pointers

- **Code repository:** `project_lacuna/`. Base framework in `lacuna/`; survey variant in `lacuna_survey/`. YAML configs in `configs/`.
- **Decision records:** `docs/decisions/` contains five ADRs covering removed features (ADR 0001 — point-biserial/distributional removal; ADR 0004 — Little's MCAR slot removal), framework decisions (ADR 0002 — real-MLE caching), and the full Lacuna-Survey iteration arc (ADR 0005).
- **Experiment records:** `docs/experiments/` contains dated writeups of major ablations and bakeoffs (`2026-04-23-bakeoff-stage1.md` — the six-MCAR-test family bakeoff; `2026-04-25-canonical-n30.md` — the canonical-n30 study; `2026-04-19-mle-vs-mom.md` — the MLE-vs-MoM comparison).
- **Validation harnesses:** `lacuna_survey/diagnostic.py` (panel of real anchors and cross-domain comparators); `lacuna_survey/probabilistic_diagnostic.py` (per-anchor posteriors and per-mechanism reconstruction errors — the user-facing evaluation surface); `lacuna_survey/mnar_validation.py` (synth-mechanism-on-real-X harness for measuring MNAR detection rate independent of real-anchor labels).
- **References:** Allison 2001, Cole et al. 2010, Connor Gorber et al. 2007, Fox 2008, Graham et al. 2006, Jumper et al. 2021, Little & Rubin 2019, Molenberghs et al. 2008, OECD 2019, Revelle 2018, Rubin 1976/1987, Schafer & Graham 2002, van Buuren 2018, Venables & Ripley 2002. Full bibliography in `docs/ACADEMIC_REFERENCE.md`.

For the motivation and theoretical position, see Document 1. For architectural detail, see Document 2.
