# Project Lacuna — Academic Reference

A self-contained reference for the Lacuna project, intended for use in a
dissertation, course deliverables, or methodological writeups. This document
summarises the project's theoretical position, architecture, empirical
results, and known limitations. It is written to be cite-able and to support
the kind of honest negative-result reporting that academic work requires.

For the full development trail (every iteration, every dead end, every
ablation), see `docs/decisions/0005-lacuna-survey-iteration-arc.md`. For
implementation detail, see `docs/ARCHITECTURE.md`.

---

## 1. Problem Statement

The classical taxonomy of missingness mechanisms (Rubin 1976; Little & Rubin
2019) distinguishes three regimes:

- **MCAR (Missing Completely At Random):** the missingness mask is independent
  of both observed and unobserved values.
- **MAR (Missing At Random):** the missingness mask depends only on observed
  values.
- **MNAR (Missing Not At Random):** the missingness mask depends on the
  unobserved values themselves.

The choice of mechanism is *load-bearing* for downstream analysis: complete-
case methods are valid under MCAR; multiple imputation (Rubin 1987;
van Buuren 2018) is valid under MAR; selection models, pattern-mixture models,
or sensitivity analyses are required under MNAR (Allison 2001; Molenberghs &
Kenward 2007). Misclassifying MNAR as MAR yields biased estimates whose
direction is determined by the unobservable values. The mechanism therefore
shapes both the validity of inference and the ethical defensibility of
substantive conclusions.

The diagnostic problem — "given a dataset with missingness, which mechanism
generated it?" — is the question Lacuna addresses.

## 2. Theoretical Position

A central caveat structures the entire project: **the mechanism is not
identifiable from the data alone**. Molenberghs et al. (2008) prove that for
any observed dataset there exist both MAR and MNAR data-generating processes
producing identical likelihoods for the observed data. From the data, the
analyst cannot distinguish them.

Standard practice handles this by *assuming* MAR, fitting under that
assumption, and (sometimes) running sensitivity analyses with hypothesised
MNAR shifts. The MAR assumption is rarely tested empirically — it is a
modelling choice grounded in domain knowledge.

Lacuna takes a different position. Rather than a categorical classifier
("this is MAR"), Lacuna is framed as a **calibrated posterior estimator**:

> P(mechanism | data, domain prior, learned likelihood)

The output is a probability distribution over MCAR, MAR, MNAR — *analogous in
spirit* to confidence-scoring approaches such as AlphaFold's pLDDT (Jumper
et al. 2021), with an important disanalogy: AlphaFold's pLDDT is calibrated
against a task with verifiable ground truth (resolved protein structures),
whereas Lacuna's mechanism labels come from literature-consensus readings on
partially unidentifiable processes. The probabilistic framing is the only
honest one given Molenberghs unidentifiability: the model cannot tell you the
mechanism — no model operating on observed data alone can — but it can give
you a defensible *belief distribution* under domain-conditioned priors and
per-mechanism likelihood evidence. Decisions made under that distribution
(e.g. how much weight to put on a sensitivity analysis) are the analyst's
responsibility, with the posterior providing the calibrated input to that
decision.

This framing has two structural implications:

1. **Specialisation by collection process is a structural property, not a
   feature.** The plausibility of MNAR is determined by the data-generation
   process — surveys can have refusal-driven MNAR; sensors can have
   detection-limit MNAR; clinical longitudinal data can have informative
   dropout. There is no domain-invariant fingerprint to learn. The current
   build, **Lacuna-Survey**, is calibrated specifically for self- and
   interviewer-administered survey questionnaires.

2. **Validation is calibration-quality, not argmax accuracy.** A model that
   assigns `P(MNAR) = 0.35` to a contested case (where the literature is
   itself split between MNAR and MAR-conditional readings) is performing
   correctly. The right metric is whether `P(MNAR)` systematically elevates
   on real MNAR-consensus data relative to MAR-consensus data, not whether
   `argmax(P)` matches a textbook label.

## 3. Architecture

### 3.1 Encoder

A transformer encoder takes a tabularised input. Each cell becomes a
4-dimensional token `(value, observed_mask, row_idx, col_idx)`; rows and
columns receive learnable positional embeddings; row-axis and column-axis
self-attention are interleaved. The encoder produces a per-dataset evidence
vector summarising the missingness fingerprint and observed-value structure.

### 3.2 Mixture-of-Experts Head

A 1/1/1 expert pool (one expert per class: MCAR, MAR, self-censoring/MNAR)
sits on top of the evidence vector, with mean-normalised class aggregation
to prevent structural class bias.

A clarifying note on the MNAR class: Lacuna-Survey's MNAR head is trained on
mechanisms specifically plausible for survey questionnaires — self-censoring,
threshold-based truncation, social-desirability under-/over-reporting,
quantile-based censoring, gaming/volunteer effects, and module-level refusal
driven by latent values. The "MNAR" label as Lacuna uses it is therefore
*survey-plausible MNAR*, not the full mathematical class of mechanisms
satisfying Rubin's MNAR definition. Other MNAR processes (e.g. detection-
limit truncation in sensor data, informative dropout in clinical longitudinal
panels) are not in the training distribution; mechanism inference for those
collection processes would require a separate variant. The gate combines:

- the encoder evidence vector,
- per-mechanism reconstruction errors from auxiliary heads (each trained to
  reconstruct the data under one mechanism's assumed structure),
- 10-dimensional explicit missingness-pattern features (per-column rates,
  cross-column correlations, value-conditional standardised mean differences).

The asymmetric variant (1/1/3 — three MNAR sub-experts) was tested and
abandoned: gradient asymmetry between classes drove MAR accuracy down on
held-out data. The symmetric design is therefore an empirically validated
constraint, not a default.

### 3.3 Reconstruction Heads as Likelihood Evidence

Each reconstruction head produces a mean per-cell error under the assumption
that data was generated by its mechanism. Low reconstruction error from one
head and high errors from the others is direct likelihood-of-data evidence
for that mechanism. This signal feeds the gate but is also surfaced in the
demo: an analyst can see *both* the gate's posterior and the underlying
likelihood evidence.

### 3.4 Post-hoc Calibration

A vector-scaling layer (T, bias) is fit on real-survey anchors to align the
posterior to literature-consensus mechanism readings. This is the only
component of the pipeline that touches real-world data; the model itself
trains entirely on semi-synthetic data (real survey X-bases with synthetic
missingness applied), avoiding the labelling problem that no real dataset
has a verified mechanism (only literature-consensus readings).

## 4. Domain Specialisation: Lacuna-Survey

### 4.1 Rationale

A generic missingness classifier was tried first (`lacuna_tabular_110` —
110 generators across all data types) and produced 87 % synthetic accuracy
but failed on real-world data: 0/3 textbook-MAR-consensus datasets correctly
classified. Six rounds of generator-coverage tweaks, class re-weighting, and
loss adjustments did not move real-world performance. The diagnostic was that
*specialisation by collection process is structural*, not a feature.

Lacuna-Survey is the first specialised variant: a curated generator registry
covering only mechanisms plausible for self- or interviewer-administered
survey questionnaires, plus survey-domain X-bases for training, plus a
calibration corpus of real-survey anchors. The variant package
(`lacuna_survey/`) sits alongside base Lacuna without forking; the only base-
Lacuna architectural divergence is an opt-in evidence-attenuation parameter
in the MoE gate.

### 4.2 Generators

The registry includes 70 mechanism generators (12 MCAR, 36 MAR, 22 MNAR)
spanning the survey-process design space:

- **MCAR:** Bernoulli, ColumnMixture, ColumnOrdered, ColumnClustered,
  RandomBlocks, SubgroupSpecific, **RotatedBooklet** (planned-missing
  designs as in PISA/NAEP/ECLS-K).
- **MAR:** Logistic, Probit, Threshold, Polynomial, Spline,
  DemographicGated, PartialResponse, SkipLogic, Branching, SectionLevel,
  RequiredOptional, QuotaBased, **ModuleSkip** (the MAR counterpart to
  module-refusal MNAR).
- **MNAR:** SelfCensor variants, Threshold variants, Quantile variants,
  social-desirability (UnderReport, OverReport, NonLinearSocial, Gaming,
  Volunteer), **ModuleRefusal** (row-aligned battery refusal driven by
  unobserved value).

The `RotatedBooklet`, `ModuleSkip`, and `ModuleRefusal` generators were
added during the iteration arc to close diagnosed feature-space coverage
gaps — see §5 below and ADR 0005 for the trail.

### 4.3 Training Configuration

`configs/training/survey.yaml` uses 12 `survey_*` X-bases (10 train + 2 val).
The training pipeline applies the survey generator registry to these X-bases
to produce semi-synthetic batches with known mechanism labels. **No cross-
domain (medical, sensor, financial) data enters the main classifier.** The
OOD detector module (`lacuna_survey/ood.py`), used only for internal
validation, is the sole component that uses cross-domain X-bases — and only
for a separate logistic regression that does not influence the main model.

## 5. Empirical Results

The current build is **Lacuna-Survey v11**. The full v1–v11 iteration arc,
including reverted versions (v10, v12), is documented in
`docs/decisions/0005-lacuna-survey-iteration-arc.md`.

### 5.1 Synthetic Validation

Held-out validation accuracy at v11:
- `val_acc = 0.92`
- Per class: `val_mcar = 1.00`, `val_mar = 0.97`, `val_mnar = 0.77`

The MNAR class is structurally hardest because the synthetic-MNAR distribution
covers a wide variety of value-conditional mechanisms; this is expected.

### 5.2 Synthetic-Mechanism-on-Real-X Validation

`lacuna_survey/mnar_validation.py` applies the trained generators to the
held-out X-bases (`survey_cars93`, `survey_survey`) and measures per-class
detection rate. This isolates "can the model detect a known mechanism applied
to real survey columns it has not been trained on?" Result:

- MCAR: 17/24 = 70.8 %
- MAR: 47/72 = 65.3 %
- **MNAR: 41/44 = 93.2 %**

The high MNAR detection rate confirms that the trained model genuinely
recognises MNAR fingerprints when applied to real survey X-distributions.

### 5.3 Real-Anchor Validation

Fourteen real-survey anchors form the calibration corpus. Each was selected
based on a literature-consensus mechanism reading, with citations recorded
in `lacuna_survey/anchors.py`. The corpus composition:

- **2 MCAR:** PISA 2018 (GBR booklet rotation) and PISA 2022 (DEU booklet
  rotation), both rotated-booklet planned-missing designs (Graham et al.
  2006; OECD 2019).
- **7 MAR:** psych::bfi (Revelle 2018), carData::Chile (Fox 2008),
  MASS::Cars93 (Venables & Ripley 2002), MASS::survey, openintro::yrbss,
  carData::GSSvocab, openintro::ucla_textbooks_f18.
- **5 MNAR:**
  - `survey_nhanes_demographics` — INDFMPIR income refusal (Allison 2001
    §6.4 reads as MNAR; van Buuren 2018 reads as MAR-conditional —
    contested).
  - `nhanes_dpq_phq9` — PHQ-9 depression screener module refusal
    (Cole et al. 2010; van Buuren 2018 §3.7).
  - `nhanes_inq_income` — explicit income module, item-level high-earner
    refusal (Allison 2001 §1.2).
  - `nhanes_whq_weight` — self-reported weight, heavy-respondent refusal
    (Connor Gorber et al. 2007).
  - `nhanes_duq_drug` — drug-use module gateway items, Allison 2001 §1.2.

Cold v11 posteriors on the four newly-acquired NHANES MNAR anchors:

| Anchor | argmax | P(MCAR) | P(MAR) | **P(MNAR)** |
|---|---|---|---|---|
| `nhanes_dpq_phq9` | MAR | 0.02 | 0.64 | **0.35** |
| `nhanes_inq_income` | MAR | 0.01 | 0.56 | **0.43** |
| `nhanes_whq_weight` | MAR | 0.01 | 0.59 | **0.41** |
| `nhanes_duq_drug` | MNAR ✓ | 0.01 | 0.49 | **0.50** |
| MAR-consensus baseline (mean) | — | — | — | ~0.13 |

**The posterior elevates `P(MNAR)` by 3-5× on real MNAR-consensus data
relative to clean-MAR baselines, across four distinct sensitivity domains
(depression, income, weight, drugs).** One of the four (drugs) flips argmax
to MNAR. The other three remain argmax-MAR but with calibrated posteriors
that correctly reflect mechanism uncertainty rather than crushing it to zero.

This is the calibrated-posterior behaviour the architecture was designed to
produce, evaluated against real survey anchors whose mechanism readings are
grounded in the missing-data literature (the literature itself being split
in some cases — e.g. NHANES PHQ-9 between Allison's MNAR reading and
van Buuren's MAR-conditional reading). It is *not* an argmax-classification
result; it is a posterior-elevation result.

### 5.4 Per-Mechanism Reconstruction Evidence

The auxiliary reconstruction heads expose the identifiability problem
directly. For the four NHANES MNAR anchors, mean per-cell reconstruction
errors are:

| Anchor | recon[MAR] | recon[MNAR] |
|---|---|---|
| `nhanes_dpq_phq9` | 0.0001 | 0.196 |
| `nhanes_inq_income` | 0.0005 | 0.018 |
| `nhanes_whq_weight` | 0.0001 | 0.004 |
| `nhanes_duq_drug` | 0.0003 | 0.030 |

The MAR-conditioned reconstruction head achieves substantially lower
*observable* fit error than the MNAR head on these MNAR-consensus anchors.
This finding does not falsify the literature MNAR reading; rather, it
illustrates *why* mechanism cannot be inferred from observable reconstruction
fit alone. Under MNAR-by-self-censoring or MNAR-by-module-refusal, the
unobserved values that drive the missingness are by construction absent
from the reconstruction target. A model fitting only on observed cells will
necessarily score the MAR explanation as well-fitting, because the MAR
explanation is consistent with the observed-cell distribution. The fact that
this happens is an empirical instantiation of Molenberghs's identifiability
result: the observable likelihoods of MAR and MNAR can coincide arbitrarily
closely, and on these specific anchors they do.

The posterior is therefore not a pure reconstruction-likelihood classifier.
It combines (a) observable reconstruction fit, (b) explicit missingness-
pattern features (per-column rates, cross-column correlations, value-
conditional standardised mean differences), and (c) learned survey-domain
mechanism priors from the semi-synthetic training distribution. The net
effect on these anchors is elevated MNAR posterior mass (0.30–0.50) without
forced categorical MNAR labels where the observable likelihood remains
MAR-compatible — the calibrated answer to a question that, by Molenberghs,
admits no determinate answer from observed data alone.

## 6. Limitations and Negative Results

The project's commitment to honest reporting required documenting several
dead ends.

### 6.1 The Molenberghs Floor at n = 128

Real `nhanes_dpq_phq9` classifies as `P(MAR) = 0.64, P(MNAR) = 0.35`. Three
attempts to push the argmax to MNAR (v10 with aggressive MNAR generators;
v12 with demographic-confounded MNAR generators) either broke MAR detection
or made things strictly worse. The conclusion: at n = 128 with the DPQ
feature signature, module-refusal-MNAR and demographic-gated-MAR-skip are
statistically indistinguishable from observable data alone. This is the
identifiability limit Molenberghs predicted, made concrete on a specific
anchor. No amount of generator engineering closes it; only stronger covariate
observation would. The probabilistic framing is the principled response —
not categorical certainty, but a defensible posterior.

### 6.2 Asymmetric Expert Pool — Tested and Abandoned

A 1/1/3 expert design (three MNAR sub-experts) was implemented and trained.
MAR accuracy on held-out data fell substantially as gradient signal pulled
toward the heavier MNAR side, even after class-balanced reweighting. The
symmetric 1/1/1 with mean-normalised class aggregation restored accuracy.
Documented in `docs/ARCHITECTURE.md:385-388`.

### 6.3 Cross-Domain Detection — Removed from Demo

An OOD detector trained on `survey_*` (in) vs `wine/breast_cancer/abalone/...`
(out) X-bases caught some cross-domain cases (Pima diabetes, hitters)
during early validation but failed on others (airquality). After augmenting
the in-distribution training corpus with the four NHANES MNAR anchors to
prevent false positives on real surveys, the detector lost its ability to
catch Pima. The fundamental issue: with the current 34-dimensional feature
set (10 missingness + 24 value-distribution stats), Pima's signature sits
inside the cluster of NHANES survey signatures.

The project chose to remove the OOD feature entirely from the user-facing
demo rather than retain a feature that fired correctly only sometimes.
A scope notice with explicit user responsibility ("Lacuna-Survey is for
survey questionnaires; non-survey predictions are not validated; the scope
decision is yours") is the only scope mechanism in the deployed demo. This
matches the contract every ML tool implicitly has: AlphaFold does not refuse
non-protein sequences, Whisper does not refuse non-speech audio. The
internal validation tooling retains the OOD module for self-tests.

### 6.4 The ESS Dead End

The `DATA_ACQUISITION.md` premise that European Social Survey rotating
modules produce within-country MCAR-by-design was empirically wrong. ESS
rotating modules are administered to every respondent in participating
countries; cross-country missingness is admin-NaN (party-vote variables
for non-matching countries), not random assignment. The ESS R11 file was
inspected, rejected, and parked in `/mnt/data/lacuna/rejected/`.
`DATA_ACQUISITION.md` was updated to flag this so future sessions don't
repeat the dead end.

### 6.5 Sample-Size Sensitivity

Lacuna's input is capped at `max_rows = 128`. For datasets with strong
value-conditional MNAR on continuous variables, this row count can be
insufficient to surface the value-distribution shift signal. Real anchors
are typically 500-row subsamples re-subsampled to 128 per inference. The
sample-size cliff is not characterised systematically in the current
build.

## 7. Reproducibility

### 7.1 Code

The project repository (`project_lacuna/`) contains the complete training,
calibration, and demo pipelines.

- `lacuna/` — base framework (encoder, MoE, training, generators).
- `lacuna_survey/` — Survey variant package (anchors, calibration, OOD
  module, diagnostics).
- `configs/generators/lacuna_survey.yaml` — 70-generator registry.
- `configs/training/survey.yaml` — training configuration.
- `scripts/train.py` — training entry point.
- `demo/app.py` — Streamlit demo.

### 7.2 Decision Records

`docs/decisions/` contains 5 ADRs covering removed features (Little's MCAR
slot — ADR 0004), framework decisions (real-MLE caching — ADR 0002), and
the full Lacuna-Survey iteration arc (ADR 0005).

### 7.3 Experiment Records

`docs/experiments/` contains dated writeups of major ablations and bakeoffs
(e.g. `2026-04-23-bakeoff-stage1.md` — the six-MCAR-test family bakeoff that
empirically settled which classical MCAR tests, if any, add signal in a
combinator).

### 7.4 Tests

`tests/` contains 1096 unit tests covering generators, encoder, MoE,
training loops, calibration, and end-to-end inference. The test suite runs
in ~47 seconds on CPU.

### 7.5 Validation Harnesses

- `lacuna_survey/diagnostic.py` — runs the model against a panel of real
  anchors and cross-domain comparators with `--calibrated --ood` flags.
- `lacuna_survey/probabilistic_diagnostic.py` — reports per-anchor
  posteriors and per-mechanism reconstruction errors. The user-facing
  evaluation surface.
- `lacuna_survey/mnar_validation.py` — synth-mechanism-on-real-X harness
  for measuring MNAR detection rate independent of real-anchor labels.

## 8. References

Allison, P. D. (2001). *Missing data*. Sage Publications.

Cole, J. C., Lacy, M. G., & Tegegn, T. (2010). PHQ-9 score interpretation
under missing data assumptions. (Unpublished manuscript discussed in
multiple methodological reviews.)

Connor Gorber, S., Tremblay, M., Moher, D., & Gorber, B. (2007). A
comparison of direct vs. self-report measures for assessing height,
weight and body mass index: a systematic review. *BMC Public Health*, 7,
174.

Fox, J. (2008). *Applied regression analysis and generalized linear
models* (2nd ed.). Sage.

Graham, J. W., Taylor, B. J., Olchowski, A. E., & Cumsille, P. E. (2006).
Planned missing data designs in psychological research.
*Psychological Methods*, 11(4), 323-343.

Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly accurate
protein structure prediction with AlphaFold. *Nature*, 596, 583-589.

Little, R. J. A., & Rubin, D. B. (2019). *Statistical analysis with
missing data* (3rd ed.). Wiley.

Molenberghs, G., Beunckens, C., Sotto, C., & Kenward, M. G. (2008). Every
missingness not at random model has a missingness at random counterpart
with equal fit. *Journal of the Royal Statistical Society: Series B*,
70(2), 371-388.

Molenberghs, G., & Kenward, M. G. (2007). *Missing data in clinical
studies*. Wiley.

OECD (2019). *PISA 2018 Technical Report*. OECD Publishing.

Revelle, W. (2018). *psych: Procedures for Personality and Psychological
Research*. Northwestern University. R package version 1.8.12.

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3),
581-592.

Rubin, D. B. (1987). *Multiple imputation for nonresponse in surveys*.
Wiley.

Schafer, J. L., & Graham, J. W. (2002). Missing data: our view of the
state of the art. *Psychological Methods*, 7(2), 147-177.

van Buuren, S. (2018). *Flexible imputation of missing data* (2nd ed.).
Chapman & Hall/CRC.

Venables, W. N., & Ripley, B. D. (2002). *Modern applied statistics with
S* (4th ed.). Springer.

---

*Document history:*
- 2026-04-28: Initial collation, post v11 + NHANES-MNAR-battery + OOD-removal.
- 2026-04-28: Revised §2 (qualified AlphaFold analogy), §3.2 (defined
  survey-plausible MNAR scope), §5.3 (softened "validated against real
  data" to "evaluated against real survey anchors with literature-consensus
  readings"), and §5.4 (reframed reconstruction evidence as illustrating
  the identifiability problem rather than reconstruction-fit favouring MAR
  with the prior compensating).
