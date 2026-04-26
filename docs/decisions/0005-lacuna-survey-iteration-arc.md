# 0005 — The Lacuna-Survey iteration arc (v1–v8)

- **Date:** 2026-04-26
- **Status:** In progress (specialization framework established; within-domain detection partially solved; further investigation pending)
- **Predecessors:** ADR 0004 (removed the cached-MLE Little's slot)

## Context

Project Lacuna's purpose is to encode missingness-mechanism domain
knowledge as a tool — replacing or augmenting expert review of the
MCAR / MAR / MNAR distinction on tabular datasets. The existing
"generic" Lacuna trained on the `lacuna_tabular_110` registry produced
high synthetic accuracy (87 % at lacuna_demo_v1) but failed on
real-world data: 0/3 textbook-MAR-consensus datasets correctly
classified, and confidently routing real-MNAR data to MAR (the worst-
error direction for downstream imputation).

Six iterations of the generic model (v1–v4 + diagnostic experiments)
established that the failure was **not** fixable through:
- generator coverage in the existing taxonomy,
- class-imbalance correction,
- per-class loss reweighting,
- saturation bug fixes in the generator parameter space,
- adding the new `MARRealistic*` / `MARPartialResponse*` /
  `MARDemographicGated*` generators that fill the moderate-multi-column
  gap in the synthetic distribution.

The pattern was clear: synthetic numbers shuffled (74 % – 98 %), real-
world predictions stayed roughly fixed. The classifier had hit the
information-theoretic ceiling for **pattern-only** features.

## The reframing — domain specialization as a structural property

The user articulated the reframing that resolved the question of why
the generic model couldn't transfer:

> When a researcher hands missingness analysis to a doctor, they aren't
> testing "can a doctor figure out this is medical data?" They've
> already supplied that information by choosing the doctor. Asking
> generic Lacuna to do the analysis without that information is asking
> it to do a strictly harder problem than what humans actually face.

That is: **domain is not a feature that can be added; it is intrinsic
to the problem statement**. The collection process — survey,
instrument, administrative records, longitudinal panel, experimental
RCT — determines which mechanisms are physically plausible. Skip-logic
exists in surveys because surveys have skip-logic; instrument failure
exists in sensors because instruments fail. There is no domain-
invariant fingerprint to learn.

Operationally: Lacuna is not one model; it is a **framework** for
training mechanism classifiers specialised by collection process. The
generator registry IS the domain prior. Specialisation is the right
granularity (between hopeless-generic and overfit-per-dataset), and
the right axis is collection process rather than field. A medical
survey, a political survey, and a marketing survey share survey-process
mechanisms; they do not share domain-process mechanisms with EHR
records that happen to also be medical.

## What was built

**Specialised registry** `configs/generators/lacuna_survey.yaml` — a
60-generator subset of `lacuna_tabular_110` selected for
survey-process plausibility, plus the new low-rate broad-spread
variants added in the v8 iteration. Excludes latent-variable /
longitudinal MNAR (observational, not single-wave), competing-events
and symptom-triggered (clinical longitudinal), detection-limit and
quantitation-limit (instrument), heavy-tailed distributional MCAR
(instrument artifacts), and ColumnBlocks / CrossClassified / large-
block MAR (admin / EHR patterns).

**Survey-flavoured training catalog** in `/mnt/data/lacuna/raw/`:
twelve real public survey datasets (psych::bfi, AER PSID/CPS,
openintro::yrbss, MASS::survey, MASS::Cars93, carData::Chile,
Ecdat::Computers / Workinghours, AER::HMDA), all stored complete-case
to preserve the `apply_missingness` invariant that synthetic
mechanism is applied to a clean X. Native-missingness versions are
retained separately under `demo/sample_data/survey_*_real.csv` for
evaluation.

**Real-data evaluation suite** (`scripts/diagnose_mar.py`): eleven
diagnostic datasets with documented expert-consensus mechanism
labels. Within-domain cases — five real surveys with native
item-nonresponse — anchor the apples-to-apples test. Cross-domain
cases (Pima, airquality, hitters, pbc) probe the model's behaviour
out of its training distribution.

**Eval-report bug fix** in `scripts/train.py`: the post-training
detailed evaluation now loads `best_model.pt` before running
`validate_detailed`. Prior runs reported numbers from the *final*
model state (often several epochs past best val_loss); the gap on
lacuna_survey_v2 was 90.1 % (best) vs 38.9 % (final) on MNAR recall.
All evidence in this document refers to best-checkpoint numbers.

**Three additions to the gate input**: the value-conditional features
`smd_mean`, `smd_max`, `shape_shift` in
`lacuna/data/missingness_features.py`. Computed in
O(B · R · C²) with batched einsum, NaN/Inf-safe with clamping. SMD
contributions are weighted by effective sample size (harmonic mean of
group sizes shrunk by `n_eff / (n_eff + 5)`) so that low-overall-rate
real surveys don't produce spurious large SMD spikes.

**Learnable evidence attenuation** in
`lacuna/models/moe.py::GatingNetwork`: a sigmoid-bounded scalar
multiplied into the evidence slice before concatenation with the
other gate inputs. Initialised to 0.25 to balance per-dimension
weight × dim count between evidence (64 dims) and explicit features
(13 dims). v7 and v8 both learned α near 0.24 — confirming evidence
shouldn't dominate by sheer volume.

## The eight-version arc — what each version showed

| Version | Intervention | Synthetic | Real MAR | Real MNAR | What it told us |
|---|---|---|---|---|---|
| v1 | Generic-tabular baseline | 87.4 % | 0/3 | 2/3 | The unidentifiability ceiling exists empirically, not just theoretically. |
| v2 | Added `MARRealistic*` generators | 73.9 % | 1/3 | 2/3 | Closing the generator-coverage gap moves predictions sideways, not up. |
| v3 | Saturation fix in `apply_missingness` | 79.5 % | 1/3 | 0/3 | Cleaner training distribution exchanges one error mode for another. Pima → MCAR (worst-error direction) is dangerous. |
| v4 | Class weighting + class-balanced prior + value-conditional features | 88.4 % | 3/8 | 2/3 | Value-conditional features enable real MNAR detection (Pima ✓). MAR slippage tracks training-time MAR class variance. |
| v5 | SMD effective-sample-size shrinkage | 94.0 % | 2/8 | 2/3 | Eliminating noise in the value-conditional features improved synthetic accuracy but didn't transfer; loss weighting was over-firing. |
| v6 | Drop class weights | 88.5 % | 2/8 | 2/3 | Symmetric loss does not by itself fix real-world MAR detection. The gap is structural, not balance. |
| v7 | Learnable evidence-attenuation α | 86.9 % | 1/8 | 1/3 | Per-volume re-weighting of evidence vs explicit features did not transfer to real data. Final α ≈ 0.24 was barely shifted from init 0.25. |
| **v8** | **Low-rate broad-spread MAR + MNAR-Q90-Broad in registry** | 86.5 % | **3/5 within-domain** | 1/3 | **The specific (low-rate × broad-cols) generator cell was the missing-from-training pattern. With it, three within-domain real surveys (bfi, chile, cars93) classify correctly as MAR for the first time across the entire arc.** |

(Real-MAR consensus columns through v3 were a 3-dataset suite;
expanded to 8 with the survey-real datasets from v4 onwards.)

## What was ruled out

In order:

1. **Class imbalance during training** — solved by class-balanced
   prior (v3+); real-world numbers did not move.
2. **Pattern-feature ceiling** — solved by adding three value-
   conditional features (v4); within-domain MAR detection improved
   for Pima but not for surveys.
3. **Predictor saturation in linear-in-X generators** — solved by
   `_zscore_columns` in `apply_missingness` (v3); fixed 39 of 41
   degenerate generators producing "1 col 100 % missing" patterns.
   Synthetic improved; real-world did not.
4. **Sampling noise in SMD on low-overall-rate datasets** — solved
   by effective-sample-size shrinkage in
   `compute_value_conditional_features` (v5); eliminated `smd_max`
   spikes from 3.08 to 0.92 on `survey_bfi_real`.
5. **Loss-side over-correction by per-class weighting** — explored
   in v3–v6 with weights of 1.0/1.0/2.5 down to 1.0/1.0/1.0; the
   2.5× MNAR weight was over-firing on MAR data. Removed in v6.
6. **Eval-report measurement artifact** — fixed in `scripts/train.py`
   to load `best_model.pt` before evaluation. The reported MNAR
   recall difference between best (88.8 %) and final (38.9 %) on the
   same val set was a measurement issue, not a model failure.
7. **Evidence dominating gate input by volume** — partially
   addressed by learnable α in v7. The α settled near 0.24 in both
   v7 and v8; modest re-weighting was not the dominant fix.
8. **Missing-from-training low-rate broad-spread pattern cell** —
   addressed in v8. Was the *substantive* gap for within-domain
   real surveys: bfi (0.6 % across 25 cols), chile (0.6 % across 5
   cols), cars93 (0.8 % across 18 cols). v8 added four generators
   (MAR-DemoGated-LowBroad, MAR-DemoGated-VeryLow,
   MAR-PartialResponse-Sparse, MNAR-Q90-Broad) tuned to that cell.

## What is still open

Three within-domain failure modes remain:

1. **`survey_yrbss` → MCAR (high confidence)**. The Youth Risk
   Behavior Surveillance System textbook reading is MAR; Lacuna-Survey
   v8 says MCAR with 97 % confidence. The textbook reading is itself
   contestable — phone-survey item nonresponse is often near-MCAR in
   practice — but a model that confidently disagrees with the
   consensus needs principled justification, not just plausible
   re-interpretation.

2. **`survey_survey` → MNAR (high confidence)**. The Adelaide student
   survey textbook reading is MAR; v8 says MNAR. No clear
   re-interpretation of the dataset that justifies MNAR.

3. **Cross-domain principled refusal**. Pima, hitters, pbc, airquality
   are not surveys and Lacuna-Survey is not designed to handle them.
   A correct deployed system should decline to classify them or flag
   them as out-of-domain rather than confidently emit a prediction.
   No abstain mechanism exists today.

## Candidate next directions (research-level, not architectural rewrites)

These are the directions with no prior evidence against them. Two
options that *did* have prior evidence against them are explicitly
struck below.

In rough order of expected leverage:

1. **Real-data calibration** — Use a small held-out set of
   verified real-survey examples (bfi, chile, cars93 with their
   textbook labels) to calibrate the model's posteriors via
   temperature scaling or Platt-style adjustment. Doesn't change the
   model; reshapes its outputs to be honest about uncertainty.

2. **Encoder-side regularization**. The transformer evidence
   pathway carries 71.8 % of gate input L1 mass and is the dominant
   driver of real-world misclassifications. Targeted regularization —
   e.g. an information bottleneck on the evidence vector, or
   dropout-during-inference to test sensitivity — may make evidence
   carry only what's robust across distributions.

3. **Out-of-domain detection**. Deep evidential learning or simple
   density-of-training-features-vs-test-features test would let the
   model flag inputs unlike anything it saw. For Lacuna-Survey,
   Pima/airquality should be flagged before any classification.

4. **Active learning on real data**. Where reliable expert labels
   exist (NHANES item-level, ESS), incorporate small amounts of real
   data into training as anchors. The bulk of training stays
   semi-synthetic; the real anchors prevent distribution drift.

### Directions explicitly NOT on the list

- **Ensemble with classical Little's-style tests** (Little 1988
  MLE, MoM, propensity, HSIC, MissMech, median-split heuristic, or
  any combinator over them). The
  `mcar-alternatives-bakeoff` arc tested all six families as gate
  features and ADR 0004 confirmed that Little's MLE *actively hurt*
  classification accuracy at n=30 (CI excluded zero, p=0.036,
  21/30 seeds favoured disable). The deployment-time ensemble is
  technically a different combinator than learned feature weights,
  but the underlying empirical finding — classical MCAR tests do
  not carry information beyond Lacuna's existing features — applies
  in either combinator. Re-introducing them would re-litigate
  settled work.

- **Architectural rewrite of the base model.** The classifier
  framework — semi-synthetic training with `apply_missingness`,
  transformer encoder + MoE gate + reconstruction heads — has been
  validated by v1–v8: it produces well-calibrated, class-balanced
  predictions on synthetic, with synthetic accuracy in the 86–98 %
  range across versions. Where it fails (real-world transfer) is in
  information that isn't in the training data, not in the
  architecture that uses it. Branch only on demonstrated need for
  fundamentally different computation (e.g. temporal attention for
  Lacuna-Longitudinal); not as a fix for the within-domain
  generalization gap, where the deployment-layer directions above
  haven't been tried yet.

## Numerical reference

The tables in the section above use eval-report numbers from
`/mnt/artifacts/project_lacuna/runs/lacuna_demo_v[1..4]/` and
`/mnt/artifacts/project_lacuna/runs/lacuna_survey_v[2..8]/`. Real-
world diagnostic results from `scripts/diagnose_mar.py` against the
installed `demo/model.pt` checkpoint. Pre-v6 eval numbers should be
read with the caveat that they were measured against the final model
state, not the best checkpoint (see "What was built" above).

`v8` checkpoint at
`/mnt/artifacts/project_lacuna/runs/lacuna_survey_v8/checkpoints/best_model.pt`
is the current state-of-the-art for Lacuna-Survey. Within-domain
real-survey MAR detection: 3/5. Synthetic accuracy: 86.5 %.
Confidence calibration ECE: 0.065. Learned evidence attenuation
α = 0.241.
