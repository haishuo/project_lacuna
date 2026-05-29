# 0006 — Column-level missingness-mechanism classification (experiment pre-registration)

- **Date:** 2026-05-29
- **Status:** Accepted as experiment design / pre-registration — implementation and results pending
- **Predecessors:** ADR 0005 (domain-specialization arc)
- **Control arm:** git tag `v1.0-canonical`, frozen at commit `ebae77a` (the commit immediately preceding this record). Canonical checkpoint RUN-054.

This ADR is a **pre-registration**, not a retrospective. It records the design we commit
to *before* running the experiment, so that whatever the outcome — improvement, no change,
or regression — the result is attributable rather than the product of post-hoc rationalization.
Per the folder convention, this record is immutable; if results force a redesign, a later
ADR supersedes it.

## Context

Lacuna today emits a single **dataset-level** posterior: `PosteriorResult.p_class` of shape
`[B, 3]`, one probability vector over {MCAR, MAR, MNAR} per dataset. The encoder pools the
entire data matrix into one 64-d evidence vector (`encoder.py`: row pooling collapses
columns, dataset pooling collapses rows), and every training instance carries exactly one
label, because `apply_missingness` applies one generator to the whole matrix
(`semisynthetic.py`). The model is therefore trained and structured to answer one question:
*"assume this dataset was produced by a single mechanism — which one?"*

Two things motivate going finer-grained:

1. **Most real datasets are mechanism mixtures.** A survey rarely has one mechanism: a
   sensitive item is MNAR-suspect, an income item shows the usual covariate-driven refusal
   (MAR), and a block of items has random administrative dropout (MCAR). A single
   dataset-wide label is too coarse to be either accurate or actionable.

2. **Column-level is more useful and more scientific.** Analysts make imputation decisions
   per variable, not per dataset. And a tool that reports *which* columns are MNAR — and
   estimates how mixed real data actually is — is a more informative scientific instrument
   than one that collapses everything to a single label.

## The distinction this experiment is built on

The current output conflates two genuinely different estimands, and the conflation is the
whole reason for this work:

- **Q1 — dataset-label confidence:** P(the dataset's single mechanism label is MNAR). A
  *belief over one categorical*. Goes to 0/1 with more data. **This is what `p_class` is today.**
- **Q2 — composition:** what *fraction* of the dataset is MNAR. A *proportion*. Does **not**
  collapse with more data.
- **Q3 — per-unit label:** the mechanism (with confidence) of each *column*.

These are not the same axis. A dataset that is unambiguously 50% MAR columns / 50% MNAR
columns, and a dataset where the model is genuinely 50/50 *unsure* whether the whole thing
is MAR or MNAR, **print identically today** (`MAR 50% / MNAR 50%`) and demand opposite
actions. The current architecture cannot tell them apart.

The structural insight: **Q3 is atomic; Q2 and Q1 both derive from it.** Given calibrated
per-column posteriors, the composition is the aggregate (`E[#MNAR cols] = Σ_j P(col_j = MNAR)`)
and a dataset headline is a *decision rule* over the column labels. So the real decision is
not "add a composition number" but "make the **column** the unit of inference."

## Scope

- **IN — column as the atomic inference unit.** General (works for all tabular data, not just
  surveys), granular, and the finest unit at which mechanism is still a coherent
  population-level estimand.
- **LAYER (deferred) — block as a prior/grouping, not a primitive.** Surveys are not all
  block-by-design; block metadata usually does not survive into the data matrix; and blocks
  can be internally mechanism-heterogeneous. Aggregating columns → block is free;
  disaggregating is impossible — so we build at the column and treat blocks, when present
  (analyst-supplied or detected), as (a) a hierarchical/partial-pooling prior that can buy
  back the precision lost to granularity, and (b) a signal source (skip-logic =
  missing-by-design = known MAR). Block modeling is out of scope for the first stages and
  enters as an explicit later dial.
- **OUT — cell-level.** A single cell does not *have* a mechanism: MCAR/MAR/MNAR are
  properties of the distribution `P(R | X)`, not of an individual draw. The
  "99-ashamed-1-tired" intuition is a *within-column mixture* whose estimand is a proportion,
  and that proportion is not identifiable even in principle (two missing cells with identical
  observed covariates are exchangeable; the distinguishing information is exactly what is not
  in the data). The honest surrogate is a **low-confidence / "ambiguous" column** surfaced by
  a calibrated per-column posterior. Within-column mixture estimation, if ever pursued, is a
  separate research project with its own assumptions.

## Design commitments (pre-registered)

1. **Joint encoder, per-column readout.** A column *cannot* be classified in isolation — MAR
   is defined by dependence on *other* columns, and the team already rejected per-column
   *encoding* for this reason (`docs/notebooklm/02_architecture.md`). The classifier must see
   the whole matrix and emit per-column labels. Architecturally this is a small reach: the
   encoder already retains per-column representations before pooling
   (`encoder.get_token_representations` → `[B, R, C, H]`), and the explicit features in
   `missingness_features.py` already compute per-column / per-pair quantities before
   aggregating them away. The per-column head pools over **rows** for each column instead of
   collapsing columns.

2. **Multi-task with the existing dataset-level head.** Keep the validated dataset-level
   output; add the per-column head alongside it; optionally add a consistency term (the
   dataset label should agree with an aggregation of the column labels). This preserves the
   v1.0 capability and lets the heads regularize each other.

3. **Mixed-mechanism semi-synthetic generation with per-column labels.** Replace the
   one-generator-per-dataset path with one that partitions columns into mechanism groups,
   applies a (possibly different) generator per group, and records per-column `class_id`s.
   This is the substantive change and the main risk surface — it shifts the training
   distribution, and this project has hard evidence that the training distribution is delicate
   (the 1/1/3 → 1/1/1 expert change; the class-balanced-prior backfire).

4. **MAR-predictor policy — clean first, entangled as ablation.** *(The key fork. Signed off
   2026-05-29.)* MCAR (independent) and MNAR-self-censoring (depends on the column's own
   latent value) are cleanly per-column. MAR is not: a MAR column's missingness depends on
   *other columns' observed values*, and the choice of which columns defines what
   "column-level MAR" even means.
   - **Clean regime:** MAR predictors point only at fully-observed / MCAR columns →
     crisp, honest labels. **Built first**, so the inaugural result is interpretable.
   - **Entangled regime:** MAR predictors may themselves be missing → the predictor is often
     unobserved, the relationship muddies toward empirically-MNAR-looking, and the "MAR"
     label becomes a partial mislabel. This is what real mixed data looks like. **Run as an
     ablation**: "how robust is per-column classification when predictors are themselves
     missing?" is a finding in either direction.
   Building entangled-only first would confound label noise with model error and produce an
   uninterpretable result — the failure mode this pre-registration exists to prevent.

5. **Mixture-prior dial.** How many MCAR/MAR/MNAR columns per dataset (uniform vs.
   survey-realistic skew) is a second explicit dial, varied one at a time after the clean
   baseline is established.

## Supervision and evaluation

- **Train + supervise: semi-synthetic only.** Per Molenberghs, real missingness mechanisms
  are unidentifiable from observed data, so ground-truth labels can only come from synthetic
  mechanisms applied to real `X`. This is unchanged from v1.0; only the generation becomes
  mixed.

- **Test: three channels, because real data has no per-column ground truth either.**
  1. **Numbers → held-out semi-synthetic** (held-out X-bases × mixed generators). The only
     place per-column accuracy and calibration (per-column ECE) can be measured.
  2. **Face validity → real survey anchors.** Does the model flag known-sensitive items
     (income, drug use, weight) as MNAR-suspect, and respect the identifiability bound?
     Reuses `lacuna_survey/anchors.py` (NHANES + Molenberghs bound), extended per-column.
  3. **Missing-by-design → real-data known-MAR check.** Skip-logic columns (missing exactly
     when a gateway item routes past them) are structurally MAR — the one subset of real
     columns with known mechanism. A necessary-not-sufficient floor on actual survey data.

- **Define "tanked" before running.** Column-level and dataset-level accuracy are **not on
  the same scale** (different task, base rates, number of decisions); "we lost N points" is a
  category error. The pre-registered comparisons are: per-column accuracy vs. a **per-column
  baseline** (a per-column regression of the missingness indicator on observed covariates;
  per-column Little's), and **aggregation consistency** (do aggregated column predictions
  reproduce the v1.0 dataset-level answer?).

## Experiment stages (one variable per stage; v1.0 is the control)

- **Stage 0 — control:** run the frozen `v1.0-canonical` model on mixed-mechanism eval data.
  *Result regardless of anything downstream:* how does a dataset-level model behave on
  mixtures — does its posterior diffuse, collapse to the most-severe mechanism, or to the
  most-common-by-count? Not previously characterized; a finding on its own.
- **Stage 1:** add the per-column head, multi-task with the dataset head, train on mixed data
  in the **clean** regime. Isolates: *can the architecture read out per-column at all?*
- **Stage 2+:** change one dial at a time — entangled regime; mixture-prior skew; block
  partial-pooling prior.

Each stage moves one variable and retains the v1.0 control, so each yields an attributable
finding whether the metric rises or falls.

## Consequences

**Enables:**
- Per-column mechanism diagnostics (Q3) and, derived from them, a calibrated composition
  estimate (Q2) — a novel descriptive instrument ("this dataset is ~7 MAR / ~3 MNAR / ~2
  MCAR columns"), potentially a standalone contribution independent of accuracy.
- A clean separation, in the output contract, of *confidence* (per-column) from *composition*
  (counts over columns) — dissolving the conflation above.

**Forecloses / costs:**
- A meaningful shift in the training distribution; the v1.0 numbers do not transfer directly
  and the model must be re-validated under mixed generation.
- Output-schema change. Per ROADMAP §2.2 (output-schema stability), it is cheaper to settle
  this in research mode now than after a 1.0 lockdown.
- Per-column posteriors will likely be **wider** than the 82.6% dataset-level figure,
  especially at the MAR/MNAR boundary (same non-identifiability, less evidence per decision).
  Under the calibrated-posterior philosophy this is acceptable — but per-column *calibration*,
  not just accuracy, becomes a first-class deliverable.

**Conditions under which we'd revisit / supersede:**
- Stage 1 shows the architecture cannot read out per-column even under the clean regime
  (→ the readout design, not just the data, is wrong).
- The entangled-regime ablation shows label noise dominates (→ reconsider the operationalization).
- A reviewer rejects per-variable ignorability as the operationalization of column-level
  mechanism (see below) — then the definition, and this ADR, need revision.

## For statisticians using this tool

Textbook MCAR/MAR/MNAR are properties of the *joint* missingness model
`P(R | X_obs, X_mis)`, not of an individual column. "Column-level MAR" is operationalized
here as **per-variable ignorability**: *is this variable's missingness ignorable given the
observed data?* That is a defensible and common applied reading, but it **is** a definitional
choice, and it is stated explicitly so a committee member can accept or contest it on the
record rather than discover it implicitly in the code. Cell-level mechanism is disclaimed
above as non-identifiable in principle — not merely hard — and is deliberately out of scope.

## Cross-references

- Control arm: git tag `v1.0-canonical` (commit `ebae77a`); checkpoint RUN-054
  (`lacuna_semisyn_20260329_032848`).
- Per-column representations already in the encoder: `lacuna/models/encoder.py`
  (`get_token_representations`, `get_row_representations`).
- Per-column / per-pair statistics computed then aggregated: `lacuna/data/missingness_features.py`.
- One-generator-per-dataset path to be generalized: `lacuna/data/semisynthetic.py`
  (`apply_missingness`).
- Posterior → class aggregation: `lacuna/models/aggregator.py`.
- Per-column-encoding rejection (distinct from per-column *readout*): `docs/notebooklm/02_architecture.md`.
- Real-data anchors and identifiability bound: `lacuna_survey/anchors.py`.
- Output-schema stability posture: `ROADMAP.md` §2.2.
- Predecessor: `docs/decisions/0005-lacuna-survey-iteration-arc.md`.
