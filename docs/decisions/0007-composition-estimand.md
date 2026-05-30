# 0007 — Composition-posterior estimand for missingness mechanisms (experiment pre-registration)

- **Date:** 2026-05-30
- **Status:** Accepted as experiment design / pre-registration — implementation and results pending
- **Predecessors:** ADR 0006 (column-level missingness); the Stage 0–5 column-level arc
- **Supersedes:** ADR-0006's commitment to the **per-column label as the atomic inference unit.**
  It retains 0006's Q1/Q2/Q3 framing but **inverts which estimand is primitive** (see below).

This ADR is a **pre-registration**, not a retrospective. It records the design we commit to
*before* running, so that whatever the outcome — a confident composition, a wide honest band, or a
null — the result is attributable rather than post-hoc. Per the folder convention this record is
immutable; if results force a redesign, a later ADR supersedes it.

## Context — what the column-level arc taught us

ADR-0006 made the per-column mechanism label the atomic primitive, from which composition (Q2) and
a dataset headline (Q1) were to be *derived by aggregation*. Five stages established:

- Per-column MCAR/MAR is recoverable; per-column MNAR on mixtures is hard (Stages 0–3).
- Detectability stratifies sharply by subtype in the **single-mechanism** setting — threshold /
  detection-limit MNAR at 0.9–1.0, self-censoring far lower (Stage 4). **Pillar 1 — not all
  mechanisms are equally unidentifiable — holds, and is untouched by what follows.**
- Diverse mixtures *appeared* to stabilise per-column classification (Stage 4b) — but Stage 5 showed
  that stability was **substantially a miss-rate confound.** The composer's direct logistic paths
  overshoot the target rate (~0.30 vs MCAR's exact 0.25), and `missing_rate` is a model input, so
  the head separated classes on *rate*, not *mechanism*. With miss rate held truly constant, per-
  column hard classification is **seed-unstable across every diversity regime** (winner-take-all
  collapse; see `docs/experiments/2026-05-29-stage5-true-mixtures.md`).

The lesson is **not** that Lacuna fails — it is that **per-column hard classification was the wrong
primitive.** The project's actual goal was never per-column labels; it was *"given the data as it
exists, what is the likelihood of this data being MAR or MNAR"* — a **composition with honest
confidence**, *founded on* Molenberghs, not beating it. Three load-bearing premises remain intact:
(1) realistic mechanisms are not equally unidentifiable; (2) real missingness lives on a small
*manifold* of mechanisms that actually occur, on which fingerprints exist; (3) semi-synthetic gives
the only ground truth against which we can check ourselves.

## The estimand (what this ADR commits to)

Lacuna's output is a **calibrated probability distribution over the dataset's missingness
composition** — the mixing fractions `(f_MCAR, f_MAR, f_MNAR)` on the 2-simplex — together with an
explicit **"can't-tell" mass.** Both of these are then *queries* on that distribution, not separate
outputs:

- "60% confident at least 70% is MAR" = the posterior puts 0.60 of its mass on `{f_MAR ≥ 0.7}`.
- "80% confident it's ~30/20/50, the rest ambiguous" = a tight mode near `(0.3, 0.2)` plus an
  explicit ambiguous remainder.

This **dissolves the Q1/Q2 conflation** of ADR-0006: a point estimate (`20/70/10`) cannot tell
*confident* from *clueless*; a distribution can (tight blob = sure, smeared = honestly unsure). So
**Q2 (composition) becomes the directly-estimated primitive; Q1 (dataset headline) is a query on it;
Q3 (per-column "where") is demoted to an optional, downstream, calibrated-suspicion layer — NOT the
primitive.** That inversion is the substance of this ADR.

## Design commitments (pre-registered)

1. **The generator is the operational definition of the estimand.** "30% MNAR" means what it means
   *because the generator produces it and tags it so.* Therefore the generator's target, the ground-
   truth bookkeeping, and the eval metric **must share one denominator.** (Stage 5's confound was a
   denominator mismatch; this rail forecloses that whole class of error by construction.)

2. **Denominator: by missing cell. [LOCKED]** The composition is the fraction of *missing cells*
   attributable to each mechanism. Well-defined, generatively known, and weighted by how much data
   each mechanism actually touches — the decision-relevant unit for "should I run a sensitivity
   analysis." (Exposed-inference weighting is a deferred v2 requiring a committed reference analysis.)

3. **Re-admit joint mechanisms, with a cell-tagging convention. [LOCKED]** Real missingness is often
   jointly driven (latent factors, wave dropout, whole-respondent nonresponse). These were *excluded*
   from the Stage-5 pools to keep per-column tagging clean; they return here for realism.
   **Convention:** a joint mechanism stamps *every cell it co-deletes* with its own class. This is a
   convention, not an identifiable fact about the individual cell — documented as such (§For
   statisticians).

4. **Composition-controlled sampler.** Draw a target mix from a **broad / near-flat prior** on the
   simplex (train prior-agnostic; deployment may supply a real-world prior), then allocate mechanisms
   to columns/cells to realise it *by missing cell.* Per-column miss rates therefore **vary** — which
   is realistic, and which returns miss rate as a *legitimate, openly-used* signal rather than a
   hidden confound. Miss rate is part of "the data as it exists"; we use it openly and report when a
   conclusion rests on it.

5. **Realism program — match the observable footprint (a floor, not proof of mechanism fidelity).**
   Generators are tuned so their *observable* missingness footprint (Appendix feature set) is
   statistically indistinguishable from real datasets'. **The discriminator is a ruler / diagnostic,
   NOT a training adversary** — its accuracy is the realism score and its attributions name the gaps;
   the generators stay mechanistic so they keep carrying the labels. Statistic-matching first; a true
   adversarial update is considered only if a strong discriminator keeps winning on a feature we
   genuinely cannot hand-fix. We do **not** build a GAN that generates masks (a neural sampler has no
   mechanism label — it would destroy the ground truth).
   *Data reality (2026-05-30 inventory):* real missingness exists only as **survey** data — 14 labeled
   anchors (`lacuna_survey/evaluation_data/*_real.csv`, registered in `lacuna_survey/anchors.py`) plus
   **unlabeled** masks cheaply extractable from the raw NHANES/PISA sources
   (`/mnt/data/lacuna/{nhanes,incoming}/`). The hole-punching catalog (`/mnt/data/lacuna/raw/`) is
   complete-data-only (`ingestion.load_csv` drops NaN). Three consequences: **(a)** the realism target
   is the **survey manifold** specifically — "realistic" means "realistic *for surveys*," coherent with
   the ADR-0005 survey-specialisation, and the claim is scoped accordingly (not general tabular);
   **(b)** with only ~14 real exemplars a *learned* adversary is infeasible regardless, which
   independently confirms the statistic-matching choice — and since footprint matching needs no labels,
   the real-mask pool can be grown cheaply from the raw surveys for Stage A; **(c)** real masks are
   loaded **mask-preserving** (via the `lacuna_survey` path), never the `dropna` catalog loader.
   `lacuna_survey/calibrate.py` is already a working real-vs-synthetic calibration loop — a direct
   scaffold for the Stage-D composition-posterior calibration.

6. **Composition head over the existing encoder.** Keep the encoder (it already learned the
   fingerprints; it is fine at the dataset level) and the validated dataset-level head; **add** a
   composition head emitting a distribution over the simplex (a Dirichlet) plus an explicit can't-tell
   mass. Uncertainty from the Dirichlet concentration and a small deep ensemble. Multi-task so v1.0
   capability is preserved.

7. **The honest seam.** "Random vs structured" (MCAR vs MAR+MNAR) is testable — report it *tight*.
   "MAR vs MNAR" within the structured part is manifold-bound and non-identifiable in general —
   report it as an explicit *wide band / can't-tell mass.* **A wide MAR/MNAR band is a correct report
   of a real limit, not a model defect**, and is framed as success, not failure.

## Supervision and evaluation

- **Train + supervise: semi-synthetic only.** Cell-tagged ground truth exists only because we punched
  the holes. Unchanged from v1.0 in spirit; only the target (composition + cell tags) changes.
- **Headline metric: CALIBRATION of the composition posterior.** When it says "p% sure of simplex-
  region Q," it should be right p% of the time — reliability curves over simplex queries (including
  the two example statements above). Point accuracy (composition-L1) is *secondary*; calibration is
  the deliverable.
- **Realism metric:** discriminator AUC → 0.5 on the observable mask-feature set; report which
  features still separate synthetic from real.
- **Define "tanked" before running.** The composition posterior must beat (a) **classify-and-count**
  off the Stage-5 per-column classifier (known-biased), and (b) a **prior-only** predictor; its
  calibration must beat an uncalibrated softmax. "Tanked" = no better-calibrated than prior-only.
- **Real-data channels (necessary, not sufficient):** NHANES anchors (known-sensitive items →
  MNAR-suspect), skip-logic columns (missing-by-design → MAR), the Molenberghs bound. These test face
  validity only — mechanism accuracy on real data is impossible by construction.

## Experiment stages (one variable at a time)

- **Stage A — Realism footprint & gap report.** Build the observable mask-feature extractor; measure
  real-missingness data *and* current synthetic; quantify the gap (discriminator-as-ruler). Real masks:
  the 14 labeled anchors + unlabeled extracts from the raw NHANES/PISA surveys (commitment 5), loaded
  mask-preserving. A finding either way (e.g., "our masks are too random vs real survey block/skip
  structure").
- **Stage B — Composition-controlled, realism-tuned generator.** Rebuild the sampler to hit a target
  *by-cell* composition with cell tags; re-admit joint mechanisms; tune nuisance knobs to the
  footprint. Gate: realised == target by cell (the generalised confound test) AND realism passes.
- **Stage C — Composition head + uncertainty.** Dirichlet + explicit can't-tell mass on the
  (frozen-first) encoder; deep ensemble for the posterior.
- **Stage D — Calibration (the headline).** Reliability over simplex queries; the honest seam; the
  can't-tell mass tracks true ambiguity.
- **Stage E — Face validity + honest framing.** Anchors; the manifold caveat stated in neon.

## Consequences

**Enables:** a calibrated composition instrument that answers "how much, and how sure" and
**honestly flags when it can't tell** — useful for the sensitivity-analysis decision in all three
regimes (confident-MNAR → must do it; confident-not → can skip it, the case that saves work;
can't-tell → default to caution, no worse than today).

**Forecloses / costs:** a substantial generator rebuild and a shift of the supervised target; v1.0
dataset-level numbers do not transfer and must be re-validated; the per-column "where" layer is
demoted to optional.

**Conditions to revisit / supersede:**
- Stage A shows we cannot source enough real-missingness data to define realism → realism degrades to
  face-validity-only (an honest, smaller claim).
- Stage D shows the posterior is no better calibrated than prior-only *even for random-vs-structured*
  → the observable signal is insufficient; narrow the estimand (e.g., MCAR-fraction only).
- A reviewer rejects by-cell *generative attribution* as the definition of composition → amend the
  denominator (this is a definitional choice, stated for exactly that scrutiny).

## For statisticians using this tool

The composition is defined by **generative attribution** — which rule deleted which cell — *known
because we built the data.* This is distinct from **inferential identifiability** — what can be
recovered from observed data — which for the MAR/MNAR split is bounded by Molenberghs and which we
explicitly do **not** claim to beat. Semi-synthetic is the only setting where the question is even
posable (real data has no cell tags). "Composition by missing cell" is a definitional choice, stated
here so a committee can accept or contest it on the record. For joint mechanisms, "this cell is MNAR"
is a *tagging convention* — the whole co-deleted block inherits the joint mechanism's class — not an
identifiable property of an individual cell.

## Appendix — observable mask-feature set (Stage A)

All computable from the missingness mask + observed values alone (no mechanism, no held-out truth):

- **Column rates:** per-column miss-rate distribution → mean, sd, skew, max, % fully-observed columns,
  % (near-)fully-missing.
- **Row patterns:** per-row miss-count distribution → mean, sd, % complete rows, % near-empty rows
  (unit-nonresponse signature).
- **Co-missingness:** pairwise missingness correlations → mean/max |corr|, # strongly-coupled column
  pairs (blocks / modules).
- **Pattern structure:** # distinct missingness patterns / n; mass in the top-k patterns (skip-logic →
  few dominant patterns; pure randomness → many).
- **Monotonicity:** best staircase-ordering score (dropout / attrition signature).
- **MAR axis (observable):** correlation of each column's missingness indicator with *other columns'
  observed values* → distribution over columns.
- **MNAR axis (observable):** shape (skew, excess kurtosis) of each column's *observed* values.

The last two are the same fingerprints the column classifier used — here repurposed as *realism*
checks (do synthetic and real agree on how much covariate-coupling and observed-distortion exist?).

## Cross-references

- Superseded primitive: `docs/decisions/0006-column-level-missingness.md`.
- The miss-rate confound that motivated the inversion: `docs/experiments/2026-05-29-stage5-true-mixtures.md`.
- Encoder + dataset-level head to reuse: `lacuna/models/encoder.py`, `lacuna/models/heads.py`.
- Generator families to re-admit (joint): `lacuna/generators/families/mnar/{latent,selection,social}.py`.
- Per-column targeting + pools (reusable for allocation): `lacuna/data/{mnar,mar}_column_pool.py`.
- Real-data anchors: `lacuna_survey/anchors.py`.
