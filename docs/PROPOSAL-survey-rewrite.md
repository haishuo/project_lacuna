# PROPOSAL — Lacuna-Survey Re-architecture

*The engineering governing document. Point me at this file at the top of every build prompt. It
answers: what should the new Lacuna have (§1), what does the old Lacuna actually contain (§2), what
is the gap (§3), and how invasive is the surgery (§4). Governed by — and subordinate to —
`docs/NORTH-STAR.md`. If this doc and the charter disagree, the charter wins.*

**Status:** DRAFT v0.1 (2026-06-02). Audit complete; verdict proposed; phased plan is for discussion.
**Method:** wishlist → audit → delta → verdict, exactly the "blood tests before surgery" sequence.
**Code touched so far:** none. This is a plan, not a change.

---

## §1. Target architecture (the wishlist)

What the new Lacuna-Survey *should* be, derived strictly from the pinned charter — not aspiration.
Eight components, grouped by layer. (KEEP/NEW tags previewed here; justified in §2–§4.)

**Data & regime (real survey X → semi-synthetic training data)**
1. **Survey-X loading & catalog** — real survey tables as the X-base. *(mostly KEEP)*
2. **Mechanism application** — apply a mechanism to real X on a z-scored predictor view, return a
   mask; observed values stay on real scale. The v1.0 `apply_missingness` + `_zscore_columns` is the
   right primitive. *(KEEP the core; re-target the label it emits)*

**The manifold (the prior's support, made operational — charter §3½)**
3. **Survey-idiom mechanism samplers**, reparameterized by **(marginal rate, δ)** rather than a
   discrete class — where δ indexes departure from MAR toward MNAR. Curated to survey idioms (item
   nonresponse, skip logic, social-desirability self-censoring, attrition, unit nonresponse).
   *(REPARAMETERIZE the existing generators)*
4. **Footprint coordinates + manifold/OOD model** — a fixed vector of observable mask statistics
   (the M2 footprint), a density/typicality model over the survey-footprint manifold, and a
   **distance-from-manifold** signal that is the OOD/abstention trigger. *(NEW model; footprint
   statistics largely already exist)*

**The network (footprint+semantics → calibrated δ-prior, with abstention)**
5. **Representation backbone** — the set-transformer encoder over (value, observed, mask-type,
   feature-id) tokens. *(KEEP architecture; retrain — do not trust frozen geometry, see §5)*
6. **δ-prior head + abstention head** — replace the 3-class MoE/decision tail with (a) a head emitting
   a *calibrated distribution over scalar δ* (B2) per column/dataset, and (b) an abstention head fed
   by distance-from-manifold. **Plus an external, auditable column-semantics input** (BERT-like
   embedding or a frozen semantics→prior table) feeding the *prior*, under strict leakage control.
   *(NEW heads on the kept backbone)*

**Objective & evaluation (calibration-first)**
7. **Calibration-first training + eval harness** — a proper scoring rule over the δ-distribution
   (CRPS/log-score) + an abstention term, replacing cross-entropy-to-class. Eval = out-of-mechanism-
   family calibration, leave-one-form-out, OOD-abstention rate, oracle distinguishability, and
   MCAR-departure detection. **Not** in-distribution accuracy/confusion. *(NEW loss + NEW harness;
   training loop infra KEEP)*

**Product (the governance output)**
8. **Estimand-propagation reporting layer** — thin, deterministic: (δ-prior × stated estimand) →
   tipping-point curve + P(conclusion flips), via δ-adjustment / pattern-mixture. Estimand enters
   *here*, not in the network (charter §3¾). *(NEW; small and well-understood statistics)*

**Cross-cutting:** determinism via injected RNG (KEEP), explicit contracts & fail-loud (KEEP),
run-tracking/provenance + the auditable semantics→prior table (ADAPT existing registry).

---

## §2. As-is audit (the salvage map)

23,543 LOC across 101 files in `lacuna/`, plus 17 scripts. Audited subsystem-by-subsystem (five
parallel readers, findings cross-checked). Verdicts: **KEEP** (objective-agnostic, reuse as-is or
near), **REPARAM** (reuse the machinery, change the parameterization/labels), **REPLACE** (tied to the
dead 3-class objective), **NEW** (does not exist).

### 2.1 `core/` (655 LOC) — foundation, mostly KEEP
Pure leaf layer; nothing imports downward into it. `rng.py` (seedable RNGState), `exceptions.py`,
`scheduling`-style validators = **KEEP verbatim**. `validation.py` KEEP except `validate_class_id`
(hard-codes {0,1,2}) → REPLACE. `types.py` is the **load-bearing hub** (every subsystem flows through
its dataclasses): `ObservedDataset`, `TokenBatch` = KEEP; `PosteriorResult`/`Decision` + `MCAR/MAR/
MNAR`/`CLASS_NAMES` constants = REPLACE (they *are* the 3-class contract). This file is the single
most depended-on artifact; its I/O contract change ripples everywhere — sequence it first.

### 2.2 `data/` (3,414 LOC) — plumbing KEEP, label-path REPLACE, footprint = hidden asset
- **KEEP:** `ingestion.py` (real-X loading), `observed.py` (container/split), `catalog.py` (survey
  registry), `normalization.py` (per-column, observed-values-only — correct for missingness).
- **KEEP the core, REPLACE the label:** `semisynthetic.py` — `apply_missingness` + `_zscore_columns`
  + `subsample_raw` are exactly the mechanism-application primitive and the load-bearing saturation
  fix; keep them. But `SemiSyntheticDataLoader` emits one `class_id` per item via `class_mapping` —
  that whole label path (also in `tokenization.tokenize_and_batch`, `batching.collate_fn`,
  `SemiSyntheticDataset.class_id`) becomes a **δ-label** path.
- **KEEP encoder-input, REPLACE label-output:** `tokenization.py` — the cell→token encoder is reusable
  infra; the `generator_ids → class_mapping → class_ids` materialization is the dead target.
- **HIDDEN ASSET (re-scope, don't discard):** `missingness_features.py` (734 LOC) computes *observable
  mask statistics* (per-column rate moments, cross-column missingness correlation, value-conditional
  shape shift). This is **the M2 footprint machinery for the manifold/OOD model** — the math is
  label-free and reusable; only the "MCAR/MAR/MNAR signature → MoE gate" framing is re-scoped.
- **REPLACE payload, KEEP machinery:** `littles_cache.py` (707 LOC) caches MCAR-test scalars (wrong
  payload for δ/manifold), but its keyed `(dataset, generator)` parallel-build cache is a reusable
  template for **precomputing per-(survey, mechanism) footprint vectors** to build the manifold.

### 2.3 `generators/` (7,824 LOC) — the crown jewel; REPARAM, don't rebuild
The most reusable subsystem and the heart of the new design too. Uniform contract across ~112 classes:
`_compute_missingness(X, rng) -> R` behind `apply_to(X, rng)`; near-zero coupling (imports only
`core.rng` + the `MCAR/MAR/MNAR` ints). `params.py` (untyped immutable bag) KEEP — already accepts any
`(rate, δ)` keys. `registry.py`/`registry_builder.py` container + auto-discovery KEEP; their
class-partitioned methods REPLACE. `priors.py` (categorical over generator-ids) REPLACE — "prior" now
means prior over δ. `base_data.py` Gaussian-X samplers REPLACE — sever the synthetic-X coupling, feed
real survey X.
- **The key reparam finding:** `MNARLogistic` already *is* `logits = β0 + β1·X_pred + β2·X_target` —
  **β2 is δ, β0 sets the rate, MAR is the same model at β2=0.** So a single `(rate, δ)` logistic family
  subsumes MARLogistic + the own-value MNARs; the discrete class boundary is an artifact of fixing
  β2∈{0,≠0}. Adding a settable marginal rate = a one-line intercept solve (bisection on realized
  miss-fraction). The `apply_to` contract does **not** change; this is a param-layer + intercept-
  calibration wrapper. (Also: promote `apply_to` to an `@abstractmethod` — today it's convention.)
- **The risk (see §5.1):** own-value self-censoring collapses onto scalar δ; **`selection.py`
  (attrition, Heckman, Berkson) and `latent.py` (factor-driven) do NOT** — they need a separate
  selection variable / shared latent. A single global δ silently drops attrition + unit-nonresponse
  (in scope) or makes δ non-identifiable. **δ must be defined per-idiom, or the manifold needs a
  multi-axis coordinate.**

### 2.4 `models/` (3,515 LOC) — ~60–65% reusable backbone, ~35–40% objective-tied tail
- **KEEP:** `encoder.py` (752 LOC) — set-transformer; references no class count; already exposes
  `return_intermediates`, `token_representations`, `row_representations`. Reusable as the δ-prior
  backbone *architecture*. (Retrain — §5.2.)
- **MIXED:** `moe.py` — the `GatingNetwork` MLP + learnable temperature are reusable plumbing; the
  `expert_to_class` buffer, `get_class_posterior`, `get_mnar_variant_posterior`, 3-class aggregation
  REPLACE. `assembly.py` — wiring + encoder reuse KEEP; the `evidence → moe → class-posterior →
  BayesOptimalDecision` tail REPLACE. The clean seam: **cut immediately after
  `evidence = encoder(...)`.**
- **REPLACE:** `reconstruction/heads.py` + `heads_container.py` (MCAR/MAR/MNAR world-models that exist
  only to feed the mechanism-discrimination error signal); inline `BayesOptimalDecision` (3×3 loss
  matrix, Green/Yellow/Red).
- **DEAD/orphan (~350 LOC):** `decision.py`, `heads.py`, `aggregator.py` are not imported by the live
  model. Salvage `GeneratorHead` (a generic evidence→K-logit MLP, near-drop-in δ-prior head shape) and
  the entropy/confidence helpers; discard the rest.
- **NET-NEW needed:** no abstention/OOD/distance-from-manifold head exists anywhere; no proper scoring
  rule. Temperature scaling exists (a real calibration seam to build on).

### 2.5 `training/` (3,671 LOC) — infra KEEP, objective + eval REPLACE, loop REFACTOR
- **KEEP verbatim:** `scheduling.py` (LR), `logging.py` (JSONL, injectable clock), `checkpoint.py`
  (I/O/manager/export — drop only the vestigial `best_val_acc`; note 649 LOC > limit).
- **REFACTOR (metric seam not isolated):** `trainer.py` + `training_step.py` — the epoch/step skeleton
  is reusable, but `val_acc`/argmax-accuracy/per-class buckets/3×3 confusion and `class_ids` are reached
  into directly across state, logging, early-stopping, and step code. The loop must expose a
  **pluggable metric+eval object**, not just a swappable loss. `early_stopping.py` KEEP mechanism,
  re-key off a calibration metric (no `val_acc`).
- **REPLACE:** `loss.py` (956 LOC > limit) — `LacunaLoss` is cross-entropy/Brier-over-one-hots +
  reconstruction MSE + load-balance; no δ proper-scoring rule. `brier_score` is the only proper-scoring
  primitive and it's discrete. Needs a new δ-calibration loss (CRPS/log-score + abstention).
  `report.py` (confusion/precision/recall/F1/per-class) → the new harness. `calibration.py` —
  temperature-scaling *scaffolding* reusable; its class-aggregation target REPLACE. (ECE actually
  lives in `lacuna.metrics`, a sibling not audited here — flag for the next pass.)

### 2.6 `experiments/` (542 LOC) — run-tracking, mostly KEEP, re-scope metric fields
`registry.py` (JSON CRUD, atomic writes) + `registry_render.py` + `migrate.py` = KEEP machinery;
`mnar_variants`/`n_experts` fields and the Accuracy/MAR-Acc/ECE render columns → δ-calibration fields.
Notably `VALID_STATUSES` already includes `"calibrated"` — on-theme. Two empty stubs.

### 2.7 `scripts/` (17 files) — not yet audited at file level
`train.py`/`evaluate.py` are the real entry points (used for the v1.0 repro). Many scripts are
arc-era (`metadata_prior/`, `run_pipeline.py`, ablations). A scripts triage is a follow-up task.

---

## §3. Delta (wishlist − as-is)

| Target component (§1) | Exists today? | Delta |
|---|---|---|
| 1 Survey-X loading/catalog | Yes (`ingestion`, `catalog`, `normalization`) | ~0 — KEEP |
| 2 Mechanism application | Yes (`apply_missingness`+`_zscore`) | Small — keep core, swap emitted label class_id→δ |
| 3 (rate, δ) idiom samplers | Generators exist; **class-framed, no rate/δ knobs, Gaussian-X** | **Medium** — reparameterize: add (rate, δ), intercept-calibration, curate idioms, sever synthetic-X, per-idiom δ (§5.1) |
| 4 Footprint + manifold/OOD | Footprint stats exist (`missingness_features`); **no manifold/OOD model** | **Large** — build manifold density + distance-from-manifold; reuse footprint math + cache machinery |
| 5 Encoder backbone | Yes (`encoder.py`) | Small (arch) — KEEP arch, **retrain**, maybe add column-axis pooling |
| 6 δ-prior + abstention + semantics | **None** (3-class MoE/decision instead) | **Large** — new heads on kept backbone; net-new abstention + auditable semantics input + leakage controls |
| 7 Calibration loss + eval harness | **None** (CE-to-class + confusion eval) | **Large** — new proper-scoring loss; new eval (out-of-family calib, LOFO, OOD, oracle); refactor loop's metric seam |
| 8 Estimand-propagation reporting | **None** | **Medium** — new, but small & standard statistics (δ-adjustment) |
| Cross: determinism/contracts/tracking | Yes | Small — KEEP, re-scope registry metric fields |
| Core contracts (`types.py`) | 3-class baked into `PosteriorResult`/`Decision`/consts | **Medium** — redefine the I/O contract (ripples widely; do first) |

**Shape of the delta:** the *data/infra spine and the mechanism layer survive*; the *objective head,
the loss, and the evaluation are replaced*; and *three genuinely new things* must be built that have no
antecedent (the manifold/OOD model, the δ-prior+abstention+semantics head family, the calibration-first
objective+harness) plus a small new reporting layer.

---

## §4. Verdict — how invasive is the surgery?

**Diagnosis: a targeted re-architecture (strangler-fig), NOT a greenfield rewrite, and NOT a light
massage.** Keep the patient; replace one organ (the objective head + eval), reparameterize another
(the generators), and graft on three new ones (manifold/OOD, δ-prior/abstention/semantics, calibration
objective+harness+reporting).

Rough LOC disposition (order-of-magnitude, to size the work — not a commitment):

- **KEEP as-is / light touch (~45–55%):** `core` infra, `data` loading/normalization/tokenizer-input,
  the `apply_missingness` primitive, the encoder *architecture*, `training` infra (scheduling/logging/
  checkpoint), `experiments` registry machinery.
- **REPARAMETERIZE (~20%):** the generator layer (class→(rate,δ), intercept-calibration, idiom
  curation, real-X) — high LOC but high-value reuse; the contract stays.
- **REPLACE (~20%):** reconstruction heads, MoE class-tail + decision, `loss.py`, `report.py`, the
  `class_id` plumbing, `types.py` posterior/decision contracts.
- **NET-NEW (additive):** manifold/OOD model, δ-prior + abstention + semantics heads, calibration
  loss, new eval harness, estimand-propagation reporting layer.

**Why strangler-fig and not greenfield:** the two subsystems that took the most engineering — the
generator framework (7.8k LOC, uniform contract, near-zero coupling) and the encoder (752 LOC,
objective-agnostic) — are *exactly* the parts the new design reuses. Throwing them away to rewrite from
scratch would discard the project's most defensible assets to re-solve solved problems. The dead weight
is concentrated and excisable (the 3-class head, decision, loss, confusion eval), and the clean seam
("cut after `evidence`") means the excision is localized.

**Why not a light massage:** the change is not "swap the loss." The 3-class assumption is *not* isolated
behind the loss interface — it's woven through `types.py` contracts, the data label path, the model
tail, the training-loop metrics, and the eval. And three load-bearing components (manifold/OOD,
δ-prior/abstention, calibration harness) don't exist at all. This is structural.

**The one finding that could widen the verdict toward "more rewrite":** if the per-idiom δ problem
(§5.1) cannot be resolved within the existing generator/data contract — i.e., if representing attrition/
unit-nonresponse forces a fundamentally different data model than (X, mask) — then the generator+data
delta grows from REPARAM toward REPLACE. **Resolve §5.1 on paper before committing to the phased plan.**

---

## §5. Open scientific risks (must be settled on paper before/early in the build)

1. **Per-idiom δ vs. global scalar δ.** Own-value self-censoring = scalar δ (=β2). Attrition, unit-
   nonresponse, Heckman/Berkson selection do NOT reduce to one logit's δ. Either define δ per-idiom
   (a small set of named departure axes) or accept B2's validity envelope excludes those idioms and
   route them to abstention/B1. **This is risk #1 and it directly bounds B2's "approximate for others."
   Decide the manifold's coordinate system here.**

   **DECISION (2026-06-02, least-invasive-first):** the **first increment is scoped to the own-value
   self-censoring axis ONLY**, where scalar δ is *exact* (δ ≡ β₂ on the z-scored target; MAR = δ=0;
   rate matched via a β₀ intercept solve; the MAR-predictor strength β₁ is a controlled nuisance).
   Attrition, unit-nonresponse, and selection idioms are **explicitly deferred** and routed to
   abstention for now. **Revision trigger (a documented next-step, not a maybe):** *if the
   self-censoring-axis δ-prior demonstrably works (passes the §6 feasibility gate and out-of-family
   calibration), we revise the manifold to a per-idiom / multi-axis coordinate system that covers
   attrition + unit-nonresponse — i.e., we then move from allopurinol toward full coverage.* If it
   does **not** work on the axis where δ is exact, we stop: no point generalizing a broken core.
2. **Encoder geometry for OOD.** The encoder was trained discriminatively for class separation; its
   latent may place true OOD inputs inside the decision manifold, making distance-from-manifold
   unreliable. Reuse weights as init only; the abstention model likely needs the footprint-statistic
   space (label-free, geometry-faithful) and/or a retrained/representation-regularized encoder. Do not
   repeat the frozen-encoder-probe mistake.
3. **Rate/δ confound, structurally defused.** Build the settable-marginal-rate intercept solve into the
   generator layer so every training pair can be matched-rate by construction (charter §4.7). Without
   it, δ-signal re-confounds with rate as it did in v1.
4. **Leakage via semantics.** If a column's semantics inform *both* the hole-punching and the model
   input, the δ-label leaks. Evaluate on independently-assigned mechanisms + held-out families; keep
   the semantics→prior table frozen and auditable (charter §4.5).

---

## §6. Phased plan (DRAFT — for discussion, not yet committed)

Sequenced **decisive-first** — and *decisive* is the operative word, not *cheap*. Every gate uses an
estimator at least as strong as what we deploy (charter §4.9); a cheap test that can only yield a
*confounded* kill is not decisive and is not run. We do not reparameterize generators or build the
δ-network until the gate shows observed-data signal exists. (Reusing the framework instead of
rewriting it is the economy that matters; skipping training is not an economy — charter §4.9, §5.2.)

- **P0 — paper resolutions (no code):** DONE for the first increment — the axis is own-value
  self-censoring; δ ≡ β₂ on z-scored target, MAR=δ=0, rate matched via β₀ solve, β₁ a controlled
  nuisance (see §5.1 DECISION). Remaining P0: pin the **X-model** for the oracle (exact on synthetic X;
  a fitted conditional model on real X) and the matched-rate / form / sample-size grid for P1.
- **P1 — decisive feasibility, done RIGHT: bracket the truth with two instruments, neither weaker
  than what we deploy (charter §4.9).**
  (a) **Information-ceiling oracle** — the Bayes-optimal discriminator on OBSERVED data only, computed
  from the KNOWN self-censoring generative model: the analytic observed-data likelihood ratio between
  δ=0 and δ=δ\* at matched rate (target *observed* → `p(z_t|z_p)·(1−σ(β₀+β₁z_p+β₂z_t))`; target
  *missing* → `∫ p(z_t|z_p)·σ(β₀+β₁z_p+β₂z_t) dz_t`). Exact on synthetic X (p(X) known — a legitimate
  math property of the mechanism family, NOT an accuracy-on-synthetic-X claim); checked on real X via
  a fitted X-model (the only assumption, reported). Yields the `(δ, rate) → Bayes-error` surface. **A
  NEGATIVE here is an unconfounded, theorem-like kill.**
  (b) **The real model, RETRAINED FROM SCRATCH** on the matched-rate self-censoring task (δ=0 vs δ>0 /
  regress δ), held-out / out-of-family eval. **Frozen-encoder + stapled-head is BANNED** (§4.9, §5.2) —
  full retrain or it does not count.
  **Reading the bracket:** oracle fails → *fundamental* (no observed-data signal at matched rate;
  stop, report honestly). Oracle succeeds, model fails → *our system leaves signal on the table*
  (fixable; iterate the model, not the conclusion). This bracket is the only thing that separates
  Molenberghs from implementation — a single weak proxy cannot.
  (c) **Leave-one-form-out** (form-invariance, charter §6) — once (a)/(b) establish signal exists:
  train on some link forms (logistic/probit/threshold), test on a held-out form (spline) at matched
  rate, with the real retrained model.
- **P2 — generator reparameterization:** (rate, δ) + intercept-calibration + idiom curation + real-X;
  promote `apply_to` to abstract. Keep the contract.
- **P3 — manifold/OOD model:** footprint cache (reuse `littles_cache` machinery, swap payload) →
  density/typicality → distance-from-manifold + abstention.
- **P4 — δ-prior head + objective:** new head on the kept (retrained) encoder; CRPS/log-score +
  abstention loss; refactor the training loop's metric seam to a pluggable eval object.
- **P5 — eval harness:** out-of-family calibration, LOFO, OOD-abstention, MCAR-departure. The success
  metrics, not accuracy.
- **P6 — reporting layer:** δ-prior × estimand → tipping-point (B2 / δ-adjustment).
- **P7 — semantics prior:** auditable semantics→prior input under leakage controls.

Each phase: update `CHANGELOG.md`, keep the suite green, split any file before it nears 500 LOC.

---

## §7. Non-goals (to prevent scope creep)

- Not building B1 (rich mechanism posterior) in the first pass — documented escalation only.
- Not generalizing beyond survey data — sibling regimes are separate instruments (charter Scope).
- Not claiming to identify the mechanism of a single dataset from its values — forbidden (charter §2).
- Not chasing in-distribution accuracy — it is not a success metric here.
- Not merging anything from the abandoned `experiment/subtype-layer` arc — concepts may be
  *re-derived* (footprint/OOD), code is not grafted.

---

## §8. Pre-existing debt to clear during the rework (not new work — hygiene)

- Split `training/loss.py` (956) and `training/checkpoint.py` (649) under the 500-LOC limit.
- Route `data/tokenization.py::tokenize_dataset` row subsample through an injected `RNGState`.
- Add tests for `core/` and `experiments/` (currently none).
- Audit `lacuna/metrics` (sibling holding ECE/selective-accuracy) — not covered in this pass.
- Triage `scripts/` (17 files; many arc-era).
