# Revised Architecture Object — Lacuna-Survey (prior-aware, deviation-optional)

*Architecture research, not implementation. **No code, no experiments, no Gate-II build.** Status: **for
PI review.** Supersedes the object framing in `ARCHITECTURE-INVESTIGATION-lacuna-survey.md` §A by
incorporating `INFERENCE-OBJECT-CRITIQUE.md`: the reference is **prior-bearing, not identified**, so the
object is stated as *observed-data law + named prior → calibrated δ-prior / detectability / abstention*,
and the deviation is demoted to an optional, empirically-gated internal module. Governed by
`NORTH-STAR.md` (§2 identification, §3 reframed estimand, §3½ manifold, §6 coverage, §8 ladder).*

---

## 1. The corrected object (what Lacuna-Survey estimates)

**Identified (data alone):** the **observed-data law** of a survey table — per-column observed
distributions, the cross-column joint, and the mask topology. Nothing else is identified (Molenberghs §2).

**Product (data + an explicit prior):**

> **observed-data law  +  named survey-manifold prior  →  calibrated δ-prior  +  detectability state  +
> abstention/coverage.**

The δ-prior is a *posterior under a named prior*, never a claim of identifying δ from data alone. The
reference distribution is **not** an observable; it is the **δ=0 slice of the named prior** (see §3). The
deviation is **one optional internal parameterization** of how the model uses the prior, not the object.

This preserves the North Star verbatim: we never claim to beat non-identifiability; we make the *prior*
over the sensitivity parameter explicit, calibrated, auditable, and cheap, and we **say so out loud** when
the footprint is flat or off-manifold.

---

## 2. Three product levels

| level | what it is | adds over previous | status |
|---|---|---|---|
| **Level 0 — identifiable-footprint model** | column-primary marginal distribution encoder φ (order-statistic/ECDF-preserving) → δ-bin head. Captures the *detectable marginal* footprint (LOD/top-coding). | — | **proven** at baseline (Stage 0: φ ≈ raw-ECDF 0.735, ≫ backbone 0.548) |
| **Level 1 — prior-aware governance model (the MVP / honest North-Star object)** | Level 0 **+** mask-topology stream **+** the **named prior** semantics **+** calibration **+** detectability state **+** abstention/coverage. Output is a *calibrated δ-prior*, not a classifier score. | the **governance wrapper**: named prior, calibration, "confidently uncertain" where flat, abstain where off-manifold | **design-ready** (light; reuses scaffold) |
| **Level 2 — optional reference/deviation module** | an **explicit, auditable** module that constructs a *prior-implied conditional reference* for the target and compares observed-vs-reference, feeding the deviation to the head. | a *conditional* (predictor-aware) use of the prior, **if** it earns its keep | **gated, unproven** (the fragile layer; §5) |

**Crucial honesty:** Level 1's δ-*discrimination* on the footprint ≈ Level 0's — the value Level 1 adds is
**governance**, not accuracy (calibration, detectability, abstention, the named-prior framing). That is
correct: the North-Star product is governance, not a higher confusion-matrix number (§5). Level 2 is the
**only** place additional *discriminative* signal could come from, and it is the historically-fragile,
proxy-exposed layer (`INFERENCE-OBJECT-CRITIQUE` Q3/Q5).

The architectural "rewrite" is therefore **light**: build Level 1 on the retained scaffold with a
column-primary φ spine. Level 2 is a *conditional upgrade*, not a default, and not yet decided.

---

## 3. The named prior (Q2, Q3, Q4)

### Q2 — What exactly is the named prior?
**Not** a hand-authored Bayesian density over δ, and **not** an LLM opinion (§8.3 forbids that). It is the
**explicit, auditable empirical measure that generates the semi-synthetic supervision**:

> **P_prior  =  {catalog of real survey datasets}  ×  {idiom vocabulary V}  ×  {δ-grid with sampling
> weights}  ×  {matched-rate regime}  ×  {φ / architecture inductive bias}.**

The model learns `P(δ | footprint)` under `P_prior`; on a new column it returns the **posterior under
`P_prior`**. It is "survey-manifold" because the datasets are survey and the idioms are the survey
vocabulary (§3½ M1). It is "named" because **every factor is enumerable, inspectable, and changeable** —
which dataset catalog, which idioms, which δ-grid, which rate regime. This is the operational meaning of
"the non-arbitrariness lives in the prior" (§2): the prior is a *measure over real-X semi-synthetic
worlds*, not an assertion about any real dataset's truth.

**The reference is the δ=0 slice of `P_prior`.** "What an un-censored survey column of this type looks
like" is *defined* by the MAR (δ=0) members of the training measure. This makes the critique's point
concrete: the reference is a **prior object**, evaluated like any other part of the prior — not an
identified MAR expectation. (Level 2 makes this reference *explicit and conditional-on-X*; Level 0/1 keep
it *implicit in φ's weights*.)

### Q3 — What information does the prior use?
- **Data-derived (the footprint):** per-column observed distributions (φ), mask topology (rates,
  co-missingness). These update the prior toward the likelihood where the footprint is informative.
- **Prior-supplied (not in the data):** the idiom vocabulary V and the δ-grid (which departures exist,
  how strong, a priori); the **reference shape** (the δ=0 slice); the **matched-rate regime** (removes the
  rate cue so signal is forced into shape — the leakage guard).
- **Deferred / gated:** column semantics / codebook metadata (§8.3) — a legitimate prior source **only**
  if itself calibrated on the held-out ladder; **excluded** from Levels 0–1, a future gated channel.
- **Never used:** any claim about the specific dataset's true δ (unobservable, §4.4).

### Q4 — How is the prior evaluated on semi-synthetic held-out data?
By the validation ladder (§8.1), measuring the *posterior under `P_prior`* on **held-out draws from
`P_prior`** (real datasets not in training, semi-synth holes, known δ):
1. **Calibration of the δ-prior** — do predicted δ-bin probabilities cover the true δ at nominal rates?
   (ECE, coverage tables — already in `metrics`.) The headline.
2. **Out-of-family / leave-datasets-out** — does calibration hold on **unseen datasets** and (the §5
   headline) **unseen idiom forms**? This is the transportability bound.
3. **Detectability calibration** — where the **oracle** says flat (`lod_oracle` BE→0.5), does the model
   return wide/high-entropy; where the oracle says sharp (BE→0), does it sharpen? The model's
   detectability state is validated against the *computable* oracle, not asserted.
4. **Abstention / OOD** — on footprints from a **held-out idiom family**, does it flag off-manifold and
   degrade to the prior marginal rather than confidently mislabel? (§6.)
5. **Reference validity (Level 2 only)** — is the δ=0-slice reference itself calibrated (does it predict
   held-out δ=0 columns' shapes)? A reference that fails this is opinion, not a prior (§8.3).

Never evaluated by "did it get δ right on real natural data" (unobservable; face-validity only, §4.4).

---

## 4. Output behavior (Q5, Q6) — and the distinction the critique enables

The model emits, per target column: **(a)** a distribution over δ-bins (the δ-prior); **(b)** a
**detectability** scalar; **(c)** an **OOD/coverage** score → abstain flag. (Estimand × δ-prior →
tipping-point is a *separate downstream layer*, §3¾, not in the network.)

Two *different* kinds of "uninformative" must be distinguished — a refinement the critique makes possible:

| situation | δ-prior output | detectability | OOD/abstain |
|---|---|---|---|
| **Strong footprint, in-manifold** (e.g. LOD, sharp truncation) (Q6) | **sharpened** toward the true δ-bin, low entropy | **high** | no |
| **Flat-likelihood, in-manifold** (e.g. own-value, matched rate) (Q5) | **≈ the prior marginal** `P(δ)` under `P_prior` (data-grounded base rate), high entropy | **low** | **no** — we *know* it is flat ("confidently uncertain") |
| **Off-manifold / novel idiom** (Q5) | degrade to the **uninformative prior**, high entropy | low | **yes — abstain / escalate** ("we don't know") |

The middle and bottom rows are **not** the same output and must not be conflated: *flat-likelihood
in-manifold* is a **known** non-identification (return the prior, do not abstain); *off-manifold* is an
**unknown** (abstain). This is the operational core of §3 ("honestly uncertain exactly where Molenberghs
says it must be is *working*") and §4.3/§6 (abstain at the coverage boundary). Where the footprint is
flat, the output is the **prior** — which is the entire product: a data-grounded, calibrated δ-prior used
as the default in sensitivity analysis, *with* the model saying "this is prior-dominated."

---

## 5. When may the reference/deviation module enter, and what earns it? (Q7, Q8)

### Q7 — When is Level 2 allowed to enter?
**Never by default.** Level 2 enters **only** after a pre-registered, deployment-strength analysis (not a
weak proxy, §4.9) shows it earns its complexity. Until then, Level 1 uses the *implicit* marginal
reference in φ. The reference module is an **upgrade conditioned on evidence**, and a negative reverts to
Level 1 — no sunk-cost escalation.

### Q8 — What counts as evidence it earns its complexity?
All of the following, pre-registered, on the held-out ladder:
1. **OOF signal gain:** Level-2 (explicit conditional reference/deviation) beats Level-0/1 **marginal φ**
   on held-out δ-prior calibration/discrimination for a detectable idiom, by a margin exceeding seed
   variance. (φ ≈ raw-ECDF is the bar to beat — Stage 0.)
2. **Proxy-absorption robustness:** the gain **holds in the high-proxy-R² stratum** (where own-value
   MNAR ≈ MAR-on-proxy). A reference that collapses under absorption (the P2.2b failure mode) fails the
   gate. This is the decisive test, because absorption is exactly where a conditional reference both
   *could* help (subtract the proxy) and *could* hurt (subtract the δ).
3. **Own-value preservation:** Level 2 does **not** spuriously sharpen own-value-at-matched-rate — it
   keeps the wide prior where the idiom is genuinely flat (no false confidence; §4.2).
4. **Calibration / auditability:** Level 2 improves ECE/coverage **or** yields an inspectable reference a
   human auditor can check — a governance gain even at equal discrimination (legitimate for a governance
   tool).
5. **Reference itself calibrated** (§3 Q4.5): the δ=0-slice reference predicts held-out δ=0 shapes; an
   un-calibrated reference is opinion, disqualified.

Failing (1)–(2) ⇒ **do not build Level 2**; ship Level 1 + the sensitivity-reporting layer. (Classical
sensitivity theory — `INFERENCE-OBJECT-CRITIQUE` Q5.3 — warns the computable deviation recovers only the
*identifiable shadow* of δ, so the prior for (1)/(2) succeeding is *guarded*; treat success as the thing
to be shown.)

---

## 6. Minimal viable Lacuna-Survey architecture (Q1)

If the deviation is optional, the **MVP is Level 1**, and it is light:

- **φ — column-primary distribution encoder** (per-column observed-distribution embedding; order-statistic
  / ECDF pooling; row-permutation-invariant; scale-invariant). *Load-bearing* (Stage 0). Reuses the
  tested `masked_quantile_pool` primitive on raw values.
- **Mask-topology stream** — per-column miss rates + cross-column co-missingness (the MCAR/MAR-identifiable
  anchor and an abstention input). Revives the *concept* of the v1.0 `MissingnessFeatureExtractor`. Can
  start minimal.
- **δ-prior head** — δ-bins + RPS + post-hoc temperature (reuse `loss`, `delta_bins`, `metrics`).
- **Detectability + abstention head** — outputs the detectability scalar (calibrated vs the oracle) and an
  OOD/coverage score (distance-from-`P_prior`-footprint-manifold, §3½ M2/M3). *Load-bearing for governance;
  the new piece.*
- **The named prior** = `P_prior` (§3), realized as the learned, held-out-calibrated `P(δ | footprint)`.

**Excluded from the MVP:** any cross-column *conditional reference* (that is Level 2, gated); any metadata
channel (later, gated); the BERT backbone as the δ spine (Gate I: replaced — it may survive only as an
optional auxiliary mask/cross-column stream, never the spine). The full scaffold (generators, oracle,
leakage gate, manifest, calibration, ladder) is **kept unchanged**.

So the MVP is essentially **Stage-0 φ generalized to the δ-bin grid + a mask-topology stream + a
detectability/abstention head, trained semi-synthetic, calibrated and evaluated on the held-out ladder** —
a light representation rewrite on the retained scaffold, **not** a heavy new architecture.

---

## 7. Answers to the eight questions (index)

1. **Minimal viable architecture** → §6: Level 1 = φ + mask-topology + δ-prior head + detectability/
   abstention; no reference module, no metadata; light, scaffold retained.
2. **What is the named prior** → §3 Q2: the explicit semi-synthetic training measure `P_prior` (datasets ×
   idioms × δ-grid × matched-rate × φ bias); its δ=0 slice *is* the reference.
3. **What information the prior uses** → §3 Q3: footprint (data) + idiom vocabulary/δ-grid/reference-shape/
   rate-regime (prior); metadata deferred/gated; never the dataset's true δ.
4. **How the prior is evaluated** → §3 Q4: held-out-ladder calibration, leave-datasets-out / out-of-family,
   detectability-vs-oracle, abstention-on-novel-idiom, reference-calibration (Level 2).
5. **Output when the law is uninformative** → §4: prior marginal + low detectability; **abstain only if
   off-manifold**, not if known-flat-in-manifold (the two are distinct outputs).
6. **Output when the law has a strong footprint** → §4: sharpened δ-prior, high detectability, no abstain.
7. **When the reference/deviation module may enter** → §5 Q7: never by default; only after a pre-registered
   deployment-strength gate; negative reverts to Level 1.
8. **What earns its complexity** → §5 Q8: OOF gain over marginal φ **and** proxy-absorption robustness
   **and** own-value preservation **and/or** calibration/auditability, with the reference itself calibrated.

---

## 8. Recommendation (no code)

1. **Adopt the corrected object** (§1) and the **named-prior** definition (§3) as the project's object of
   record, replacing "observed-vs-MAR-reference deviation."
2. **Treat Level 1 as the MVP / honest North-Star product** (§6) and Level 2 as a gated upgrade (§5).
3. **Next architecture-phase step (still analysis/spec, your call):** either (a) a **Level-1 design spec**
   (φ + mask-topology + δ-prior + detectability/abstention on the retained scaffold), or (b) the single
   pre-registered **Gate-II / Level-2 analysis** (does an explicit conditional reference beat marginal φ
   OOF without proxy collapse) to decide whether Level 2 is ever worth building. **No implementation until
   one of these specs is approved.**
