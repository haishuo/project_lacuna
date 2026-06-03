# Lacuna — North Star

*Re-read this whenever a metric, an architecture, or a clever generator starts to feel like the
goal. It is not the goal. This document is the goal. Its job is to stop local optimization from
overwhelming global optimization — the specific way this project has failed before.*

*Status: charter. Supersedes the per-experiment framing of the abandoned arc. Written 2026-06-02
after the reset to v1.0-canonical, distilling the identification analysis that motivated the
re-scope. Change it deliberately and in the open; do not drift away from it silently.*

---

## 0. One sentence

**Lacuna is a governance layer for missing-data assumptions.** It makes the *prior over the
missingness mechanism* — and the sensitivity analysis that prior feeds — **repeatable, inspectable,
calibrated, and logged**, at a scale and consistency no human can match. It does **not**, and
**cannot**, identify the mechanism of a single dataset from that dataset's values alone.

If a sentence we write about Lacuna contradicts that one, the sentence is wrong, not this.

---

## Scope — Lacuna-Survey

This instance governs **survey missingness only** (self- or interviewer-administered
questionnaires). The validity claim is explicit and bounded: *Lacuna-Survey's outputs are valid for
survey-collected tabular data and are not claimed to transfer to instrument, administrative-record,
or longitudinal data.* This is not a limitation to apologize for — it is the human-parity principle
(§4.8) applied: a human analyst is **told** what kind of data she has, so we tell the network too,
by restricting its regime. The same architecture re-trained on another regime's manifold yields
sibling instruments — Lacuna-Instrument, -Records, -Longitudinal — each honest within its own
manifold, none claiming to govern another's. Narrowing the regime also shrinks the manifold (§3½),
which is a feature, not a cost.

---

## 1. The problem we are actually solving

There are three ways people justify an imputation:

1. **"Assume MAR."** An abdication. Sets the unidentified parameter to zero by fiat and never looks.
2. **"A domain expert said MAR."** A real prior — but unauditable, unrepeatable, uncalibrated,
   unfalsifiable. Right or wrong, it leaves no track record.
3. **"We did a sensitivity analysis."** The only principled option — *and the least often performed*,
   because it is slow, it requires choosing a mechanism/parameterization, and it hides an unaudited
   copy of (2) inside it: **the "plausible range" of the sensitivity parameter is almost always a
   guess.** "What if you guessed the range wrong?" is the unanswered question at the heart of (3).

**Lacuna's job is to supply the ingredient (3) is missing:** a data-grounded, calibrated, auditable
prior over the sensitivity parameter, plus the cheap automation of the sweep — so that the only
honest method becomes the cheap, standardized, logged default instead of the rare exception.

---

## 2. The identification line (what the data can and cannot tell us)

This line is a **theorem**, not a tuning choice. Do not cross it.

- **MCAR vs. structured (MAR∪MNAR): IDENTIFIABLE.** Departures from MCAR leave testable restrictions
  on the observed-data law. Measurement claims here are fair game. This is Lacuna's solid ground.
- **MAR vs. MNAR, per-dataset, assumption-free: NOT IDENTIFIABLE.** Molenberghs et al. (2008): every
  MNAR model has a MAR counterpart with identical observed-data fit. No data, no compute, no
  architecture escapes this. **Any claim of the form "Lacuna measured that this table is MNAR from
  its numbers alone" is forbidden, permanently.**
- **The non-arbitrariness of real missingness is real information — but it lives in the PRIOR, not
  the likelihood.** Molenberghs flattened the likelihood; he did not touch the prior, auxiliary
  variables, or the population. That is the only door, and Lacuna goes through it honestly.

Detectability is **per-mechanism**, not universal: sharp truncation / limit-of-detection MNAR is
blatantly detectable; smooth self-censoring on correlated columns at matched rate is nearly
invisible. "MNAR is undetectable" is false; "*this* MNAR mechanism is near-indistinguishable from
its MAR twin at matched rate" may be true and is a *computable* property (see §5, oracle test).

---

## 3. The reframed estimand

We move the central question **off the non-identified axis and onto a generalization axis** — because
identification failure is fatal and unfixable, while a generalization gap is *estimable and
improvable*.

- **OLD (dead):** "Name the mechanism class of this dataset." Non-identified, brittle, forces a
  confident call where honesty demands a shrug.
- **NEW:** "Given the observable footprint, the column semantics, and the domain, return a
  **calibrated prior over the sensitivity parameter δ**, and — for a stated estimand — the
  **tipping-point robustness** (where does the conclusion flip, and how much prior mass lies beyond
  it?)."

In synthetic-world we *know* δ because we set the mechanism, so the map (footprint, semantics) → δ is
a **supervised learning problem with real labels**, identified as a learning problem. Its only
weakness is transportability to real data — a bounded, measurable problem, not a theorem-level wall.

A consequence to internalize: **the matched-rate "collapse" is a success in this framing.** Where
MAR and MNAR are genuinely indistinguishable, the correct output is a wide, high-entropy prior near
δ=0. A model that is honestly uncertain exactly where Molenberghs says it must be is *working*.

---

## 3½. The manifold hypothesis (the geometric form of the prior)

The bet the whole project rests on: **the set of missingness mechanisms nature actually produces is
a small, structured, recurring subset of the set of mathematically possible mechanisms** — and
restricting to *survey* missingness shrinks it further. This is the spirit of AlphaFold (vast
conceivable space; a few thousand fold topologies nature reuses), with one load-bearing disanalogy:
**we have no PDB.** Mechanisms are unobservable, so the manifold can never be validated the way
folding was. Stated precisely, as three claims of decreasing checkability:

- **M1 — the mechanism manifold is small.** A short vocabulary of survey idioms (item nonresponse,
  skip logic, LOD/top-coding, social-desirability censoring, attrition, unit nonresponse).
  Supportable by domain literature; **not directly confirmable** (mechanisms unobservable).
- **M2 — the footprint manifold is small.** Real survey masks occupy a low-dimensional, structured
  region of footprint-space. **Empirically testable on real masks, no labels needed.** The
  foundation.
- **M3 — the footprint→mechanism map is near-injective *on the real manifold*.** The load-bearing
  claim, and the one that does real work against Molenberghs: his pathological MAR-twin of a real
  MNAR mechanism is bet to lie *off* the manifold — nature doesn't build it. So **the manifold
  hypothesis *is* the prior that places zero mass on the adversarial twins.** Only *indirectly*
  supportable (on semi-synthetic data: do real idioms separate in footprint-space better than
  arbitrary mechanisms?).

Lacuna's task is therefore the **degenerate inverse** (many mechanisms → one footprint → which?),
structurally like protein *design*, not folding; the manifold prior is the regularizer on that
degeneracy. The optimization is **constrained, not global** — MAP estimation with a
manifold-supported prior, where the Lagrange multiplier λ is the strength of the prior in the
non-identified direction. **Sweeping λ *is* the sensitivity analysis.** The manifold, the δ-prior,
and the sensitivity sweep are one object seen from three sides.

**Operationalization: HYBRID** (decided). Enumerate survey idioms top-down → build generators →
*validate* via M2 that their footprints cover the real-survey-footprint manifold. The manifold also
*is* the coverage boundary of §6: distance-from-manifold is the OOD signal; off-manifold ⇒ abstain.

## 3¾. Estimand enters at the reporting layer, not inside the network *(proposed; confirm)*

Robustness is always robustness *of a specific quantity* — a mean, a quantile, a regression
coefficient — and the same δ propagates to each differently, so a robustness number with no estimand
is meaningless. But the object the **network learns** — a calibrated prior over the mechanism /
sensitivity parameter on the survey manifold — is **estimand-free**: it is a property of the
missingness, not of what you compute downstream. Therefore the estimand enters at a **thin,
deterministic propagation/reporting layer** (δ-prior × stated estimand → tipping-point curve), *not*
inside the network. This keeps the hard learned part focused and reusable across all estimands
(UNIX rule), and satisfies §4.8: a human is told her estimand, so the tool is too — at the layer
where it matters.

---

## 4. Non-negotiable commitments (the guardrails)

1. **Never claim to beat non-identifiability.** §2 is a theorem.
2. **Calibration over accuracy.** A well-calibrated uncertain answer beats a confident wrong one.
   High entropy where the problem is non-identified is the target, not an embarrassment.
3. **Abstain, do not confabulate.** Lacuna can only recognize what spans its training (§6). The model
   MUST detect when a footprint is outside its span and degrade to the uninformative prior +
   "escalate to manual review." A confident answer on an out-of-span mechanism is the **worst
   possible output** — it is the exact shape of every past failure.
4. **Semi-synthetic is the only ground truth.** Real missingness is unrecoverable ("missing in real
   life is missing forever"). On real data, accuracy is unmeasurable; only face-validity / consensus
   applies, and must be labeled as such.
5. **Guard the leakage trap.** Our labels are self-assigned. Never let the mechanism label leak into
   the features — especially when semantics inform both the hole-punching and the model input.
   Evaluate on **independently-assigned mechanisms and held-out generator families**, never on the
   generators you trained on. A spectacular in-distribution number is the classic fake win.
6. **Run the REAL model.** A frozen-encoder probe or an RF/MLP-on-statistics is an *ablation*, never
   "Lacuna." Any claim about the network uses the full `LacunaModel` (MoE) via the real pipeline. A
   result that took seconds/minutes instead of ~15–35 min is a proxy. (Carried-over scar tissue.)
7. **Control the rate confound** before any MAR/MNAR comparison. Natural-rate separation has been
   substantially a miss-rate cue; match the rate or the comparison is void.
8. **Never ask Lacuna a question strictly harder than a human's.** A human doing sensitivity
   analysis already knows the data is a survey — not an instrument log or a longitudinal panel —
   and knows what quantity she is estimating. That context is *given*, not inferred. So we *give*
   Lacuna the same context: restrict it to one data regime (see Scope), and supply the downstream
   estimand at the reporting layer (§3¾). Withholding context the human always has, then asking the
   network to recover it from raw numbers, is a self-inflicted harder problem and a violation of
   this charter.

---

## 5. How we know it is working (the RIGHT metrics)

Success is **not** a high in-distribution confusion-matrix number. That metric has misled this
project before; it measures memorization of our own generators. The metrics that count:

- **Out-of-mechanism-FAMILY calibration of the δ-prior.** Train on some departure families, test
  calibration on *unseen* families. This is the transportability bound and the single most important
  number in the project. It is also the direct measurement of the §6 concern.
- **Leave-one-FORM-out within-axis generalization** (the threshold→sigmoid test): train on many link
  shapes of one axis, hold one out, test on the same axis. Distinguishes the *fixable* coverage
  failure from the *fundamental* novel-axis failure.
- **OOD-detection rate:** does the model flag family-novel footprints and abstain, rather than
  confidently mislabel them?
- **MCAR-departure detection** (the identifiable anchor — should be strong and is honest to report).
- **Oracle distinguishability** (no learned model in the loop): the TV/KL distance between the
  observed-data laws of a matched-rate MAR vs MNAR generator. Tells us, per mechanism, whether any
  signal exists *before* we ask whether Lacuna found it. Run this before invoking Molenberghs to
  explain a collapse, and before blaming the model for missing a signal that isn't there.

---

## 6. The coverage boundary (the lesson that must not be re-learned the hard way)

Lacuna can only recognize what spans its training. This is **not a bug to be fixed** — it is the
problem of induction, and no inductive system escapes it. Manage it; do not pretend to abolish it.

- **Level 1 — new functional *form*, same departure axis** (threshold vs. sigmoid self-censoring):
  *should* generalize, because a shared invariant exists. Past failure here was narrow coverage +
  a discriminative objective that learned generator *fingerprints* instead of the *consequence*
  (observed-marginal distortion). Fix with diversity/domain-randomization training and
  consequence-features. Verify with leave-one-form-out (§5).
- **Level 2 — genuinely new dependence *axis*** (a driver absent from all training mechanisms):
  *cannot* be recognized by a discriminative model, by construction. Widening coverage turns many
  Level-2 cases into Level-1 interpolations but never closes the boundary — there is always an
  outside.

The requirement is therefore **not** "generalize to every mechanism" (impossible). It is **"know the
boundary and abstain at it"** (§4.3). A predictable failure the *model detects at runtime* is a
managed risk; a predictable failure it cannot detect is a liability. OOD-detection is the machine
that converts the second into the first.

---

## 7. What would falsify the project

State the kill condition up front, so we notice if we hit it:

> If, even with wide coverage and an explicit abstention mechanism, the δ-prior **cannot be made
> calibrated out-of-family** — i.e., the model cannot reliably tell in-span from out-of-span — then
> Lacuna cannot be a governed, defensible tool, and the thesis fails. Better to find that early and
> honestly than to ship a confident-wrongness machine.

That is the bargain Lacuna makes with the identification theorem: we do not claim to know the
unknowable. We claim to make the *assumption* about the unknowable **explicit, calibrated, auditable,
and cheap** — and to **say so out loud when we are outside what we can support.**
