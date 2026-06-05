# Decision Memo — P2.2c Consolidation: Detectability of LOD on Real Survey X, and the Reframe

*Project-history record + decision memo. Status: **accepted (PI, 2026-06-05)** as the closing
consolidation of the P2.2c (LOD/top-coding) arc and the decision to pause the footprint-learning
push. Governed by `NORTH-STAR.md` (§2 identification line, §3 generalization axis, §8 validation
ladder / flat-likelihood idioms / metadata guardrail). All experiments here are **semi-synthetic
missingness imposed on real survey X** (real values, our holes, known δ) — never natural missingness.*

## 0. The recorded conclusion (the line for project history)

> On semi-synthetic missingness imposed on real survey X, own-value self-censoring appears flat.
> LOD/top-coding is distinguishable at the oracle level and faintly distinguishable by simple OOF
> consequence features, but the current learned δ-prior does not recover a calibrated coarse δ signal
> out-of-family. Thus the detectability spectrum is supported theoretically/oracle-level and weakly
> feature-level, but not yet demonstrated in the held-out learned neural channel.

This is **not** a death certificate for Lacuna; it is enough to **stop this particular tuning arc**.

## 1. own-value self-censoring — demonstrated flat / boundary idiom

Own-value smooth self-censoring (single column, matched rate) is a **flat-likelihood idiom** under
the current footprint (NORTH-STAR §8.2). Ruled out as causes of the floor, across the P2.2b ladder:
head/loss/objective (rung 1 passes on synthetic), target localization (rung 3), per-example scale
(rung 3b, 1024 rows), proxy absorption (R²≈0.02 still floors), cardinality/discreteness, and ordinal
resolution (δ=0-vs-strong AUC ≈ 0.5). The breadth of the rule-out list makes this a **demonstrated
boundary condition about the mechanism**, not an implementation failure.

## 2. LOD/top-coding — oracle PASS

Bayes-optimal oracle (point + β₁′-profiled MAR null) on ConditionalGaussian X-models fit to real
survey (target, predictor) columns: **δ=0 → BE≈0.5; δ≥1 → BE collapses; δ≥2 → BE=0.000 on all 6/6
datasets.** LOD carries strong δ information *in principle*. Honest caveat recorded at the time: a
Gaussian-oracle pass is **necessary, not sufficient** (own-value would likely pass the same oracle;
the real test is the model on real, non-Gaussian X).

## 3. Consequence features — correction and the phase-locking bug

- The first consequence-feature A/B (17 marginal features added to the head) **floored**; I initially
  diagnosed a "generalization gap" with binary OOF AUC **0.43 ("direction flips")**.
- Building the transfer gate I found a **phase-locking bug**: `delta = grid[i%len]` and
  `raw = pool[i%len]` lock each test dataset to one δ when both lists have length 2 → the classifier
  separated **datasets, not δ**. Present in the gate runner and earlier inline binary diagnostics.
- **Corrected:** the 17 features' binary OOF AUC is **0.679–0.734** (transfers coarse δ), **not** 0.43.
  The **7-bin** numbers (coprime periods) and the **neural A/B floors** (random dataset sampling)
  were unaffected and stand. Two mid-investigation hypotheses (z-scoring destroyed signal; model
  under-exploits features) were also **wrong**, corrected by the diagnostics.
- **Transfer-robust feature set** (predictor-conditional / predictor-ruler) was then tried and
  **rejected by the no-training gate**: OOF AUC **0.523 < 0.65**, *worse* than the simple 17 (0.734).
  Predictor-referencing added column-specificity rather than removing it.

## 4. The corrected old-feature coarse OOF signal

The simple 17 marginal features **do** transfer the *coarse* contrast out-of-family — but **weakly**:
LR binary δ=0-vs-δ2.5 OOF AUC **~0.68–0.73**; LR **3-bin** OOF accuracy **0.477** (chance 0.333, below
majority-class ~0.5 — i.e. 3-bin does **not** transfer even for the LR). So the transferable structure
is essentially the *presence/absence of strong truncation*, not calibrated δ magnitude.

## 5. Coarse / curriculum learned-model failure

Held-out coarse A/B (binary + 3-bin; features-only MAIN + encoder-features diagnostic; LOD vs
own-value), against base-rate AND the LR baseline:

- 3-bin neural (both models, both idioms) beat **uniform** (~+2.4 SE) but this is a **base-rate
  artifact** (imbalanced marginal); **none beats base-rate** (−0.2…−0.7 SE). The models predicted the
  marginal. **LOD ≈ own-value** (0.1980 vs 0.1988) — no spectrum separation in the learned channel.
- Binary features-only LOD AUC **0.581 < LR 0.679** — the **learned model underperforms the simpler
  LR** on identical features. Encoder+features ≈ features-only ≈ base-rate (not an integration gap).
- Per the pre-registered rule: *features-only fails to exceed the LR ⇒ the learned model adds no value
  over the simpler formulation.* The coarse milestone is **not** met.

## 6. The three levels of detectability (the key distinction)

| level | what it asks | LOD result | own-value |
|---|---|---|---|
| **Oracle detectability** | Is δ recoverable in principle (Bayes-optimal, known model)? | **Yes** (BE→0) | weak/flat |
| **Feature-level detectability** | Do simple OOF consequence features carry δ (LR, held-out)? | **Faint** (binary AUC ~0.68; 3-bin ≈ base-rate) | ~chance |
| **Learned-channel detectability** | Does a trained, calibrated δ-prior recover it out-of-family above base-rate? | **No** (base-rate; ≤ LR) | No |

The detectability *spectrum* (LOD ≫ own-value) is **real at the oracle level, faint at the feature
level, and not demonstrated in the held-out learned neural channel.** Conflating these three is the
error this memo exists to prevent.

## 7. Why we stop more feature/model tuning now

- The bottleneck is **no longer** "wrong head" or "missing feature": rung 1 proved the head/loss;
  the consequence features expose the signal in-distribution; the corrected gate shows the simple
  features already capture what transfers.
- The **OOF transferable signal is intrinsically weak** (LR binary ~0.68; 3-bin ≈ base-rate). The LR
  is a **ceiling** for any model on these features, and it is low. More feature variants made it
  *worse* (transfer gate); more bins made it *harder* (3-bin doesn't transfer).
- The learned model **does not beat that ceiling**. Continued head/feature/architecture tuning has
  little expected headroom against a low LR ceiling. Stopping is the disciplined call.

## 8. What remains open

- **Other idioms** (skip-logic, attrition, unit-nonresponse, LOD at *unmatched* rate): untested;
  may sit at more detectable / more transferable regions of the manifold (NORTH-STAR §3½).
- **Domain-randomization** (many more train datasets) to strengthen OOF transfer — but the low LR
  ceiling suggests limited headroom; would need a cost/benefit call.
- **Metadata-prior channel** (semantics → δ): admissible later **only** with held-out semi-synthetic
  evaluation (NORTH-STAR §8.3); never unconstrained LLM opinion.
- **Estimand / sensitivity-reporting layer** (PROPOSAL §7): unbuilt — the deterministic δ-prior ×
  estimand → tipping-point report.
- **A formalized identifiability/detectability map**: the oracle + feature-level transfer per idiom,
  as a first-class artifact: not yet built.

## 9. Proposed next strategic path (PROPOSE — DO NOT IMPLEMENT)

Consolidate first; no experiments until the record is clear and the next goal is explicit. The
PI's leaning, recorded as the proposed direction:

1. **Stop treating a neural δ-estimator as the immediate deliverable.** It is not the near-term
   product; it has not beaten base-rate out-of-family on these idioms.
2. **Reframe the near-term contribution as an identifiability / detectability MAP** over missingness
   idioms on semi-synthetic real survey X — per (idiom, regime, column-type) cell, report: the
   **oracle separability** (computable), the **feature-level OOF transfer** (computable), and the
   resulting **recoverability verdict** — **paired with calibrated sensitivity reporting** (the
   estimand layer: δ-prior × estimand → tipping-point). This uses everything we built (generator,
   oracle, leakage gate, manifest, δ-bins, consequence features) **honestly**: it reports *where δ is
   recoverable and where it is not*, with the sensitivity sweep as the governance output. It is
   North-Star-faithful (§3 reframed estimand; §4.3 abstain where unidentified).
3. **Keep learned δ-estimation as a FUTURE, conditional component** — deployed only for cells where
   the detectability map shows **sufficient transferable signal** (e.g. detectable idioms not yet
   tested), evaluated on the held-out semi-synthetic ladder.
4. **Metadata priors remain a later option**, gated by held-out semi-synthetic evaluation (§8.3) — no
   unconstrained LLM expert opinion.

**Recommended immediate action:** none beyond this consolidation. The next concrete proposal (if the
PI adopts the reframe) would be a short spec for the **detectability-map + sensitivity-report
artifact** — written, reviewed, and approved before any implementation.

## 10. Status / discipline

Code on branch `p2/delta-prior-rearchitecture`; full suite green (1346 passed, 1 skipped); no
architecture/objective change was made to reach these conclusions; the phase-locking bug was found,
fixed, disclosed, and its effect on prior claims corrected. Findings docs:
`feasibility-p2p2c-lod-oracle-findings`, `-lod-ladder-findings`, `-consequence-features-findings`
(with correction banner), `-transfer-gate-findings`, `-coarse-ab-findings`. Memory updated.

**Decision of record:** pause the LOD footprint-learning push; record the scoped boundary; consolidate
toward a detectability-map + sensitivity-reporting reframe; no further experiments until the next goal
is specified and approved.
