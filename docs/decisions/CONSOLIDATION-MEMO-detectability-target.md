# Consolidation Memo — Why Oracle Informativeness Is the Wrong Runtime Detectability Target

*Consolidation document — **analysis only; no design work, no replacement head, no experiments**
(binding; PI 2026-06-08, accepting the E-JUSTIFY/E-FALSIFY outcome). Explains why the study invalidated
the assumption — held since `PROPOSAL-Level1-design-spec.md` §3 — that **oracle informativeness is the
correct calibration target for a runtime detectability signal**. Governed by `NORTH-STAR.md` and
`MASTER-lacuna-survey-architecture.md` (updated alongside this memo). Evidence:
`E-JUSTIFY-E-FALSIFY-findings.md`, `runs/detectability_study.json`.*

---

## 0. The decision being consolidated

The Stage-2 detectability head is **removed from the build plan** (PI, 2026-06-08). "Detectability" is
split into two quantities that the design had conflated:

1. **Oracle identifiability** — *can a Bayes-optimal observer, knowing the true mechanism family,
   distinguish this cell from its best plausible MAR competitor?* An **evaluation quantity**: computable
   only where truth is known (semi-synthetic cells), comparison-class-relative, used for eval-time claims
   (idiom spectrum, "a no is a result only if oracle-gated", the lab-coat fraction). **Solid; retained.**
2. **Deployed-channel reliability** — *can the trained, transferable δ-prior be trusted on this column?*
   The quantity a **runtime** governance signal must estimate. **An open research problem** — currently
   estimated by nothing we have validated (coverage gates its support; nothing estimates its in-manifold
   variation, per F4).

No replacement head is being designed. This memo records *why* the conflation failed, so the next attempt
does not repeat it.

## 1. The assumption, as originally held

The Level-1 design spec (§3, PI-approved 2026-06-05) defined the detectability head as a learned map from
the model's representation to **oracle informativeness**: *"for each training example with known (idiom,
dataset, δ, rate) we compute the oracle informativeness … and regress the detectability head to it."* The
derived-stack alternative (`I_gain = KL(posterior ‖ prior_marginal)`) was likewise to be **validated by
agreement with the oracle** (E-study criterion F1). Both presuppose:

> **(A) The oracle's ranking of cells by informativeness ≈ the deployed channel's ranking of cells by
> reliability** — so that estimating the former is a usable proxy for governing the latter.

This was reasonable ex ante: the oracle is the project's only ground truth with known δ; the oracle-gating
discipline ("a flat result is a result only at the Bayes-optimal level") had served the project well; and
the recorded caveat ("oracle is necessary-not-sufficient — a ceiling") seemed to make the proxy
conservative. The error was subtle: **a ceiling is a safe proxy only if the gap to it is roughly uniform.
If the gap varies across cells, the ceiling's ranking is uninformative — or inverted.**

## 2. What the study measured

Three pre-registered measurements, all leave-domain-out, coverage-gated, 16 frozen-recipe subjects:

1. **F1:** Spearman(`I_gain`, `I_oracle`) over gated-in held-out cells = **−0.03 ± 0.02** — the calibrated
   posterior's information gain carries **zero rank information** about oracle informativeness.
2. **F2 (the sign mismatch):** the learned channel separates idioms as top_coding > own_value (+0.031 —
   the spectrum every prior result shows), but the **oracle separates them the other way** (own_value
   **more** distinguishable: Δ = −0.083; e.g. at δ = 0.75, own_value I = 0.75 vs top_coding I = 0.60).
3. **J1/J2:** even a probe trained *directly on the oracle target* from the φ representation recovers only
   **0.17** rank agreement OOF and **cannot separate the idioms** (+0.006) — the representation itself does
   not transferably encode the oracle's ordering.

So assumption (A) is false in the strongest available sense: not noisy agreement, not attenuated
agreement — **rank-zero agreement with an inverted idiom ordering**, and the inversion is not an estimator
artifact (the probe aimed at the oracle cannot find the ordering either).

## 3. Why the divergence is real, not a measurement accident

The two quantities are properties of **different objects**, and the project's own history says the gap
between those objects is large and non-uniform:

- **Oracle informativeness is a property of the data-generating cell** under the true mechanism family.
  The Bayes-optimal LLR *knows the parametric mechanism* and can **accumulate a diffuse, per-row
  likelihood tilt across all n = 384 rows**. Own-value self-censoring produces exactly such a tilt: tiny
  per-row evidence, but the optimal test integrates it — hence own_value's high oracle distinguishability
  (fully consistent with P1, where the own-value signal *survived* the profiled MAR at the Bayes level; the
  "flat-likelihood idiom" finding was always about the **learned/transferable** channel, never the oracle).
- **Deployed-channel reliability is a property of the trained estimator** — what a generic,
  mechanism-agnostic, *transferable* representation actually extracts. Top-coding leaves a **localized,
  structural** footprint (a truncated upper tail) that a distribution encoder learns and transfers;
  own-value's diffuse tilt requires the true model to accumulate and does **not** survive learning +
  transfer (P2.2c; Stage-0; D3).
- Therefore the **oracle→channel gap varies by idiom and cell** — small for structural footprints, enormous
  for diffuse ones. A varying gap means the ceiling's *ranking* does not survive projection onto the
  channel: the cells the oracle ranks highest (own-value at moderate δ) are precisely cells where the
  channel has nothing. This is the **three-levels result** (oracle / feature / learned, DECISION-MEMO
  P2.2c) re-manifesting *inside* detectability — we had established the levels diverge for *δ-estimation*
  and then, in the detectability design, implicitly assumed they coincide for *informativeness*.

**The governance consequence that makes this fatal rather than academic:** a runtime signal calibrated to
the oracle would assign **high detectability to own-value cells** — telling the analyst "the data speak
here" exactly where the deployed channel is uninformative. That is the *confidently-wrong* failure mode
(D3 §5) reproduced at the meta level: the safety signal itself would be confidently wrong. A detectability
signal aimed at the wrong target is worse than none.

## 4. One honest tension resolved, one sharpened

- **Resolved:** MASTER §3's comparison-class story said own-value is "flat because its MAR competitor is
  genuinely plausible and reproduces the footprint." At the *Bayes* level this is overstated — the profiled
  MAR does **not** fully reproduce the footprint (profiled BE ≪ 0.5 at n = 384; P1 said the same). The
  split fixes the language: own-value is **identifiable-in-principle at the Bayes level under the fitted
  X-model, but unlearned/untransferable so far** — its flatness is a fact about the channel, not about
  identification. (The lab-coat-fraction claim survives in sharpened form: for sensitive items, *deployed,
  semantics-free* analysis is uninformative — that is what governs practice — while the in-principle
  ceiling is a separate, eval-time statement.)
- **Sharpened:** "detectability" claims must now name **which quantity** (identifiability vs channel
  reliability) *in addition to* the comparison class. The N-S §2 per-mechanism spectrum
  (top-coding detectable / own-value flat) is a statement about the **channel**, and it remains correct.

## 5. Scope and caveats of the invalidation

- Established at n = 384, rate 0.3, fitted-Gaussian 2-col X-models, top-coding-trained subjects, binary
  δ-scheme. The **size** of the rank divergence may differ elsewhere; the **direction** of the conclusion
  (calibrate runtime signals to the deployed channel, keep the oracle for identifiability) does not depend
  on it — it follows from the varying-gap argument (§3), which the study confirmed in the regime we
  actually deploy.
- The oracle infrastructure is **not** diminished — it is *re-scoped to where it is valid*: eval-time
  identifiability, the idiom map, oracle-gated negatives, the lab-coat fraction.
- This memo does **not** conclude that a runtime reliability signal is impossible — only that (i) oracle
  informativeness is the wrong target for it, (ii) `I_gain` does not track even the wrong target, and
  (iii) coverage gates its support but not its in-manifold variation (F4). Whether a deployed-channel
  target (e.g., held-out δ-prior calibration over `P_prior` cells) is estimable at runtime is the **open
  problem** — explicitly not designed here.

## 6. What changes in the design of record (enacted in MASTER alongside this memo)

1. **Stage-2 detectability head: REMOVED** from the build plan. Re-entry only via a new pre-registered
   spec whose target is a **deployed-channel reliability** quantity — never oracle informativeness, and
   never the gate role (closed to learned components; self-reference argument + D3).
2. **Per-column runtime output becomes** { MCAR-departure, δ-prior, **coverage-state**, **UNKNOWN** } —
   no runtime detectability/info-gain claim.
3. **Eval-time reporting keeps** oracle identifiability (comparison-class-named) — the manifest carries it
   as an evaluation block, not a runtime head output.
4. **Open-problem registration:** deployed-channel reliability estimation — with F4 as its first concrete
   datum (in-manifold reliability varies in ways coverage does not capture).

---

*No design, implementation, or experiment is proposed by this document. It records the reason the
oracle-as-runtime-target assumption failed — a varying oracle-to-channel gap that inverts rankings — so
that any future runtime detectability proposal starts from the deployed channel, not the ceiling.*
