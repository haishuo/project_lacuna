# Option C — Column-Primary Distributional Lacuna-Survey (clean-sheet design spec)

*Design spec only. **No implementation, no runs.** Status: **for PI review.** Designed from the
inference target backward (Architecture-Change-Audit §6/§7). Governed by `NORTH-STAR.md` (§2
identification, §3 generalization axis, §3½ manifold, §6 coverage/abstention, §8 ladder) and motivated
by `probe-encoder-representation-findings.md` (the row-primary averaged backbone does not transferably
surface the within-column order-stat signal; a column-primary ECDF representation carries 0.722 OOF
where the backbone carries ~0.52).*

## 0. Thesis

The inference target changed from **mechanism classification** (cross-column mask structure → MAR
signature; BERT-over-columns is a good prior) to **δ-prior / missingness-consequence inference**
(within-column distributional distortion vs a reference; an order-statistic / distribution-native prior
is needed). Option C **inverts the primary axis**: from *row-as-sequence, cross-column attention, then
average* to *column-as-distribution-object, set-encode each column's observed values, then compare to a
predictor-conditional reference*. It is **mechanism-general** (no LOD/idiom detector) and stays on the
semi-synthetic → held-out ladder.

## 1. Inference target (the fixed point everything is designed to serve)

Given a survey table `X[n,d]`, observed mask `R[n,d]`, a supplied target column `t`, and (eventually)
column metadata: return a **calibrated prior over the sensitivity parameter δ** for `t` — ordered bins
or continuous + uncertainty — **plus an abstention/coverage signal** when the footprint is out-of-span.
Validity is measured on **held-out semi-synthetic survey tasks** (ladder rung 2), leakage-gated,
oracle-anchored. δ is never claimed beyond the oracle ceiling.

## 2. Explicit intermediate objects (the architecture is organized around these, not around tokens)

1. **Per-column empirical distribution object** `D_j` — for each column `j`, a permutation-invariant-
   over-rows embedding of its **observed values** that *preserves order statistics* (this is precisely
   what the current backbone destroys). Built from a sorted/ECDF/quantile representation, not an average.
2. **Mask-topology object** `M` — per-column miss rates + cross-column co-missingness (revives the
   *concept* of the v1.0 `MissingnessFeatureExtractor`, which the original design already trusted as an
   explicit object). Carries the MCAR/MAR-identifiable structure (§2 solid ground).
3. **Predictor-conditional reference** `Rref_t` — the distribution of `t` **expected under MAR given the
   observed predictors**: what `D_t` *would* look like if missingness depended only on observed columns.
4. **Consequence (observed-vs-reference deviation)** `Δ_t = D_t ⊖ Rref_t` — the distributional distortion
   the censoring left behind. **This is the object δ lives in, and nothing in the current architecture
   represents it.**

## 3. Backbone (two axes, distribution-native)

- **Within-column encoder** `φ`: observed values of column `j` → `D_j`. Candidate primitives (decide at
  design-review, all order-statistic-preserving): a fixed/learnable **ECDF/quantile embedding**, a
  **sorted-set / DeepSets with sorting-or-quantile pooling**, or a **Neural-Statistician** distribution
  encoder. Permutation-invariant over rows by construction; scale-handling explicit (within-observed
  standardization, as `consequence_features` already does). Shared `φ` across columns (UNIX reuse).
- **Across-column set encoder** `ψ`: a **set transformer over the column objects** `{D_j} ∪ M`, so the
  target's `D_t` attends to predictor columns' `D_{≠t}` and the mask topology — this is where `Rref_t`
  and the deviation `Δ_t` are formed (the conditional reference is a learned function of the predictor
  column distributions, evaluated on the held-out ladder, never asserted).
- **Reference-deviation head**: consumes `Δ_t` (+ `M`, + `D_t`) → δ-prior logits + an OOD/abstention
  score (distance-from-training-manifold, §6). Loss: the existing RPS over δ-bins (reuse `loss.py`),
  plus a calibration/coverage term and an abstention objective.

Analogy: **Neural Statistician / DeepSets-over-distributions + set-transformer-over-columns + two-sample-
deviation head** — *not* BERT. The row axis is consumed *inside* `φ` (as an exchangeable set), not as a
sequence with cross-column attention.

## 4. What it can / cannot represent

- **Can:** per-column empirical distributions and their order statistics (natively, not via averaging);
  predictor-conditional reference + observed-vs-reference deviation (idiom (a)'s missing object); mask
  topology (the identifiable anchor); abstention/coverage. Mechanism-general — describes *distributional
  distortion*, shared by LOD, top-coding, self-censoring, plausibly skip/block.
- **Cannot (by theorem, and correctly so):** beat non-identifiability — where `Δ_t ≈ 0` at matched rate
  (own-value flat idiom), the correct output is a wide high-entropy prior (§3/§4.2). Option C does not
  change the oracle ceiling; it changes whether the *learned channel* can reach the *observable* signal.

## 5. Why this is not Lacuna-LOD (mechanism-generality guardrail)

No τ, no step indicator, no truncation-edge detector, no idiom label enters the architecture. It
represents *distributions and their deviations*; LOD is merely one idiom whose deviation is sharp.
Generality is **tested**, not assumed, exactly as before: the same architecture must keep **own-value
flat** while recovering detectable idioms. Building a truncation-edge feature would violate this and is
out of scope.

## 6. How it stays on the validated paradigm

Reuses the entire evaluation spine unchanged: semi-synthetic generators (`delta_generator`,
`lod_generator`), δ-bins, RPS loss, **leakage gate**, manifest, the **oracle** (`lod_oracle` /
profiled MAR null) as the pre-train ceiling, and the held-out leave-datasets-out LOD-vs-own-value A/B as
the test. The estimand/sensitivity-reporting layer (PROPOSAL §7) sits downstream, unchanged.

## 7. Cost, risk, and what would gate the build

- **Cost: high** — a new within-column encoder, a new across-column set encoder, a new data path
  (column-major), new tests (Rule 7). Tokenization is reconsidered (column-value sets, not per-cell
  4-tuples) — the one place a rewrite is genuinely warranted (audit §5).
- **Risk: medium-high** (more moving parts) but it is the **only option that targets the inference
  object directly** and the only one that *earns* the re-architecture claim.
- **Pre-build gate (decisive, cheap, recommended BEFORE full build):** a **column-primary
  representation probe** — confirm that a distribution-native encoder `φ` on the *target column alone*
  (e.g. a learnable ECDF/quantile embedding) recovers the LOD signal OOF at/above the raw-ECDF baseline
  (~0.72), while own-value stays flat. This is the positive-capable mirror of the §8 probe: if a
  column-primary `φ` clears the baseline where the BERT backbone could not (~0.52), the rewrite is
  fully earned and the across-column reference module is the next increment. If even `φ` cannot clear it
  OOF, the bottleneck is the **transfer ceiling / regime** (DECISION-MEMO §9), *not* the architecture —
  and we would not pour effort into the full Option C. **This gate is analysis, not the rewrite.**

## 8. Staged plan (each stage gated; nothing built until the prior stage's spec is approved)

1. **Stage 0 (analysis):** the column-primary representation probe of §7 (does a distribution-native
   `φ` clear the OOF baseline?). Decides whether Option C is worth building at all.
2. **Stage 1 (spec→build, if Stage 0 positive):** `φ` (within-column distribution encoder) + a minimal
   target-only δ-head; A/B vs raw-ECDF-LR and vs the current model on the held-out ladder.
3. **Stage 2:** add `ψ` (across-column set encoder) + the reference-deviation module `Δ_t`; test whether
   the conditional reference improves OOF over the marginal `φ` (and whether own-value stays flat).
4. **Stage 3:** mask-topology stream `M`, abstention/coverage head, calibration; full governance output.
5. **Stage 4:** estimand/sensitivity-reporting layer integration.

## 9. Reuse-vs-rewrite recommendation (for PI decision)

The §8 probe makes a **head-only patch unlikely to suffice** (the backbone's averaged reps don't carry
or preserve the signal; the rep-ECDF patch confirmed this end-to-end). The recommendation is therefore:

- **Do not** scale or further patch Option B.
- **Do** run **Stage 0** (column-primary representation probe) as the next analysis step — it is the
  cheapest decisive evidence for whether a distribution-native architecture can clear the OOF baseline.
- **Build Option C only if Stage 0 is positive**, on a clean architecture branch, staged as in §8.
- If Stage 0 is negative, the constraint is the **transfer ceiling / regime**, and the right move is
  domain-randomization / the detectability-map reframe (DECISION-MEMO §9), not a rewrite.

**No implementation until Stage 0 is run and the PI approves the Stage-1 spec.**
