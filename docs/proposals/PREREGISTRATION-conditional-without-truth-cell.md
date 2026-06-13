# Pre-Registration — The Conditional-Without-Truth Cell (own-value primary)

*Phase-1 **specification only** — no implementation, no runs, no synthesis amendments (binding; PI
2026-06-09). Pre-registers the measurement of the **empty cell** identified by
`ADVERSARIAL-REVIEW-four-level-hierarchy.md` §3: own-value, conditional structure preserved, truth
removed. Locked before any code exists; thresholds may not be revised after results. Governed by the
project's comparison-class discipline and the G1 protocol (`PREREGISTRATION-G1-imputation-channel.md`).*

---

## 0. Why this cell decides the interpretation of G1 (and why nothing already measured can)

G1 established that a channel with **truth + conditional structure** recovers own-value at 0.90–0.99
OOF, against a marginal-φ baseline of 0.574 — a +0.33–0.42 increment. That increment **confounds two
ingredients** removed together at Level 3: (i) the realized deleted values (truth) and (ii) the
conditional (target | predictors) representation that φ discards. Two interpretations follow:

- **H-truth (truth-bottleneck):** the increment is carried by truth; conditional structure without it
  collapses toward φ. ⇒ own-value's wall is **identification**; the lattice stands as amended.
- **H-cond (conditional-structure):** the increment is substantially carried by the conditional
  representation; much of it survives truth removal. ⇒ the 2→3 collapse was partly a
  **representation gap** (φ is marginal-only), the "identification wall" weakens, and the deployment
  pessimism for own-value is partly an artifact of marginal-only features.

Nothing measured discriminates these: G1 confounds them; `transfer_features` (0.523) is top-coding-only,
cross-era, contaminant-corpus (review §3); the oracle is mechanism-informed (a different apex). **The
cell is not merely missing — it is the unique measurement that separates the two readings of the
project's most important result.** Its outcome also determines whether the substitutes theorem (S2) is
complete or premature.

## 1. Operational definition — the no-truth-conditional station

**The observer receives exactly one object: the mechanism-masked observed view** of a semi-synthetic
example — `(x_obs, mask)` where `x_obs` carries values only at observed cells (predictors complete,
target punched by the mechanism). The observer may compute **anything** from this view, including
fitting arbitrary conditional models of (target | predictors) on observed rows and self-punching
additional holdout cells **from observed cells**. The observer may **never** access:

- the complete matrix or any value at a mechanism-punched cell (the realized truth — Level 2's input);
- **any alternate-mask view of the same rows** (see §4-L1: the paired-MCAR view *contains the deleted
  values* — cross-view comparison is truth access in disguise);
- the answer sheet (δ, idiom), except as the LR *label*, identically to every other station's eval.

**Interface contract (binding for the future implementation):** the feature extractor's signature takes
`(x_obs, mask)` only — structurally incapable of truth access — and a test must assert it never reads a
mechanism-punched cell.

## 2. How the station differs from its neighbors

| station | sees | conditional access | truth access | features (locked) |
|---|---|---|---|---|
| φ baseline (L3, measured: 0.574) | mech view | **no** (marginal only) | no | frozen 17 consequence features |
| **THIS CELL** | mech view | **yes** (fit target‖predictors; self-holdouts) | **no** | §3 groups H/S/R |
| G1 channel (L2, measured: 0.90–0.99) | complete matrix + both masks | yes | **yes** (deleted values; cross-view pairing) | 6 paired Δ-statistics |

The φ-vs-cell contrast isolates **conditionality**; the cell-vs-G1 contrast isolates **truth** —
provided the protocol matches G1 in everything else (§3) and no leak exists (§4).

## 3. Design (locked)

**Maximal protocol identity with G1:** same corpus (4 domains, 40 continuous targets), same cells
(dataset × target × idiom × δ ∈ {0, 2.5}), same 24 examples/cell, **identical example RNG seeds**
(the 31000+i scheme) so the realized rows and mechanism masks are *bit-identical* to G1's, same matched
rate 0.3, β₁ = 1.0, same 5 imputer configs (fit on observed rows of the mech view), same 4-domain
block-aware leave-one-domain-out LR, same pooled-OOF arbiter. The only change: **the features**.

**Feature groups (per example, per imputer; all from the mech view only):**
- **H — holdout calibration (the within-view control):** self-punch `k = min(|M|, n_obs − 30)` MCAR
  holes into **observed** target cells (deterministic per-example RNG); **re-fit the imputer excluding
  the holdout** (in-sample-optimism guard, §4-L5); compute the six G1 statistics (B, PIT_loc, PIT_tail,
  B_top, W1, cov80) **at holdout cells** (their truth is legitimately known — it was observed). H
  measures the imputer-column baseline; it replaces G1's cross-view MCAR pair as the
  intrinsic-unpredictability control.
- **S — imputation-shift:** statistics of imputed values `ŷ` at the **real** punched cells vs in-view
  references: mean(ŷ_mis) − mean(y_obs); W1(ŷ_mis, y_obs); W1(ŷ_mis, ŷ_holdout); tail mass of ŷ_mis
  above q90(y_obs); mean predictive σ at punched vs holdout cells.
- **R — residual-selection signature (the own-value conditional shadow):** on observed rows, residuals
  e = y − μ̂(predictors): skewness, spread ratio (q95−q50)/(q50−q05), reach deficit — the
  `transfer_features` Group-A statistics, **now finally applied to the idiom and corpus they were
  conceptually aimed at**. Under own-value censoring the observed residuals are upper-truncated;
  under MAR(β₁) they are not.
- Anchors: realized missing rate (matched ⇒ no δ cue), corr(target, predictor).

≈ 16 features per imputer; standardization of the target uses **mech-observed cells only** (§4-L2).

**Tasks:** primary — **own_value δ0-vs-δ2.5**, cell-features-only, pooled OOF AUC per imputer; plus the
φ-increment variant (base17 + cell features). Secondary — **top_coding** same tasks (this *replaces* the
cross-era `transfer_features` import and completes the matched 2×2 in the same run); idiom-separation at
δ=2.5 (reported, not gated).

## 4. Leakage red-team — vectors and preventions (locked)

| # | leak vector | prevention |
|---|---|---|
| L1 | **Cross-view pairing** — comparing the mech view to the paired-MCAR view of the same rows: the MCAR view's observed set **contains mechanism-deleted values**, so any cross-view statistic (e.g., W1 of the two observed marginals) reconstructs the truth signal | **Forbidden by definition** (§1). The control role moves to the within-view holdout (H). This is itself a pre-registered insight: G1's pairing is part of truth access, not a neutral control |
| L2 | **Complete-column standardization** — z-scoring the target with full-column moments imports deleted-tail information | standardize with **mech-observed** moments only |
| L3 | **Feature extractor touching punched cells** — any statistic indexing `y` at mech-missing positions | interface contract (§1) + a unit test asserting structural inaccessibility |
| L4 | **Holdout punched from the wrong set** — holdout cells drawn from missing positions would have unknown "truth" or, worse, known-only-to-the-experimenter truth | holdout drawn from **observed** cells only |
| L5 | **In-sample PIT optimism** — scoring holdout cells with an imputer fit on data including them inflates calibration features | **re-fit excluding the holdout** before scoring H |
| L6 | **Generator-side knowledge** (δ-dependent β₀ solving, target selection use complete data upstream) | identical for every station incl. the φ baseline; upstream of the observer; not station-distinguishing — recorded, accepted |
| L7 | **Label leakage via realized rate** — if rates differed by δ the LR could read rate | matched-rate regime as in G1; rate included as an anchor feature precisely to verify it carries no signal (its LR weight is reported) |
| L8 | **Seed coupling** — sharing G1's example seeds is mask-identity, not information (masks are upstream); but the **holdout** RNG must be independent of δ/idiom | holdout seeds derived from the example index only |

## 5. Hypotheses and locked decision bands

**H₀ (null) = truth-bottleneck** — consistent with the project's documented prior (`transfer_features`
failure on top-coding; the E-study divergence; the proposal's own pessimism about truth-free signals).
**H₁ (challenger) = conditional-structure.** Burden of proof on H₁.

**Primary quantity:** per imputer *i*, the **recovery fraction**
`ρ_i = (AUC_cell,i − 0.574) / (AUC_G1,i − 0.574)`, with locked anchors: φ baseline 0.574 (G1 run) and
per-imputer G1 truth-channel AUCs (linear 0.900, rf 0.989, gbm 0.993, nn 0.994; mice ≡ linear, so 4
distinct classes).

| outcome band (per imputer) | locked range | interpretation |
|---|---|---|
| **truth-bottleneck** | ρ ≤ 0.25 **or** AUC_cell ≤ 0.62 | conditional structure without truth recovers ≤ a quarter of G1's increment / is within noise of φ |
| **conditional-structure** | ρ ≥ 0.60 **or** AUC_cell ≥ 0.80 | the majority of G1's increment survives truth removal |
| **mixed** | 0.25 < ρ < 0.60 | both ingredients carry real weight |

**Stability rule:** the experiment's verdict is the band on which **≥ 3 of the 4 distinct imputer
classes** agree; otherwise the verdict is **"imputer-dependent (mixed)"** and is reported as such — not
resolved by post-hoc class selection. Secondary (top-coding) bands use the same formula with its anchors
(φ 0.633; G1 tc 0.885–0.947).

**Comparison-class discipline (binding):** every claim is relative to *this feature family (H/S/R) and
imputer class on this corpus at δ = 2.5, matched rate 0.3* — the cell is a **lower bound** on the
no-truth-conditional station. A truth-bottleneck outcome therefore reads: *"generic conditional
statistics of the named family do not recover the shadow"* — never "no conditional information exists"
(the mechanism-informed oracle already shows mean observed-data information exists; that is the other
apex, not this station).

## 6. Outcome branches → consequences for the lattice and the project

- **Truth-bottleneck (H₀ holds):** the lattice amendment proceeds *strengthened*: the substitutes
  theorem (S2) becomes complete — *truth or mechanism assumptions; with neither, conditional **and**
  marginal statistics sit on the same floor* — now measured, not inferred. S1 upgrades from "inferred"
  to "isolated." The own-value **identification wall** stands; G2's prior worsens further (its best-case
  input family just floored under ideal eval conditions); the dissertation's epistemology claim
  (the lab coat covers absent identification) gains its missing leg. The 2×2 becomes a complete,
  protocol-matched, publishable result.
- **Conditional-structure (H₁ wins):** **major revision.** The 2→3 collapse is substantially a
  *representation* gap; "own-value is identification-limited" is overclaimed; the deferred Level-2
  conditional-reference question (Gate II) reopens with direct evidence; a truth-free informative signal
  family exists *in principle* (whether it deploys is still a separate, later question — explicitly not
  designed here). The lattice gains a mid-level station between floor and apexes; S2's "neither ⇒ floor"
  clause is falsified as stated and must be weakened to "neither ⇒ mid-level."
- **Mixed:** the lattice's floor splits (marginal floor < conditional mid-rung < apexes); the lab-coat
  fraction becomes a **two-term decomposition** (representation share vs identification share of the
  deployment gap) — arguably the richest outcome for the dissertation; both walls remain but with
  measured partial heights.
- In **every** branch, the secondary top-coding cells replace the cross-era `transfer_features` datum,
  closing the adversarial review's §3 demotion with a matched measurement.

## 7. Guardrails & execution plan (on PI approval only — nothing now)

No synthesis/lattice amendments until this result exists (PI instruction) · no runtime/proxy/deployment
claims in any branch (a conditional-structure outcome licenses *eval-level* statements only) · no
architecture changes · no imputer/feature/threshold tuning after results · failures and ambiguities
reported as-is · the verdict uses only §5's locked bands and stability rule.

Execution (Phase 2, gated on PI go): (1) extend `imputation_channel.py` with the no-truth extractor
(interface contract + leak test); (2) `scripts/run_conditional_without_truth_cell.py` reusing G1's cell
enumeration/seeds verbatim; (3) findings doc scoring §5 and the §6 branch; stop for review. Estimated
cost ≈ G1's (the imputers dominate; one extra re-fit per example for L5).

---

*Specification only. No code, no runs, no amendments. The cell is pre-registered as the measurement that
decides between the truth-bottleneck and conditional-structure readings of G1 — with the leak that would
have silently invalidated it (cross-view pairing) identified and excluded by construction.*
