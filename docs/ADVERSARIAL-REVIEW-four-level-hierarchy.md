# Adversarial Review — How Much Weight Can the Four-Level Hierarchy Bear?

*Specification-only adversarial review (PI 2026-06-09). **No new experiments, no implementation, no G2.**
Pressure-tests `CONSOLIDATION-four-level-information-hierarchy.md` against the repository's actual
records (scripts, findings docs, run artifacts, git history). Where the review finds errors in the
synthesis, it says so plainly. Verdict first.*

---

## 0. Verdict

**The hierarchy survives — in an amended form that is weaker in one place and *stronger* in two.**

- **DEMOTED:** the `transfer_features` cell of the truth × conditionality 2×2 is **not a valid isolation
  experiment** for the claim as stated (§3). It is a *top-coding* measurement, on a pre-reset corpus
  containing a contaminant, with a labor-only test split — and the **own-value conditional-without-truth
  cell is EMPTY**. The synthesis's table misattributed it. "The binding 2→3 loss is truth, not
  conditionality" is **protocol-matched-supported only via the φ-increment inside G1** and *suggested*,
  not isolated, by `transfer_features`.
- **CORRECTED (and strengthened):** the hierarchy is **not a chain — it is a lattice.** Level 1 does not
  contain Level 2 (the oracle knows the mechanism *law*, never the realized deleted values; G1 knows the
  values, never the mechanism). They are **two incomparable apexes over one observed-data floor** — and
  this is now *empirically demonstrated*, not just conceptual: the existing oracle artifact shows
  own-value cells with `I_oracle` as low as **0.08–0.12 even at δ = 2.5** (near-flat at the Bayes level),
  on a corpus where G1's truth channel is uniformly high. **There exist cells where truth ≫ mechanism
  knowledge.** Truth and mechanism assumptions are *substitutes, not stations on one ladder.*
- **What bears publishable weight now:** the *protocol-matched core* — everything measured inside G1
  (channel vs φ on identical examples, splits, arbiter) plus the E-study oracle on the *same corpus,
  same n, same rate, overlapping δ*. The full 2×2 narrative does **not** yet bear weight (one cell
  empty, one imported cross-era).

## 1. Q1 — Alternative explanations for G1 besides "identification-limited, not information-limited"

| # | alternative | status after review |
|---|---|---|
| A1 | **Slice extremity:** G1 ran only δ ∈ {0, 2.5}; the "1→2 near-lossless" and "L2 ≈ 0.99" claims might collapse at interior δ where the oracle itself is at 0.6–0.9 | **LIVE.** Nothing contradicts it; nothing supports extrapolation. All hierarchy claims must carry the δ=2.5 qualifier. (Interior-δ L2 was *not* run — G1's grid was binary by locked design.) |
| A2 | **L2 near-tautology:** "with truth you can see what was deleted" — a value-dependent deletion mechanism is L2-visible *by construction*, so high L2 is definitionally guaranteed and carries no surprise | **PARTIALLY CONCEDED — and absorbed.** The proposal itself pre-registered this ("near-tautological at the truth level"). The non-tautological content is (i) *transfer* (leave-domain-out incl. wealth 0.98 — nothing guarantees the paired statistics transfer across regimes), (ii) the *φ-increment* (+0.30–0.42, protocol-matched), (iii) *estimator-class robustness* (even linear, in-family). The hierarchy's surprise must be located there, never in "L2 is high." |
| A3 | **"Information" is estimator-relative:** every level number is the LR-recoverable AUC of one *specific feature family* — L3 = 0.574 is a fact about 17 particular marginal statistics, not about the observed data | **CONCEDED AS A BOUND, then bounded in turn.** All level numbers are *lower bounds* on the station's information. But the oracle bounds the other side: for own-value, observed data demonstrably *contains* recoverable information (mean `I_oracle` 0.73–0.93 at δ ≥ 0.75) that generic statistics do not reach — so the L3 floor is a statement about *assumption-free* recovery, which is exactly the deployment-relevant statement. The gap could still narrow with better generic features; it cannot be assumed away. |
| A4 | **Regime conditionality:** rate fixed at 0.3, β₁ = 1, n ≈ 384, |M| ≈ 115. At low rates L2 degrades mechanically (fewer punched cells); at small n everything degrades | **LIVE as scope, not as refutation.** The orderings are large (0.99 vs 0.57); plausible regime variation shifts magnitudes, not orderings, but this is asserted, not measured. |
| A5 | **The paired design reads "MAR-ness," not MNAR:** at δ=0 the mech mask is MAR(β₁), and the channel already shows ΔB ≈ −0.19 vs MCAR — the LR may exploit a *generic selection-strength gradient* rather than mechanism information per se | **MINOR / ABSORBED.** The δ=0 class *is* the MAR null, so separating δ=2.5 from it on a selection-strength gradient *is* the task. It does caution against reading the channel as a mechanism *classifier* beyond what G1-c measured (0.68–0.83 — the weakest of the three gates). |
| A6 | **Effective-n optimism:** 24 examples per (dataset, target) cell share rows/targets; pooled OOF AUC has no stated CI and the effective sample is ~40 targets, not ~3,840 rows | **CONCEDED as a reporting gap.** The per-held-out-domain table (4 independent splits, all ≥ 0.97 for flexible imputers) is the real evidence; the 0.99-vs-0.57 gap dwarfs any plausible clustering correction. A publishable version needs cluster-aware uncertainty. |

**None of the alternatives rescues "own-value is information-limited."** The strongest live caveats are
A1 (δ-extremity) and A3-as-scope (assumption-free vs in-principle recovery).

## 2. Q2 — Which assumptions do the most work in the 2×2?

Ranked by load:

1. **Cross-protocol comparability of cells (the heaviest, and it FAILS for one cell).** Only three of
   the four populated stations are protocol-compatible: G1's two cells and the E-study oracle share the
   corpus (same 4 domains + anchors), n = 384, rate 0.3, and δ = 2.5. The `transfer_features` cell is an
   import from a different era: pre-reset corpus (TRAIN included `survey_computers`, later excluded as a
   **contaminant**), labor-only TEST (`cps1985`/`workinghours` — not the 4-domain LODO), a different
   feature family, and **top-coding only** (§3).
2. **The chain/nesting claim** — wrong as written in the synthesis ("strict information ordering by
   construction"). L1 ⊉ L2: the oracle never sees the realized deleted values. Nesting holds only for
   2 ⊇ 3 ⊇ 4. The lattice correction (§0) is mandatory.
3. **"Information" = LR-AUC of a fixed feature set** (A3 above) — all numbers are lower bounds.
4. **δ = 2.5, rate = 0.3 as representative** (A1/A4).
5. **The paired-MCAR control as complete** — it controls confounds *within* G1; it does nothing to make
   G1 comparable to `transfer_features`.
6. **Two idioms → two idiom *families*** ("structural vs diffuse") — a generalization from n = 2 idioms.
7. **Fitted-Gaussian oracle X-models** (recorded necessary-not-sufficient caveat) — note this cuts both
   ways: the oracle's *low* own-value cells (next section) could partly be X-model misfit.

## 3. Q3 — Is `transfer_features` a valid isolation experiment? **No — audited against the repo:**

Facts established from `scripts/run_p2p2c_transfer_gate.py`, `feasibility-p2p2c-transfer-gate-findings.md`,
and git history (`2dfa0d9`, `c05f230`):

- ✅ **Post-bug-fix:** the 0.523 was measured *after* the dataset/δ phase-locking fix (the script
  explicitly decouples dataset choice from δ). The number itself is clean.
- ❌ **Wrong idiom for the synthesis's table:** the gate generated **LOD/top-coding examples only**
  (`generate_lod_example` is its only generator import). The synthesis placed 0.523 in the **own_value**
  column of the 2×2 — **a misattribution.** The own-value conditional-without-truth cell has **never
  been measured.**
- ❌ **Non-matched protocol:** pre-reset corpus incl. a contaminant; labor-only test; 10 bespoke
  predictor-referencing features (not the no-truth analogue of G1's Δ-statistics).
- ⚠️ **Not "no information":** in-distribution AUC was **0.668**; the failure was *transfer*
  (0.523 OOF, direction flip). "Conditional-without-truth is dead" should read "one conditional feature
  family failed to *transfer* on top-coding, on an old corpus."

**Corrected status:** `transfer_features` is a *suggestive cross-era datum*, not the load-bearing
isolation experiment the synthesis promoted it to. The claim "the binding 2→3 loss is truth, not
conditionality" currently rests on: (i) G1's protocol-matched φ-increment (truth-conditional ≫ marginal,
same everything), and (ii) the *absence* of any demonstration that no-truth conditional features work —
an absence, not a measurement. **The matched no-truth-conditional cell is the single most informative
unrun measurement in the project** — and it coincides exactly with G2's question in feature form (noted;
not designed, per constraints).

## 4. Q4 — The skeptical reviewer's strongest argument

> *"This is a post-hoc narrative. The 'hierarchy' was named after G1; its cells were never run under one
> protocol; the high levels are high by definition (an oracle distinguishes its own mechanism; truth
> reveals what was deleted) and the low levels are low relative to one arbitrary feature set. Strip the
> framing and you have rediscovered two textbook facts with extra steps: MNAR is not identifiable
> without untestable assumptions (Rubin; Molenberghs), and learned models fail off-support. The 'two
> walls' are non-identifiability and distribution shift wearing new names."*

**What must be conceded:** the *frame* is post-hoc (named after G1); the full 2×2 is not yet a single
result (§3); L2-high is partially definitional (A2); textbook theory indeed predicts the *existence* of
both walls.

**What defeats the argument as a dismissal:**

1. **The decisive contrasts are protocol-matched and pre-registered, not assembled.** G1's channel-vs-φ
   increment (+0.30–0.42) used identical examples, splits, and arbiter, with thresholds locked before
   code existed. D3's coverage-transfer law was pre-registered numerically. These are not narrative.
2. **Textbook theory does not predict the measured structure.** It does not predict that mechanism
   knowledge is *nearly redundant given truth* at the slice; that own-value carries **more** mean oracle
   information than top-coding at moderate δ (inverting the founding intuition — the PI's own prior
   theory, which the project's early results seemed to confirm); that the deployed channel's
   self-estimate of informativeness is **rank-uncorrelated with the oracle** (the E-study inversion); or
   that oracle own-value distinguishability is **bimodal across cells** (0.08–1.00 at δ=2.5) while
   truth-recovery is uniform. "Non-identifiability exists" is textbook; *its finite-n, per-idiom,
   per-cell anatomy* is not.
3. **The frame made successful predictions before it was named.** D2's coverage metric *postdicted* the
   SCF split and then *predicted* D3 (locked numbers, r = −0.91). G1's pre-registration predicted the
   sign and localization of every channel statistic, and the wealth-split fragility (P4).
4. **The honest record is the rebuttal.** Every negative is preserved with its protocol (the project's
   attributable-not-confounded discipline); the hierarchy's weak cell is identified *by this review*,
   from the project's own records, before any reviewer found it.

**Residual exposure:** until the own-value no-truth-conditional cell is measured under the G1 protocol,
a referee can correctly say the truth-vs-conditionality attribution is inferred, not isolated. That is
the publishable claim's one structural hole.

## 5. Q5 — Falsification patterns that could have occurred and did not

The hierarchy was falsifiable at every joint; observed data declined every opportunity:

| would have falsified… | the pattern | what occurred |
|---|---|---|
| "own-value is identification-limited" | G1 own-value ≈ its φ floor (0.55–0.65) | 0.90–0.99 |
| "the channel adds nothing beyond φ" (§5.C redundancy) | increment ≈ 0 | +0.30–0.42, protocol-matched |
| "estimator power is the bottleneck" | flexible ≫ linear everywhere | linear matches in-family; diverges only on heavy-tail wealth |
| "transfer is the binding constraint at L2" | wealth-held-out collapse | 0.977–0.985 (rf/gbm/nn) |
| "1→2 is lossy (mechanism knowledge essential)" | G1 ≪ oracle at δ=2.5 | ≈ equal (0.90–0.99 vs 0.93–0.99 mean) |
| the two-wall split itself | top-coding failing at L2, or own-value succeeding at L3/L4 | neither occurred (any study, any era) |
| the lattice (vs chain) | L2 ≤ L1 cell-wise everywhere | **refuted in the data:** own-value cells with `I_oracle` 0.08–0.12 at δ=2.5 where the truth channel is high |

**Falsifiers never put at risk (honest list):** interior-δ behavior of L2; the own-value
no-truth-conditional cell; any idiom beyond the two; low-rate regimes; cluster-aware uncertainty on G1's
pooled AUC.

## 6. Q6 — What can now be stated that could not be justified before G1

**Justified, protocol-matched, full weight:**

> **S1.** *For own-value self-censoring at δ = 2.5, matched rate 0.3, across four held-out survey
> domains: a generic MAR imputer **given the deleted values** recovers the mechanism at 0.90–0.99 AUC,
> while the same examples' marginal footprint supports 0.574 under identical splits and arbiter. The
> deployment-level failure on this idiom is therefore attributable to the runtime unavailability of the
> counterfactual (or of trusted mechanism assumptions that substitute for it) — **not** to absence of
> statistical signal, estimator weakness, or transfer failure.*

> **S2 (the substitutes theorem, empirical form).** *Counterfactual truth and mechanism knowledge are
> **substitutes**: either alone recovers own-value (truth: 0.90–0.99 with generic imputers; mechanism:
> mean oracle 0.93 at δ=2.5 from observed data); with **neither**, recovery collapses to the structural-
> marginal floor (0.57). Moreover the substitutes are **non-nested**: cells exist where truth succeeds
> and the mechanism-informed oracle is near-flat.*

> **S3 (the founding intuition, corrected).** *"Top-coding detectable, own-value flat" was never a
> statement about information content — at the Bayes level own-value carries **as much or more** mean
> information at moderate δ. It is a statement about **which station the information survives to**:
> top-coding's footprint survives to the assumption-free observed-data stations; own-value's does not.*

**Justified with stated qualifiers:** the two-wall decomposition (S1-style for own-value; D3's
pre-registered coverage law for top-coding) — qualified by δ-slice, rate, corpus, and n = 2 idioms.

**NOT yet justified (do not state):** "conditionality without truth is worthless" (empty own-value
cell; `transfer_features` is top-coding, cross-era); any cross-δ or cross-idiom generalization of the
near-losslessness of 1→2; any runtime implication whatsoever.

## 7. Amendments this review requires (doc edits only; PI approval to apply)

1. Replace the synthesis's **chain** with the **lattice**: two incomparable apexes — *mechanism-informed
   observed-data test* (L1) and *truth-informed generic estimator* (L2) — over the nested observed-data
   floor (L3 ⊇ L4). Cite the cell-level oracle evidence (`I_oracle` min 0.08 at δ=2.5 own-value).
2. Correct the 2×2 table: move `transfer_features` to a *top-coding, cross-era, suggestive* annotation;
   mark the own-value conditional-without-truth cell **EMPTY — the most informative unrun measurement**
   (coincides with G2's question; not designed here).
3. Re-word the headline from "the binding 2→3 loss is TRUTH" to: **"with neither truth nor mechanism
   assumptions, only structural (marginal-visible) footprints survive; truth and mechanism assumptions
   are non-nested substitutes."**
4. Attach the A1/A4/A6 scope qualifiers (δ-extreme; rate; cluster-aware uncertainty) wherever S1/S2 are
   quoted.
5. Note the oracle's own-value **cell-bimodality** (mean-level idiom-ordering claims must say "mean").

---

*No experiments, designs, or implementations are proposed. The hierarchy, as amended, carries: S1–S3 at
full weight on the protocol-matched core; the two-wall decomposition with qualifiers; and one named
structural hole (the empty cell) that any referee would find — found here first.*
