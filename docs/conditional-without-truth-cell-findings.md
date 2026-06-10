# The Empty Cell, Measured — Conditional-Without-Truth (findings)

*Result of the pre-registered empty-cell experiment
(`PREREGISTRATION-conditional-without-truth-cell.md`, commit `e1374f9`; implementation committed before
results, `3adb8da`). Executed exactly as locked: same corpus/cells/24 examples with **bit-identical
masks to G1** (all 3,840 rows verified by recomputing the frozen base17 features against G1's stored
rows before any new feature was trusted); the paired-MCAR view remained **forbidden** (leak L1 — its
RNG draw consumed-but-unused to preserve identity); feature family H/S/R + anchors per spec; no tuning;
bands and stability rule untouched. Artifact: `runs/conditional_without_truth_cell.json`. 256 survey
tests green. **Stopped for review per instruction: no lattice amendments, synthesis changes, runtime
implications, G2 planning, or architectural conclusions are made here.***

---

## 0. Verdict under the locked rules

> **MIXED — unanimous (4/4 distinct imputer classes).** Own-value conditional-without-truth lands at
> **AUC 0.743–0.796, ρ = 0.41–0.59** (mean ρ **0.506**) — squarely inside the locked mixed band
> (0.25 < ρ < 0.60), for every distinct class. Neither the truth-bottleneck reading nor the
> conditional-structure reading of G1 survives intact: **the G1 gain decomposes into two measured,
> comparably-sized parts.**

## 1. Results (pooled OOF AUC, 4-domain LODO, identical splits/arbiter to G1)

**Primary — own_value δ0-vs-δ2.5** (anchors: φ = 0.574; per-imputer G1 truth channel):

| imputer | AUC_cell | ρ | locked band | base17+cell | G1 anchor |
|---|---|---|---|---|---|
| linear | 0.766 | 0.589 | **mixed** | 0.801 | 0.900 |
| mice_lite | 0.768 | 0.594 | mixed (≡linear) | 0.803 | 0.900 |
| rf | 0.743 | 0.407 | **mixed** | 0.738 | 0.989 |
| gbm | 0.783 | 0.498 | **mixed** | 0.733 | 0.993 |
| nn | 0.796 | 0.529 | **mixed** | 0.741 | 0.994 |

**Secondary — top_coding** (anchors: φ = 0.633; G1 tc 0.883–0.947): AUC_cell **0.813–0.830**,
ρ = 0.59–0.79 ⇒ **conditional_structure band, 5/5** (all clear the 0.80 absolute criterion).

**Idiom separation at δ=2.5 (reported, not gated):** 0.54–0.68 — **collapsed** relative to G1's
0.68–0.83. Without truth, the features detect *that* selection is occurring far better than *which
mechanism* is doing it (the truth-referenced PIT shape was the mechanism reader).

## 2. The decomposition of the G1 gain (the experiment's purpose)

For own-value, averaging the four distinct classes (G1 increment ≈ +0.36 over φ):

| component | share of the G1 increment | AUC terms |
|---|---|---|
| **conditional-representation share** (no truth needed — what φ's marginal-only design discards) | **≈ 51 %** (mean ρ 0.506) | 0.574 → ≈ 0.77 |
| **truth/identification share** (recoverable only with the deleted values) | **≈ 49 %** | ≈ 0.77 → 0.93–0.99 |

Both competing interpretations of G1 were half right: the φ representation **does** discard substantial
observed-view conditional signal for own-value, **and** a comparably large remainder is carried by
truth alone.

## 3. The secondary result supersedes `transfer_features`

On the **matched protocol** (same corpus, splits, idiom, arbiter), conditional-without-truth features
reach **0.81–0.83** for top-coding — versus the cross-era `transfer_features` 0.523 that the adversarial
review demoted. The 0.523 is now attributable to its *feature family and corpus*, not to the station:
the no-truth-conditional station carries strong transferable top-coding signal. The matched 2×2 is now
**complete in all four cells** for top-coding, and complete for own-value:

| own_value | conditional | marginal |
|---|---|---|
| **with truth** | 0.90–0.99 (G1) | — |
| **without truth** | **0.74–0.80 (this run)** | 0.574 (φ) |

## 4. Branch logic (locked §6 — applied, not extended)

The **mixed** branch's pre-registered consequences:

- The lattice's floor **splits**: marginal floor (0.57) < **conditional mid-rung (≈ 0.77)** < apexes
  (0.93–1.0). Both walls remain, each with a **measured partial height**.
- The **lab-coat fraction becomes a two-term decomposition**: a *representation share* (≈ half of the
  deployment gap — attributable to marginal-only features, in principle closable from the observed
  view) and an *identification share* (≈ half — closable only by truth or assumptions).
- **S2 (the substitutes theorem) must be weakened as the spec anticipated**: "with neither truth nor
  mechanism assumptions ⇒ the marginal floor" is falsified as stated; the correct form is "⇒ the
  conditional mid-rung," which sits well below both apexes but well above φ.
- Per the pre-registration, this is "arguably the richest outcome for the dissertation" — both
  interpretations of G1 are quantified rather than one eliminated.

**Everything beyond this paragraph is gated** (PI instruction): the observation that the H/S/R features
are computed from the observed view alone has obvious downstream relevance, and it is **not pursued
here** — no runtime claims, no G2 planning, no architectural conclusions, no lattice/synthesis edits.

## 5. Honest caveats

1. **Boundary proximity:** nn's AUC (0.796) sits 0.004 below the 0.80 absolute criterion and linear's ρ
   (0.589) 0.011 below 0.60 — the cell occupies the *upper half* of the mixed band, flirting with the
   conditional-structure boundary for 2 of 4 classes. The locked verdict is mixed (unanimous); the
   proximity is reported, not acted on.
2. **base17+cell sometimes *underperforms* cell-alone OOF** (rf 0.738 < 0.743; gbm 0.733 < 0.783) —
   feature dilution in leave-domain-out LR; the 17 marginal features are not uniformly helpful company.
   Increment-over-φ claims here use the cell-alone numbers.
3. **No feature-group ablation** (H vs S vs R) was pre-registered, so none is reported — which group
   carries the own-value signal is unknown and would require a new pre-registration.
4. Scope as inherited from G1: δ ∈ {0, 2.5} binary slice, matched rate 0.3, n ≈ 384, this corpus;
   pooled OOF without cluster-aware uncertainty (per-protocol; the unanimity across 4 classes and the
   band widths are the robustness evidence).
5. mice_lite ≡ linear (complete predictors), as in G1 — 4 distinct classes, not 5.

---

*Executed exactly per the locked pre-registration; bit-identity to G1 verified for every example; the
forbidden cross-view pairing never computed; bands, stability rule, and feature family untouched after
results. The own-value G1 gain is a measured combination: ≈ half representation, ≈ half identification.
Stopped for PI review.*
