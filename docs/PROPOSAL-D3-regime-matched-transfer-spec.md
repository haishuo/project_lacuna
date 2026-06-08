# D3 — Regime-Matched Continuous Transfer Test (specification)

*Specification only — **no runs, no training, no acquisition, no architecture change** (binding;
PI 2026-06-07). A pre-registered hypothesis test of the footprint-regime **coverage** theory. Governed by
`NORTH-STAR.md` (the inference object = the observed-data footprint; §2 identification; §3½ manifold) and
`MASTER-lacuna-survey-architecture.md`. Builds on `CONSOLIDATION-MEMO-SCF-wealth-conditional-scaling.md`
(H1) and `REGIME-MAP-findings.md` (D2 metric). Pre-registration computed by
`scripts/preregister_d3_coverage.py` (descriptive; no training) → `runs/d3_preregistration.json`.*

---

## 0. Why this is the North-Star question

Lacuna's core object is the **observed-data footprint**. If the model is genuinely data-driven, its
ability to estimate a column's δ-prior should track **how well the training pool covers the footprint
geometry of the held-out column** — not the raw count of training domains (D2 showed count is too crude).
So the question is **not** "does more data help?" but:

> **Does transfer improve when the training pool contains columns close to the held-out target in
> footprint-regime space?**

A "yes" makes regime-coverage a measurable, North-Star-aligned driver of detectability; a "no" falsifies
the coverage theory.

## 1. Hypothesis

- **H1 (coverage):** regime-**matched** continuous training domains transfer to a held-out continuous
  domain **better than** regime-**mismatched** domains — *even when both training pools are continuous*, so
  the difference is **footprint geometry**, not "continuous vs ordinal" and not semantics.
- **Corollary (the SCF reinterpretation):** SCF wealth failed **not** because continuous targets are
  hopeless, but because wealth is **isolated** in φ-footprint space (D2: coverage 3.09; isolation ~10×).
  A continuous domain that **is** covered should transfer.
- **Null / falsifier:** if a regime-**mismatched** continuous pool (heavy-tailed wealth) transfers to a
  moderate continuous target as well as a regime-**matched** pool (labor), the coverage theory is wrong or
  incomplete — regime geometry is not what drives transfer.

## 2. The test design (architecture frozen)

**Held-out continuous domains** (each its own block; cross-semantic to their matched pools, so a win is
*geometry*, not shared subject matter):
- **D₁ = NHANES-continuous** (self-reported weights, poverty, income — health/demographic semantics).
- **D₂ = HMDA / finance** (mortgage loan/income — finance semantics).
- **D₀ = SCF wealth** — the **negative control** (already observed flat; D2 coverage 3.09).

**Training pools per held-out D** (the contrast that isolates regime):

| pool | definition | role |
|---|---|---|
| **Matched (M)** | moderate-continuous domains regime-near D (labor; +nhanes for D₂) | regime-matched, **continuous** |
| **Mismatch-heavytail (Xₕ)** | wealth (SCF) | regime-**far**, **also continuous** ⇒ isolates regime from continuity |
| **Mismatch-ordinal (Xₒ)** | bfi + yrbss | regime-far, low-cardinality (reported; clean only for D₂ — see §3) |
| **Diverse-but-mismatched (M+Xₕ)** | matched **+** wealth | does adding a regime-far domain to a matched pool help? (predict: no / destabilizes — the SCF seed-collapse echo) |

**The load-bearing comparison is M vs Xₕ** — both continuous, differing only in footprint regime.

**Controls (mandatory, to attribute a difference to regime and not size/teacher-quality):**
- **Size control:** equalize the training budget across M and Xₕ — same number of source **continuous
  columns** and same `train_size` / `max_rows`. Report size as a covariate; if M still wins under matched
  size, attribute to regime.
- **Teacher-quality control:** the **multi-pair correlation** (§4 strong test) varies the *target* under a
  fixed *teacher*, separating "wealth is a poor teacher universally" from "wealth mismatches D specifically."
- **Block-aware / leakage:** held-out D fully excluded from training (NHANES = one block; SCF = one block;
  labor waves kept together is unnecessary here since labor is only ever a training pool).
- Everything else identical to `run_curve_tightened.py`: same RPS head, temperature, leakage gate, δ-grid,
  top-coding idiom, fixed VAL (chile + hmda when not the held-out), fixed seeds. **No tuning, no new
  variant, no metadata, no Level-2 deviation.**

## 3. The metric & the pre-registration (computed BEFORE training)

Regime-distance = the **D2** scale-invariant footprint coordinate (L-skew τ₃, L-kurt τ₄, log₁₀ card,
has_neg), standardized across corpus columns; **coverage_distance(D | pool)** = mean over D's continuous
columns of the nearest pool-column distance (lower ⇒ better covered ⇒ predict more transfer).

**Pre-registered coverage (`runs/d3_preregistration.json`; lower = predict more transfer):**

| held-out D | Matched | Mismatch-heavytail (wealth) | Mismatch-ordinal (bfi+yrbss) | Diverse (M+mismatch) |
|---|---|---|---|---|
| **NHANES** (12 cont cols) | **0.348** (labor) | **0.729** | 0.368 | 0.236 |
| **HMDA/finance** (3 cont cols) | **0.704** (labor+nhanes) | **1.422** | 1.174 | — |
| **SCF wealth** (control) | 3.092 | — | — | — |

**Pre-registered directional predictions (locked before any training):**
1. **NHANES:** transfer(M=labor) **>** transfer(Xₕ=wealth)  [0.348 ≪ 0.729]. *(Ordinal 0.368 ≈ matched ⇒
   ordinal is **not** a clean discriminator for NHANES; the clean test is labor vs wealth. The
   diverse/NHANES coverage 0.236 even predicts a small diversity gain — consistent with the observed noisy
   +0.045 — so for NHANES, "diverse-but-mismatched" is defined as **M + wealth only**, not M + bfi/yrbss.)*
2. **HMDA:** transfer(M=labor+nhanes) **>** transfer(Xₕ=wealth) **and** **>** transfer(Xₒ=bfi+yrbss)
   [0.704 ≪ 1.422, 1.174]. HMDA is the **cleanest** held-out domain (matched clearly < both mismatches).
3. **SCF wealth (control):** transfer ≈ flat from every pool [3.09] — already observed (§8 findings); the
   anchor that fixes the high-coverage-distance end of the line.
4. **M + wealth ≤ M:** adding a regime-far domain to a matched pool does **not** improve (predict flat or
   destabilized) transfer to D.

## 4. Success criteria

Primary (directional, per held-out domain, size-controlled):
- **H1 SUPPORTED** if, for both D₁ and D₂: **Matched > Mismatch-heavytail** (the regime-isolating contrast)
  **and** Matched ≥ Mismatch-ordinal **and** Diverse(M+wealth) ≤ Matched. (Effect must exceed seed noise:
  Δ > 2·SE on ≥8 seeds.)
- **H1 WEAKENED** if Matched ≈ Mismatch (within 2·SE) — regime geometry does not separate transfer among
  continuous pools.
- **Coverage theory WRONG/INCOMPLETE** if Mismatch-heavytail (wealth) transfers to a moderate continuous
  target **as well as or better than** Matched — geometry-coverage is not the driver.

Strong (the real test — regime as a *continuous predictor*, not a binary label):
- Across **all (held-out D, pool) pairs** {D₁, D₂, D₀} × {M, Xₕ, Xₒ, M+Xₕ}, regress observed transfer AUC
  on pre-registered **coverage_distance**. **H1 supported** if the slope is **negative and significant**
  (more coverage ⇒ more transfer), anchored by the wealth control at the far end. This subsumes the
  per-domain contrasts and controls teacher-quality by varying the target.

**This is not about maximizing AUC** (§5). A *small* matched-vs-mismatched gap that is **directionally
pre-registered and replicated** is the result; a large AUC is not.

## 5. North-Star outputs — coverage → detectability / abstention

The point is **not** discrimination; it is learning **when Lacuna has footprint support**:
- **Record calibration / confidence**, not just AUC, under matched vs mismatched training: when
  coverage_distance is high (wealth), the model should return **low detectability / prior-dominated,
  high-uncertainty** output ("confidently uncertain" or the **UNKNOWN** off-manifold label) — **not** a
  confident wrong δ. When coverage_distance is low (NHANES under labor), confidence may be earned.
- **Deliverable:** a table of (coverage_distance, transfer AUC, **calibration/RPS, output entropy /
  detectability proxy**) per (D, pool). The North-Star-relevant success is **graceful degradation**:
  uncertainty rises with coverage_distance.
- **Forward role:** if H1 holds, **coverage_distance becomes a candidate input to the L1 detectability /
  OOD-abstention head** (MASTER §5, Stage-B Level-1) — the data-driven signal that a column is
  off-footprint-manifold. *(Designing/adding that head is out of scope here — D3 only tests whether the
  signal exists and behaves; the head is future work, gated separately.)*

## 6. Feasibility — the existing corpus instantiates D3 (no acquisition)

**Verdict: NO acquisition required.** The corpus already contains a tight cluster of moderate-regime
continuous domains across **different blocks** (D2 / pre-registration): labor (L-skew 0.17), nhanes (0.13),
hmda (−0.02), chile (0.10), yrbss-continuous (0.12) — with wealth (0.90) the lone outlier. This supplies:
- **matched continuous pools** (labor; labor+nhanes) — regime-near, cross-semantic to the held-out domains;
- a **regime-far continuous** comparator (wealth) for the isolating contrast;
- **two cross-semantic held-out continuous domains** (NHANES, HMDA) each in its own block;
- the **negative control** (wealth) already measured.

**Minimal construction (re-curation, not acquisition):** build a **NHANES-continuous-only** role-B base
(drop categorical predictors, keep weights/poverty/income — analogous to the SCF continuous-only base) so
the held-out NHANES test is not diluted by categorical-target draws (the confound found in §8). This reuses
existing on-disk NHANES data and `build_nhanes_role_b.py`; it is **not** an acquisition.

**If (and only if) a robustness check wants a *third*, independent cross-semantic regime-matched continuous
domain** beyond NHANES and HMDA, that would be the *minimal acquisition* — a single moderate-regime
continuous survey (e.g., a consumption/expenditure or earnings module whose pre-registered coverage to the
labor/nhanes cluster is < 0.5). **Not required for the core test**; flagged only as an optional strengthener,
and itself gated by a pre-registered coverage prediction (the D2 acquisition screen).

## 7. Concrete runner (to build only on PI go — NOT now)

`scripts/run_d3_regime_transfer.py`, mirroring `run_curve_tightened.py`:
- Held-out D ∈ {NHANES-cont, HMDA}; pools {M, Xₕ, Xₒ, M+Xₕ}; ≥8 seeds; size-controlled (equal #continuous
  training columns + `train_size`/`max_rows`); fixed VAL; block-aware; architecture/HP frozen.
- Outputs per (D, pool, seed): transfer AUC + calibration/RPS + output entropy; aggregated mean ± SE.
- Computes the **coverage-vs-transfer regression** against the pre-registered table (§3).
- Writes `runs/d3_regime_transfer.json` + a findings doc. **No tuning; no metric-chasing.**

## 8. Guardrails (binding)

No architecture change · no hyperparameter tuning · no new model variant · no metadata channel · no
Level-2 reference/deviation · no training on natural missingness · pre-registration is **locked before**
training (committed in `runs/d3_preregistration.json` + this spec) · count worlds/blocks not files ·
detectability/transfer claims name their comparison class (the **current pool**) · success is
**directional + pre-registered**, never "higher AUC."

---

*No runs, training, downloads, or architecture changes are performed by this document. It specifies a
pre-registered, size-controlled, architecture-frozen test of whether footprint-regime coverage drives
transfer, instantiable from the existing corpus with no acquisition. Execution awaits PI go.*
