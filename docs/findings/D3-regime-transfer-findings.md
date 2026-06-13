# D3 — Regime-Matched Continuous Transfer Test (findings)

*Result of the pre-registered D3 test (`PROPOSAL-D3-regime-matched-transfer-spec.md`; PI-approved
2026-06-07). Architecture / HP / metadata / Level-2 / acquisition all **frozen**; no extra arms; same cfg
as `run_curve_tightened`. Code: `scripts/run_d3_regime_transfer.py`; pre-registration
`scripts/preregister_d3_coverage.py` → `runs/d3_preregistration.json`; result `runs/d3_regime_transfer.json`.
8 seeds. AUC + RPS + ECE + entropy reported as specified.*

---

## 0. Decision criterion & verdict

> **Does held-out transfer improve when the training pool is closer in footprint-regime space?**

**YES — H1 supported.** Across the 8 pre-registered (held-out D, pool) pairs, transfer AUC tracks the
**locked footprint-regime coverage** strongly and in the predicted direction: **Pearson(coverage, AUC) =
−0.91, Spearman = −0.74, slope = −0.067 AUC per coverage-unit.** Lower coverage-distance ⇒ higher transfer.
**Two binding qualifications** (below): the effect is **pool-level**, not single-dataset (the strict
size-control does not isolate it), and the model is **confidently wrong off-manifold** (the North-Star
payoff).

## 1. Results (8 seeds; continuous-only bases; coverage locked before training)

| held-out D | pool | pre-reg coverage | AUC | RPS | ECE | entropy (bits) |
|---|---|---|---|---|---|---|
| **NHANES** | **M matched (labor)** | **0.348** | **0.640 ± 0.010** | 0.258 | 0.161 | 0.831 |
| NHANES | Xₒ mismatch-ordinal (bfi+yrbss) | 0.368 | 0.640 ± 0.021 | 0.271 | 0.175 | 0.748 |
| NHANES | M+Xₕ diverse (labor+wealth) | 0.328 | 0.627 ± 0.025 | 0.247 | 0.115 | 0.882 |
| **NHANES** | **Xₕ mismatch-heavytail (wealth)** | **0.729** | **0.577 ± 0.032** | 0.317 | 0.256 | 0.634 |
| **HMDA** | **M matched (labor+nhanes)** | **0.704** | **0.648 ± 0.032** | 0.331 | 0.300 | 0.594 |
| HMDA | Xₒ mismatch-ordinal | 1.173 | 0.528 ± 0.041 | 0.370 | 0.326 | 0.563 |
| **HMDA** | **Xₕ mismatch-heavytail (wealth)** | **1.422** | **0.536 ± 0.048** | 0.381 | 0.332 | 0.542 |
| **wealth** (control) | all-moderate (labor+nhanes+hmda) | 3.092 | **0.461 ± 0.024** | 0.387 | 0.335 | 0.513 |

## 2. The directional contrasts (spec §4)

- **Matched > Mismatch-heavytail (both continuous ⇒ isolates regime from continuity):**
  - NHANES: 0.640 vs 0.577, **Δ = +0.063** (≈1.9·SE).
  - HMDA: 0.648 vs 0.536, **Δ = +0.112** (≈1.9·SE).
  Both positive, both just shy of the strict per-domain 2·SE bar — **decisive only in aggregate** (§3).
- **Matched ≥ Mismatch-ordinal — and this is the cleanest *confirmation of the coverage theory*:** for
  NHANES the ordinal pool was **coverage-matched** (0.368 ≈ 0.348) and transferred **identically** (0.640
  = 0.640); for HMDA the ordinal pool was **coverage-far** (1.173) and transferred **poorly** (0.528,
  Δ = +0.120, >2·SE). So transfer follows **coverage**, not the surface "continuous-vs-ordinal" label — a
  within-experiment confirmation that the footprint geometry, not the domain type, is what matters.
- **Diverse (M+wealth) ≤ Matched:** NHANES 0.627 ≤ 0.640 — adding a regime-far domain did not help (and
  shows the SCF seed-collapse echo: one diverse seed fell to 0.50).
- **Wealth control = the anchor:** held out, wealth is the most isolated (coverage 3.09) and transfers
  **below chance (0.461)** from every moderate domain — reproducing the SCF result and fixing the
  high-coverage end of the line.

## 3. The strong test — coverage predicts transfer (the spec's designated real test)

Regress transfer AUC on pre-registered coverage across all 8 pairs: **Pearson −0.91, Spearman −0.74,
slope −0.067.** This is the decisive result: the **pre-registered** footprint-regime coverage — computed
from data shape *before any training* — predicts held-out transfer. The per-domain contrasts are
individually noisy (~1.9·SE), but the corpus-level relationship is strong and correctly signed, anchored
by the wealth control.

## 4. Binding qualification — the effect is POOL-LEVEL, not single-dataset

Size-controlled isolation (held-out NHANES; **K = 6 continuous columns each, one dataset each, same cfg**):

| pool | AUC | ECE | entropy |
|---|---|---|---|
| M_cap = psid1976 (6 cols, regime-matched) | **0.562 ± 0.009** | 0.100 | 0.967 |
| Xₕ_cap = wealth (6 cols, regime-mismatched) | **0.577 ± 0.032** | 0.256 | 0.634 |

**Under strict single-dataset size control, the matched advantage disappears** (0.562 ≈ 0.577). The full
matched pool's edge (0.640) over a single capped matched survey (0.562) is the effect of having **five
regime-matched labor surveys**, not one. **Honest reading:** footprint-regime coverage predicts transfer
**at the pool/corpus level** — through having a *cluster* of regime-matched data that covers the held-out
geometry — **not** through a single regime-near dataset. This is exactly the D2/acquisition framing ("need
a cluster of regime-similar continuous domains so the pool can interpolate the geometry"), now confirmed:
one regime-matched dataset is **not** sufficient; coverage is a property of the pool.

## 5. North-Star payoff — the model is CONFIDENTLY WRONG off-manifold

The North-Star question is **whether Lacuna knows when its empirical prior covers the footprint regime.**
Reading calibration + entropy against coverage (within each held-out domain, to avoid the cross-domain
base-rate confound):

- **NHANES:** matched → ECE 0.161, entropy **0.831**; mismatch-wealth → ECE 0.256, entropy **0.634**.
- **HMDA:** matched → ECE 0.300, entropy 0.594; mismatch-wealth → ECE 0.332, entropy 0.542.
- **wealth control (most isolated):** AUC 0.461 (below chance), ECE 0.335, entropy **0.513** (most confident).

**As coverage-distance rises, the model gets MORE confident (lower entropy) and WORSE calibrated (higher
ECE) while being LESS accurate** — the **opposite** of graceful degradation. **The current Level-1 model
does not know when it is off-footprint-manifold; it is confidently wrong exactly where it should abstain.**

This is the actionable North-Star result: **regime coverage is real and measurable, but the model does not
self-regulate its confidence by it.** It is the empirical case for making **coverage_distance an explicit
input to the L1 detectability / OOD-abstention head** (MASTER §5, Stage-B Level-1) — the data-driven signal
of footprint support the model cannot infer on its own. *(Designing/adding that head remains out of scope
and separately gated; D3 only establishes that the signal exists, predicts transfer, and is needed.)*

## 6. What D3 settles (and what it does not)

**Settles:**
- **H1 supported:** held-out transfer improves when the training pool is closer in footprint-regime space
  (pre-registered coverage → transfer, Pearson −0.91). Transfer is **structured by footprint-regime
  coverage**, not domain count and not the continuous-vs-ordinal label (the coverage-matched ordinal arm
  transferred like labor).
- **The SCF reinterpretation is confirmed:** continuous targets are **not** hopeless — NHANES and HMDA
  continuous targets transfer well **when covered** (0.64); wealth failed because it is **regime-isolated**,
  both as a held-out target (0.461) and as a teacher (worst transfer).
- **Detectability gap identified:** the model is overconfident off-manifold ⇒ coverage is needed as an
  explicit detectability input.

**Does not settle:**
- Regime-coverage is a **pool-level** effect; the strict single-dataset size-control did **not** isolate a
  regime effect (matched-single ≈ mismatched-single). The mechanism needs a *cluster*, and "regime alone at
  equal size" is **not** demonstrated. Per-domain contrasts are ~1.9·SE (decisive only in aggregate).
- Whether coverage_distance, added to the detectability head, actually yields calibrated abstention — that
  is the **next** question, not tested here.

## 7. Decision bearing (for review — no further runs taken)

- **H1 holds at the corpus level** ⇒ the footprint-regime coverage theory is the right frame; the
  acquisition screen (score candidates by coverage to the pool) is validated **as a pool-level predictor**.
- **The pool-level qualification sharpens acquisition:** acquire **clusters** that cover a target geometry,
  not single regime-near domains (a lone domain ≈ no isolatable gain at equal size).
- **The strongest forward step is the detectability head** using coverage_distance, because the model's
  confidence does **not** track coverage on its own — arguably more North-Star-central than more
  acquisition. *(Proposal/spec only when chosen; not started.)*

246 survey tests green (no `lacuna/` source changed in D3 — scripts + docs only). **Stopped for review per
the D3 plan; no architecture, tuning, metadata, Level-2, or acquisition touched.**
