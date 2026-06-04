# P1 Feasibility — Consolidated Result (GATE PASSED, scoped)

*The single authoritative record of the P1 feasibility arc. Governed by `docs/NORTH-STAR.md`
(esp. §4.9) and `docs/PROPOSAL-survey-rewrite.md` §6. Supersedes the individual run-findings docs as
the summary of record; those remain as primary sources. Milestone frozen at git tag
`p1-feasibility-passed`.*

## 0. Question and scope

**P1 question.** On the own-value self-censoring axis (δ ≡ β₂, where scalar δ is exact), at **matched
missingness rate**, against a **β₁′-flexible (profiled) MAR null**: does an MNAR signal exist that a
re-fitting MAR cannot explain away — and can a **deployment-strength model, trained from scratch,
recover it near the optimal (Bayes) ceiling**?

**Scope.** Survey regime; single censored column; synthetic-X with controlled predictor–target
correlation ρ; the *existing* v1.0 `LacunaModel` backbone used as a binary H0-vs-H1 discriminator
(NOT the re-architected δ-prior system). This establishes a **feasibility precondition**, not the
product.

**Discipline applied throughout (charter).** Every claim of "signal exists" or "model can/can't" is
attributable, not confounded: matched rate (no rate cue), a genuine information-theoretic ceiling, the
*real* model retrained from scratch (no frozen encoder, no checkpoint, all layers trainable), null and
near-chance controls, manifests validated before any metric was read, and the rule that **no proxy
weaker than the deployment model may license a negative/kill conclusion** (§4.9). Negative gap was
always treated as a leakage red flag, never a victory.

---

## 1. The point-null oracle (`oracle_20260603_084703`)

Analytic Bayes-optimal discriminator on observed data, computed from the known generative model
(float64; MC estimate of the theoretical Bayes error, reported with SE/CI). 498 synthetic cells
(coarse + boundary refinement) + 80 real-survey-X cells, ~6 min, no model.

- Mechanics validated on the live grid: δ=0 ⟹ Bayes error 0.5; monotone in δ.
- **Finding (later shown misleading):** predictor–target correlation ρ *improved* distinguishability,
  monotonically — diagnosed as shadow-variable identification.
- **Critical caveat we raised ourselves:** this compares two FIXED point hypotheses (fixed β₁). It is a
  correct ceiling for the *fixed-MAR-vs-fixed-MNAR* question, **not** the deployment-relevant
  *MNAR-vs-the-MAR-family* question. We did **not** read a MAR-vs-MNAR conclusion from it.

## 2. The β₁′-profiled oracle correction (`profiled_20260603_100549`)

The deployment-relevant ceiling: H1 = MNAR(δ, β₁=1) vs H0 = the best-fitting MAR over { MAR(β₂=0, β₁′
free) }, β₁′ chosen to maximize the MAR family's observed-data likelihood under H1 (KL-projection ⇒
minimize distinguishability). 192 cells.

- The point-null "ρ helps" did **not** survive: a re-fitting MAR absorbs most of it (selected β₁′ rises
  with ρ: 0.83 → 1.11 → 1.46 → 1.94 at β₁=1). The corrected ceiling **restores the Molenberghs
  intuition — higher ρ makes MNAR harder.**
- **But the signal SURVIVES:** 142/144 δ>0 cells remain distinguishable under the profiled null. The
  own-value truncation is a non-identifiable-by-MAR signature (fully genuine at ρ=0; recoverable with
  more data even at ρ=0.9). One near-collapse corner: ρ=0.9, δ=0.5, n=128 → profiled 0.453 ≈ chance.
- **Minimum-viable caveat (declared):** single β₁′ per cell, β₁′-only, synthetic-X ⇒ the surviving
  signal is an UPPER bound on identifiability; a richer MAR null could absorb more.

**Four representative regimes selected** (β₁=1, rate 0.1; exact recorded params used downstream):

| regime | ρ | δ | n | profiled Bayes ceiling |
|---|---|---|---|---|
| near-chance control | 0.9 | 0.5 | 128 | 0.453 ± 0.011 |
| boundary | 0.9 | 1.0 | 512 | 0.251 ± 0.010 |
| moderate | 0.3 | 1.0 | 512 | 0.104 ± 0.007 |
| strong | 0.0 | 1.0 | 2048 | 0.004 ± 0.001 |

## 3. The model-arm pilot (`modelarm_pilot_20260603_190551`, seed 42)

Full `LacunaModel` (899,275 params) trained from scratch per regime; binary MCAR-safe head
(q = pMNAR/(pMAR+pMNAR), exactly independent of the MCAR component); fixed oracle params so model error
is directly comparable to the recorded ceiling. Plus a null-control regime (identical H0/H1).
Manifest valid.

| regime | ceiling | model_error | gap |
|---|---|---|---|
| control | 0.453 | 0.498 | +0.046 |
| boundary | 0.251 | 0.473 | **+0.222** |
| moderate | 0.104 | 0.127 | +0.023 |
| strong | 0.004 | 0.009 | +0.006 |
| null control | 0.500 | 0.500 | +0.000 |

- **Leakage controls clean:** null = 0.500 (no leakage); per-class realized rates matched ~0.10
  everywhere; no suspicious negative gaps; control ≈ chance (does not beat its chance ceiling).
- Model reaches ceiling in moderate/strong; **boundary failed (+0.222), early-stopped at epoch 15.**

## 4. The multi-seed follow-up (`modelarm_multiseed_20260603_200653`, seeds 42/43/44)

| regime | ceiling | per seed | mean ± std | gap |
|---|---|---|---|---|
| boundary | 0.251 | 0.309 / 0.480 / 0.458 | 0.416 ± 0.076 | +0.165 |
| moderate | 0.104 | 0.109 / 0.137 / 0.106 | 0.118 ± 0.014 | +0.014 |
| strong | 0.004 | 0.007 / 0.013 / 0.009 | 0.010 ± 0.002 | +0.007 |

Scoped conclusion: moderate/strong stable near-ceiling. **The boundary regime is learnable by the
current architecture, but optimization is unstable: one seed approaches the profiled ceiling (0.309),
while two seeds fall into a chance basin (0.48/0.46).** Not "a bad seed"; not a capacity or
identifiability wall.

## 5. The boundary stabilization result (`boundary_stab_20260603_230213`)

Training-procedure changes ONLY (no architecture/data/generator/oracle changes, no checkpoint): lr
1e-4, 300-step warmup, cosine→1e-5, patience 20, max 80 epochs; 10 restarts (seeds 100–109); one shared
fresh held-out test set (seed 9000); val-only checkpoint selection. Manifest valid.

- **Escape fraction 10/10**; test error **0.276 ± 0.0024**; median 0.276; best (val-selected) 0.273;
  **gap mean +0.025**; ECE mean 0.035; rates 0.101/0.102 matched; no suspicious negative gaps.
- Pre-declared success criteria (≥6/10 escape AND median ≤ 0.32, no leakage, valid manifest): **all met
  — `stabilization_success = True`.**
- Before → after: escape **1/3 → 10/10**; std **0.076 → 0.0024** (~30×); gap **+0.165 → +0.025**.

The high-ρ instability was an **optimization-procedure** problem, fully resolved by procedure changes
alone. The model now reaches the profiled Bayes ceiling at high ρ, reliably, across every restart.

---

## 6. Final scoped conclusion (P1 gate PASSED)

> **Under the controlled self-censoring P1 probe with matched missingness rates and a β₁′-profiled MAR
> null, Lacuna's existing backbone can recover the surviving MNAR signal near the profiled Bayes
> ceiling across representative regimes, including the high-ρ boundary after training stabilization.
> This establishes the feasibility precondition for building the re-architected δ-prior /
> sensitivity-prior Lacuna system.**

Evidence supporting the gate: (a) the profiled oracle proves a non-identifiable-by-MAR MNAR signal
exists in the selected regimes; (b) the full model, trained from scratch, reaches the ceiling in
moderate (+0.014), strong (+0.007), and boundary (+0.025 after stabilization); (c) null and near-chance
controls behaved correctly; (d) rate cues controlled; (e) no checkpoint/frozen-layer/provenance issue;
(f) no negative gaps or leakage; (g) the boundary failure was an optimization instability, not an
identifiability or architecture wall.

## 7. Remaining caveats (do NOT overclaim)

- **Does NOT prove real-data MAR-vs-MNAR accuracy.** Real missingness is unrecoverable; this is
  semi-synthetic ground truth on controlled mechanisms.
- **Does NOT validate the final product:** the δ-prior head, abstention, semantics prior, OOD/manifold
  checks, and the calibration system are unbuilt and untested. P1 tested the existing backbone as a
  binary discriminator, a feasibility precondition only.
- **Conditional on the minimum-viable β₁′-profiled null** (single fit, β₁′-only, synthetic-X). The
  surviving signal is an upper bound on identifiability; a richer MAR null could lower the ceiling.
- Scope is the own-value self-censoring axis; attrition / unit-nonresponse / selection idioms are
  deferred (PROPOSAL §5.1) and out of P1 scope.

## 8. Next planned robustness stages (deferred; not started)

1. **Richer MAR null** — profile over the MAR predictor choice / link family, or a per-replicate GLRT —
   to see how much of the gap survives a stronger null. The main thing that could still move the
   headline; would tighten (lower) the ceiling.
2. **Real-survey-X** — profiled oracle + model arm on fitted-X survey columns (with the fitted-X-model
   assumption stated), to check transfer off synthetic X.
3. **Multi-axis / per-idiom δ** — extend beyond own-value self-censoring to attrition & unit-nonresponse
   (PROPOSAL §5.1 revision trigger), now that the scalar-δ core is validated.
4. **Re-architecture (the product)** — build the δ-prior head + abstention + calibration objective + eval
   harness on the kept backbone (PROPOSAL §1), justified by this precondition.

Per the current decision, NONE of these run until this consolidation is reviewed.

---

## Artifacts & provenance

Run dirs under `/mnt/artifacts/project_lacuna/feasibility/`: `oracle_20260603_084703`,
`profiled_20260603_100549`, `modelarm_pilot_20260603_190551`, `modelarm_multiseed_20260603_200653`,
`boundary_stab_20260603_230213` — each with a validated `manifest.json`, results, and summary. Code:
`lacuna/feasibility/` (tested package) + `scripts/run_feasibility_*.py`. Primary-source findings docs:
`docs/feasibility-oracle-run1-findings.md`, `docs/feasibility-oracle-run2-profiled-findings.md`,
`docs/feasibility-model-arm-pilot-findings.md`, `docs/feasibility-boundary-stabilization-findings.md`.
