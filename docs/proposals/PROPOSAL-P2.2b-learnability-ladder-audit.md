# P2.2b — δ-Prior Learnability Ladder — Implementation Audit / Spec

*Branch `p2/delta-prior-rearchitecture` (continues from P2.2; implementation `8d4210e`, pilot
findings `5bfc0df`). Governed by `docs/proposals/PROPOSAL-P2-delta-prior-rearchitecture.md`,
`docs/proposals/PROPOSAL-P2.2-model-loss-audit.md`, `docs/NORTH-STAR.md`, project `CLAUDE.md`.
**SPEC ONLY — no code, no training until approved.***

## 0. Why P2.2b exists

P2.2 is a **valid negative pilot**: the wide-table, real-X, 7-bin, single-target δ-prior sits at
the uniform-RPS floor across three configs (incl. in-distribution `max_rows=512`). The machinery
is correct and the leakage gate is clean, so the floor is not a plumbing or rate-cue artifact —
the model simply is not extracting the own-value self-censoring signal at the formulated task
difficulty. **P2.2b reduces the task to the simplest setting that should be learnable, confirms
learning there, then re-adds complexity one axis at a time** until learning breaks — localizing
the bottleneck (head/loss vs ordinal granularity vs real-X geometry vs localization vs scale).

This is the charter discipline: a result must be **attributable, not confounded**. The current
failure confounds four difficulty axes at once; the ladder de-confounds them.

## 1. What the P2.2 result does and does not tell us

- **Does tell us:** with all four axes stacked, RPS ≈ uniform and bin-acc ≈ chance; calibration is
  (trivially) good because predictions collapse to ≈ the δ-bin marginal; the matched-rate gate is
  clean (|corr(δ,rate)| ≤ 0.05).
- **Does NOT tell us which axis is responsible.** The four stacked axes are: (a) **ordinal
  resolution** (7 bins), (b) **real-X covariance geometry** (vs controlled-ρ Gaussian),
  (c) **table width / target localization** (d up to 28 vs d=2), (d) **per-example evidence /
  scale** (rows × missing cells).
- **Important reframing of "localization":** in the *current* generator only the target column is
  censored — so the censored column is the **unique column carrying missingness**, which is
  directly observable in the `is_observed` token channel. The model is therefore not asked to
  *search blindly* for the target; it must extract δ from a column it can already identify by its
  missingness. This suggests the dominant bottleneck is more likely **signal extraction / ordinal
  granularity** than localization per se. The ladder tests this explicitly (rungs 1–4), and the
  conditioning proposal (§4) addresses the human-parity question regardless.

## 2. Shared harness (identical across every rung — reuse, do not duplicate)

Every rung reuses the validated P2.2 components unchanged: `DeltaBinHead`/`DeltaPriorModel`
(`survey.delta_head`), `rps_loss`/`log_score`/`fit_temperature` (`survey.loss`), all metrics
(`survey.metrics`), the leakage gate (`survey.leakage`), the run manifest (`survey.run_manifest`),
and the answer-sheet/δ-bin scheme (`survey.answer_sheet`, `survey.delta_bins`). **Only the DATA
source and (rungs 3–4) the optional target-conditioning change between rungs.** This is the whole
point — holding head+loss+eval fixed is what makes a rung-to-rung difference attributable.

**Per-rung evaluation protocol (mandatory, every rung):**
- **RPS vs references** — test RPS compared to `uniform_rps(K)` AND a "predict the train δ-bin
  marginal" base-rate RPS; a rung *passes* only if it beats both by a margin outside the test CI.
- **Bin accuracy + adjacent accuracy** (order-aware), vs chance (1/K) and base-rate adjacent.
- **Calibration** — ECE + interval coverage (50/80/90), before/after temperature.
- **Leakage gate** — same blocking diagnostic; `leakage_pass` must be True for any rung whose
  number is interpreted (δ→rate corr, per-bin rate table, rate-only baseline).
- **Manifest** — `survey.run_manifest` validates; `kind ∈ {main, ablation, smoke}`; the rung id and
  the axis-removed are recorded. No metric is read unless the manifest validates.
- **From scratch** — fresh init, `all_layers_trainable=True`, `checkpoint_loaded=False` (asserted).

**Forbidden (all rungs):** old 3-class CE objective; binary MAR/MNAR as a *main* result. A binary
δ=0-vs-δ>0 probe is allowed ONLY as an explicitly labeled `kind="ablation"` diagnostic to confirm
gradient signal — never headlined and never the ladder's pass criterion.

## 3. The ladder

Fixed for rungs 1–4 (so only the named axis moves): a single recorded `target_rate`, β₁∈[0,2]
swept nuisance, a fixed adequate rows-per-example `n` (NOT starved — e.g. n≈2000, matching the P1
regime; scale is rung 5's variable), the 7-bin scheme (except rung 2), matched-β₀ rate solve.

| Rung | Data X | Bins | Target known? | Axis REMOVED vs P2.2 | Reuses |
|---|---|---|---|---|---|
| 1 | synthetic 2-col `ConditionalGaussian(ρ)` | 7 | yes (col 1) | real-geometry + width + localization | `feasibility.xmodel.ConditionalGaussian`, `feasibility.delta_generator.apply_self_censor` |
| 2 | synthetic 2-col | **3 (coarse)** | yes | + ordinal resolution | rung-1 data + coarse bins (§5) |
| 3 | **real survey X** | 7 | **yes (supplied + conditioned)** | width/localization only (real geometry back) | `survey.delta_generator` + target conditioning (§4) |
| 4 | real survey X | 7 | no (current P2.2) | nothing (full task) | current P2.2 path |
| 5 | best learnable rung | — | — | scale UP rows/examples | the rung that learned |

### Rung 1 — oracle-aligned 2-column δ-bin task
- **Data:** draw a 2-col dataset from `ConditionalGaussian.synthetic(ρ)` (the exact P1 X-model),
  apply own-value self-censoring on col 1 with **δ sampled from the grid** (not the P1 fixed
  H0/H1) and the matched-β₀ rate solve — i.e. `apply_self_censor(X, target_idx=1, predictor_idx=0,
  …)`, the same call P2.1 already wraps. Sweep ρ across the P1 regimes (recorded). Known
  (target, predictor) = (1, 0); the answer sheet records δ/bin exactly as today.
- **Model:** `DeltaPriorModel` with `max_cols=2`; nothing else changes.
- **Purpose / question:** *Can the δ-bin head + RPS learn ordered δ AT ALL when geometry, width,
  and localization are all removed?* This is the cleanest test of the head/loss formulation.
- **Pass:** test RPS beats uniform & base-rate (outside CI); bin-acc ≫ 1/7; adjacent-acc high;
  coverage near nominal. Compare, informally, to the P1 profiled-oracle separability per ρ (a
  strong δ regime should be easy; near-MAR hard) — a sanity tie to the P1 ceiling, not a gate.

### Rung 2 — 2-column coarse (3-bin) δ task
- **Data:** rung-1 data, relabeled into **3 ordered bins** (§5): `{δ=0}`, `weak (0,τ]`,
  `strong (τ,∞)`.
- **Purpose:** *distinguish an ordinal-RESOLUTION failure from a total-SIGNAL failure.* If 7-bin
  (rung 1) fails but 3-bin succeeds, the head can detect δ but not resolve 7 levels at this n.
- **Pass:** same protocol with K=3 (uniform/base-rate recomputed for K=3).

### Rung 3 — target-known real survey X
- **Data:** the P2.1 real-survey generator (`survey.delta_generator`) unchanged, but the chosen
  (target, predictor) indices are **supplied to the model** via explicit conditioning (§4), not
  inferred. 7 bins.
- **Purpose:** *re-introduce real covariance geometry while keeping localization removed.* Isolates
  "is real-X geometry the problem?" from "is finding the column the problem?"
- **Pass:** beats uniform/base-rate on a held-out (leave-datasets-out) test pool.

### Rung 4 — target-localization added back (= current P2.2)
- **Data/model:** exactly today's P2.2 (no conditioning; model identifies the censored column from
  the missingness pattern). Run **only after rungs 1–3 show learning.**
- **Purpose:** *does removing the supplied target (rung 3 → 4) collapse performance?* If rung 3
  learns and rung 4 does not, localization/aggregation IS the bottleneck → P2 needs explicit
  target conditioning or per-column processing as the design (§4, §6).

### Rung 5 — scale test (LAST)
- Increase rows/example and/or examples/epoch **only after a simplified rung learns**, to map the
  evidence–accuracy curve. **Explicitly gated:** do not spend compute scaling a rung that has not
  already learned at modest scale — that would mask a design failure with FLOPs (charter §4.6; the
  user's explicit instruction). Record the scaling and the marginal gain.

## 4. Explicit target-column conditioning (human-parity design)

**Motivation (NORTH-STAR human-parity, §4.8):** a human analyst knows *which* column has the
sensitivity/missingness concern. Forcing Lacuna to infer the target is an unnecessary,
confounding burden. The workflow can supply the candidate target (and a predictor); Lacuna should
report δ *for that column*.

**Two implementation options (decide at build time; recommend B):**
- **(A) Token marker channel** — add a 5th token channel `is_candidate_target` (and/or
  `is_candidate_predictor`). Clean signal, but **touches shared `lacuna/data/tokenization.py`
  (TOKEN_DIM 4→5)** and every consumer — a cross-cutting change to validated infra. Higher
  blast radius (CLAUDE.md Rule 3/8 caution).
- **(B) Head-side conditioning (recommended)** — leave tokenization untouched; the encoder already
  exposes `token_representations [B, max_rows, max_cols, hidden]` (via `return_intermediates` /
  `get_token_representations`). Pool the **target column's** token reps over rows → a target
  embedding; optionally the predictor's too; concatenate with `evidence` and feed a
  `TargetConditionedDeltaHead`. The per-example target index comes from the answer sheet
  (`target_col_idx`), passed out-of-band like the label. **Non-invasive, isolated to
  `lacuna/survey/`, fully reversible.** This is the rung-3 conditioning mechanism and the candidate
  P2 design if localization proves to be the bottleneck.

The audit recommends **(B)**; (A) is a later option only if head-side conditioning underperforms
and the marker is shown to matter. Conditioning is a NEW small module
(`survey/conditioned_head.py` / `survey/conditioning.py`), tested, ≤500 LOC, RNG-injected.

## 5. Coarse 3-bin scheme (rung 2)

A separate, recorded coarse mapping (do NOT mutate the canonical 7-bin `delta_bins`): bin 0
`{δ=0}` (MAR), bin 1 `weak (0, τ]`, bin 2 `strong (τ, ∞)`. Propose **τ = 1.0** (the midpoint edge
of the 7-bin scheme; δ>1 is "clearly strong" self-censoring), confirmed against the δ-grid. Provide
`assign_coarse_bin(δ)` + edge metadata + tests (boundaries, δ=0, δ<0 reject), mirroring
`delta_bins`. K=3 changes only the head's output width and the uniform/base-rate references; RPS,
metrics, leakage, manifest are K-agnostic (already parameterized by `num_bins`).

## 6. Decision rules (the user's, made operational)

| Outcome | Diagnosis | Action |
|---|---|---|
| **Rung 1 fails** | The δ-bin head/loss formulation is wrong, or δ-bin estimation is fundamentally harder than expected | Redesign head/loss before anything else: revisit RPS vs ordinal-log / cumulative-link (ordered-threshold) head; check optimization (LR, init, output scaling, flat-minimum near the marginal); add the labeled binary δ=0-vs-δ>0 diagnostic to confirm any gradient signal exists |
| **Rung 1 passes, rung 2 ALSO needed** | If rung 1 (7-bin) fails but rung 2 (3-bin) passes | Ordinal RESOLUTION is the limiter at this n — signal exists but 7 levels are unresolvable; consider fewer bins / coarser reporting / more evidence per example |
| **Rung 1 passes, rung 3 fails** | Real-X covariance geometry breaks extraction | Investigate the z-scoring/predictor-view on real survey columns, feature representation, geometry-induced confounds |
| **Rung 3 (target-known) passes, rung 4 (localized) fails** | Wide-table target localization / aggregation is the bottleneck | P2 adopts **explicit target-column conditioning (§4B)** or per-column processing as the design — not an optional add-on |
| **All simplified rungs pass** | Original P2.2 failure was task-complexity / SCALE, not conceptual | Proceed to rung 5 scale-up and a tuned main run; the conceptual approach is sound |

Each rung is run, reported, and reviewed before the next; the ladder **stops at the first failing
rung** with its implied diagnosis (no point climbing past a broken rung).

## 7. Manifest & audit (every rung)

Reuse `survey.run_manifest` with added recorded fields: `rung_id` (1–5), `axis_removed`,
`x_source ∈ {synthetic_2col, real_survey}`, `num_bins`, `target_conditioned` (bool + method A/B),
`n_rows_per_example`, `rho_grid` (rungs 1–2), and the standard from-scratch / leakage / metrics
blocks. `kind="main"` for the headline rung run; `kind="ablation"` for the binary diagnostic and
any frozen/conditioning ablation; `kind="smoke"` for the fast wiring check. **No rung number is
interpreted unless its manifest validates and (if interpreted) its leakage_pass is True.**

## 8. Proposed module plan (P2.2b — new code; production model + P2.2 path untouched)

- `lacuna/survey/synthetic_xtask.py` — 2-col synthetic δ-bin example builder: wrap
  `ConditionalGaussian.synthetic(ρ)` + `apply_self_censor` into the `batching.DeltaExample`
  shape (real reuse; no formula duplication). Rungs 1–2.
- `lacuna/survey/coarse_bins.py` — the 3-bin scheme (§5) + `assign_coarse_bin`.
- `lacuna/survey/conditioning.py` (or `conditioned_head.py`) — head-side target/predictor
  conditioning (§4B): gather per-column token reps at the supplied index, `TargetConditionedDeltaModel`.
- `lacuna/survey/ladder.py` — a thin rung runner that selects the data source + bins + conditioning
  and calls the EXISTING `train.train_delta_prior` eval path (or a light generalization of it to
  accept a pluggable example generator and num_bins). Emits the per-rung manifest.
- `scripts/run_p2p2b_ladder.py` — runs a chosen rung, prints the protocol table, writes the manifest.
- Tests for every new module (Rule 7); ≤500 LOC each (Rule 4); determinism via injected RNG
  (Rule 6); fail-loud (Rule 1). Likely small refactor: `train.train_delta_prior` to accept an
  injected `example_factory` + `num_bins` so all rungs share one loop (keeps it DRY and ≤500 LOC).

## 9. Sequencing & stop conditions

Implement and run **rung by rung, lowest first**, reporting after each:
1. Build rung-1 data + (if needed) the `num_bins`/`example_factory` seam in `train`; run rung 1.
2. If rung 1 fails → STOP, redesign head/loss per §6 (do not build rungs 2–5).
3. If rung 1 passes → rung 2 (coarse) and rung 3 (target-known real X); then, only if both learn,
   rung 4 (localized) and finally rung 5 (scale).
Each step: suite green, manifest valid, leakage gate evaluated, RPS-vs-references reported. No
P2.3 until the ladder localizes the bottleneck and a learnable configuration is demonstrated.

## 10. Constraints honored

No old 3-class objective; no binary MAR/MNAR as a main result (binary only as a labeled
`ablation` diagnostic); from-scratch for every interpreted run (no checkpoint, all layers
trainable); same blocking leakage gate; manifest-validated before any metric is read; v1.0
production model and the existing P2.1/P2.2 code paths left intact; new code isolated to
`lacuna/survey/` (Rule 8).

## 11. Open decisions for review (before coding P2.2b)

1. **Conditioning method:** head-side gather (§4B, recommended) vs token marker channel (§4A).
2. **Coarse τ:** propose τ=1.0 for the 3-bin split — confirm or adjust against the δ-grid.
3. **`train` refactor:** generalize `train_delta_prior` to a pluggable `example_factory` + `num_bins`
   (DRY across rungs) vs a separate ladder loop — propose the refactor (smaller total surface).
4. **Rung-1 ρ grid & n:** reuse the P1 profiled-oracle ρ regimes and n≈2000/example — confirm.
5. **Pass margin:** define "beats uniform/base-rate" as a fixed multiple of the test SE (propose
   ≥2·SE) so the pass criterion is objective and recorded.

## Approval gate

This is the P2.2b spec. **No code until approved.** On approval I will implement rung 1 first
(plus the minimal `train` seam + coarse-bin/conditioning scaffolding only as each rung needs it),
run it, get the suite green with a validated manifest, report RPS-vs-references + leakage, and
then proceed up the ladder rung-by-rung with review between — stopping (per §6/§9) at the first
rung that fails, with its diagnosis.
