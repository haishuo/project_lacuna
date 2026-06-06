# Stage-1 Implementation Spec — Level-1 φ-spine δ-prior (no code)

*Implementation spec only. **No code, no experiments until approved.** Status: **for PI approval.**
Stage 1 of the staged plan in `PROPOSAL-Level1-design-spec.md` §7 (decisions resolved 2026-06-05).
Module contracts are interface-level; **no code bodies**. Governed by `CLAUDE.md` (one-job modules,
≤500 LOC, fail-loud, determinism via injected RNG, tests first-class) and `NORTH-STAR.md`.*

## 0. Narrow goal & success criterion

> **Demonstrate that the new Level-1 column-primary φ spine reproduces the Stage-0 signal inside the
> governed Lacuna pipeline — with the named prior recorded and the non-survey contaminants removed.**

Success (pre-registered, held-out leave-datasets-out on the **9 genuine** surveys):
- **M1 (reproduction):** binary δ0-vs-δ2.5 OOF AUC ≈ **Stage-0 (~0.73)**, and **≫ the BERT-backbone
  ~0.55** — i.e. the φ spine carries the column-primary signal *inside the full pipeline*, not just in a
  probe script.
- **M2 (governed):** the model emits a **calibrated δ-prior over the δ-bin grid** (RPS + temperature),
  with calibration/coverage reported; **own-value ≈ prior marginal** (flatness sanity).
- **M3 (discipline):** **leakage gate passes**; the **manifest records the `named_prior` block**
  (incl. `prior_marginal`); determinism contract holds; every new module ≤500 LOC with tests; full suite
  green.

Stage 1 is **not** judged on beating raw-ECDF discrimination (non-goal, design-spec §8). It is judged on
*reproducing the signal in-pipeline* + *governance discipline*.

## 1. Scope (and exclusions)

**In scope:** column-major batching · φ distribution encoder · δ-prior head · named-prior manifest block ·
reuse of existing RPS / temperature-calibration / leakage gate / metrics / generators · the learning-curve
diagnostic plan · dropping the 3 contaminants.

**Out of scope (later stages):** mask-topology stream + detectability head (Stage 2) · OOD/abstention
(Stage 3) · calibration/reporting integration (Stage 4) · **Level-2 conditional reference/deviation** ·
**metadata channel** · **the BERT `LacunaEncoder` as the δ spine** (it is not imported into the Stage-1
model path).

## 2. Module map (each: one job, ≤500 LOC, fail-loud, deterministic, tested)

| module | new/reuse | one job | key contract (interface only) |
|---|---|---|---|
| `lacuna/survey/survey_catalog.py` (or a constant) | **new (small)** | the **curated** survey list = 9 genuine surveys (12 − {cars93, computers, survey}) | `GENUINE_SURVEYS: tuple[str,...]`; a helper that fails loud if asked for a contaminant in the δ path |
| `lacuna/survey/column_batching.py` | **new** | turn a list of `DeltaExample` into a **column-major** batch for the φ spine | `collate_columns(examples, *, max_rows) -> ColumnBatch` with fields: `target_values [B,Rmax,1]` (observed target values, standardized within observed), `value_mask [B,Rmax]` (True=observed), `target_idx [B]`, `delta_bin [B]`, `delta [B]`, `sheets`. Fail loud on empty list / all-missing target column. Determinism: row subsample via injected RNG. |
| `lacuna/survey/column_phi.py` | **new** (productionize Stage-0 `ColumnPhi`) | per-column distribution encoder | `ColumnPhi(m=16, quantile_levels=<12>, e_col=<moderate>)`; `forward(values [B,R,1], mask [B,R]) -> e [B,e_col]`. Reuses `distributional_stream.masked_quantile_pool` on **raw** values. Deterministic init via injected `RNGState` (shared `init_parameters_`). Fail loud on degenerate column (constant/`<MIN_OBS`) → documented safe embedding. |
| `lacuna/survey/level1_model.py` | **new** | assemble **φ → δ-bin head** (Stage-1 spine = φ only; fusion slot reserved for Stage-2 mask-topology) | `Level1Model(...).forward(batch) -> δ-bin logits [B,K]`; `predict_proba` (temperature); `set_temperature`. **No encoder, no v1.0 heads** (asserted in tests). |
| δ-bin head | **reuse** `delta_head.DeltaBinHead` | evidence→δ-bin logits | unchanged |
| training loop | **reuse + thin extension** `train.py` *or* `lacuna/survey/level1_train.py` | from-scratch RPS train, temperature on val, leakage, manifest | injected `RNGState`; emits `{model, results, manifest}`; **named_prior** passed through |
| `run_manifest.py` | **reuse + extend** | record `named_prior` | `build_manifest(..., named_prior=<dict>)`; `validate_manifest` requires it for Level-1 runs (§3) |
| RPS / temperature / metrics / leakage / generators / `masked_quantile_pool` / `delta_bins` / `consequence_features` | **reuse unchanged** | — | — |

**Note on `column_batching`:** it consumes the **existing** generators (`make_example`, `make_lod_example`)
to produce `DeltaExample`s (role **B** complete-projection + imposed holes), then re-expresses the target
column **column-major**. It does **not** tokenize per-cell row-major (that path is the BERT encoder's and
is not used here).

## 3. Named-prior manifest block (exact schema)

`build_manifest` gains a required `named_prior` argument; `validate_manifest` rejects a Level-1 manifest
lacking it. Schema:
```
named_prior:
  dataset_catalog: [ {name, n, d, content_hash} ]      # the GENUINE survey base tables used (role B)
  contaminants_excluded: ["survey_cars93","survey_computers","survey_survey"]
  idiom_vocabulary: [ {family, params} ]               # e.g. {own_value_self_censoring}, {lod_top_coding, tau}
  delta_grid: [...]
  delta_grid_weights: [...]                            # the δ sampling distribution
  prior_marginal: { bin_index: prob }                  # = the δ-grid marginal over bins (the flat-output target)
  rate_regime: { target_rate, matched: true }
  phi_config: { m, quantile_levels, e_col }
  data_role: "B_complete_projection"                   # binding: supervised stream is B (+C later), never A
```
This makes the prior reproducible and auditable, and records `prior_marginal` so the §3/§4 flat-output
behavior is checkable.

## 4. Training objective & calibration (reuse)

- **Main:** δ-bin **RPS** (`loss.rps_loss`) over the δ-grid; from-scratch (assert all-trainable, no
  checkpoint). Ordered bins via RPS (no cumulative-link architecture).
- **Calibration:** post-hoc **temperature** on val (`fit_temperature`); report ECE/coverage
  (`metrics`) before/after.
- **Reproduction metric:** alongside the grid output, evaluate the **binary δ0-vs-δ2.5 contrast** (held
  out) for M1. (Train on the grid; the binary contrast is an evaluation slice, not a separate objective.)
- **No** binary/3-class mechanism objective as a target. **No** detectability/OOD losses yet (Stages 2–3).

## 5. Evaluation & milestones

- **Corpus:** role-**B** semi-synthetic holes on the **9 genuine** surveys; **leave-datasets-out**
  split (e.g. 5 train / 2 val / 2 test), same discipline as prior runs.
- **Report:** held-out δ-bin **RPS + calibration/coverage**; the **binary δ0-vs-δ2.5 OOF AUC** (M1);
  **own-value flatness** (δ-prior ≈ `prior_marginal`, low info-gain) (M2); **leakage_pass** (M3).
- **Milestones:** M1 reproduction (AUC ≈ 0.73 ≫ 0.55); M2 calibrated grid output + own-value flat;
  M3 leakage pass + `named_prior` recorded + determinism + suite green + LOC limits.
- **Honest small-catalog caveat:** with only 9 genuine surveys the leave-datasets-out split is
  **under-powered and high-variance** → run **multiple seeds** and report variance; tie directly to §6.

## 6. Learning-curve diagnostic plan (parallel) — the deliverable scientific result

*Answered at the **dissertation/grant** standard (design-spec §9.A), **not** the deployment standard. The
question is "is there a credible positive scaling trend?", not "is this deployable?"*

- Train/eval **Level-0/φ-spine** at **N ∈ {4, 8, 9}** genuine training datasets (9 = max post-drop;
  reaching N>9 *requires* acquisition — which the curve exists to justify), **many seeds**, fixed held-out
  test set.
- **Report (per N, with seed/split variance):** held-out **δ-prior calibration/coverage** + **LOD
  sharpness** (binary OOF AUC/RPS) + **own-value prior-dominated behavior** + **variance**. *(Detectability-
  vs-oracle and OOD/abstention curves are added when Stages 2–3 land; Stage-1 reports the available
  subset.)*
- **Success bars (design-spec §9.A):**
  - **Dissertation:** Stage-0 signal reproduced in-pipeline (M1) **and** metrics **improve** 4→8→9.
  - **Grant:** the **slope is still positive at N=9** ⇒ acquisition is a justified next investment.
- **Read (per design-spec §9.7):** positive slope at 9 → **data-limited → acquire**; apparent saturation →
  architecture *candidate* **but not a verdict** (redundant 9; confirm with ≥1 genuinely new domain first).
- Uses only Level-0/φ + the eval ladder (cheap); runs **alongside** Stage-1 and **does not block** its
  completion. The curve is a **result to report**, not a gate to pass.

## 7. Test plan (Rule 7 — per module: normal / edge / failure)

- `column_batching`: column-major shapes & masks correct; within-observed standardization;
  variable observed-count padding; **fail loud** on empty list / all-missing target; determinism
  (same seed → same batch).
- `column_phi`: output shape `[B,e_col]`; gradient flow; **degenerate column** (constant / `<MIN_OBS`) →
  safe finite embedding; init determinism (injected RNG); reuses `masked_quantile_pool` (already tested).
- `level1_model`: forward shape `[B,K]`; backprop reaches φ; **no encoder / no v1.0 heads** present
  (assert `named_children`); temperature normalization; from-scratch all-trainable assert.
- `run_manifest`: `named_prior` required & validated; missing → fail loud; `prior_marginal` present.
- `survey_catalog`: the δ path **refuses** the 3 contaminants (fail loud if requested).
- training loop: leakage gate wired & blocking; manifest validates; determinism of data/eval.

## 8. Sequencing (each step gated; suite green before the next)

1. `survey_catalog` (genuine-9 constant + contaminant guard) + tests.
2. `column_batching` (role-B, column-major) + tests.
3. `column_phi` (productionized Stage-0 φ) + tests.
4. `level1_model` (φ → δ-bin head) + tests.
5. `run_manifest` `named_prior` extension + tests.
6. training-loop wiring (RPS + temperature + leakage + manifest) + a small end-to-end smoke (in-process,
   tiny) confirming M1/M2/M3 plumbing — **not** a full run.
7. the held-out leave-datasets-out evaluation run (M1–M3) **and** the learning-curve diagnostic (§6).

## 9. Non-goals (binding, restated)

No mask-topology/detectability (Stage 2) · no OOD/abstention (Stage 3) · **no Level-2 reference/
deviation** · **no metadata channel** · **no BERT backbone as the δ spine** · no natural-missingness in
supervision (role A is archive/validation only, §9.0) · no claim to beat raw-ECDF discrimination.

## 10. Open Stage-1 sub-decisions (small; for PI before implementation)

1. **`e_col`** (per-column embedding dim) — proposal: 32 (moderate; preserves Stage-0 signal, avoids a
   capacity sweep).
2. **δ-grid for Stage 1** — full 7-bin grid (`delta_bins`) vs a coarse grid. Proposal: **full 7-bin**
   (the governed output) **plus** the binary δ0-vs-δ2.5 evaluation slice for M1.
3. **Split** — which 5/2/2 of the 9 genuine surveys for train/val/test. Proposal: keep `cps1985`,
   `workinghours` as held-out test (continuity with prior runs); PI to confirm.
4. **Training loop home** — extend `train.py` (a `model_kind="level1"` branch) vs a new `level1_train.py`.
   Proposal: **new `level1_train.py`** (keeps `train.py` ≤ limits and the column-major path isolated).

**No implementation until this Stage-1 spec and these four sub-decisions are approved.**
