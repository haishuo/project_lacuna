# P1 Model Arm — Implementation Audit (for approval BEFORE running)

*Governed by `docs/NORTH-STAR.md` (esp. §4.9) and the approved scope: train the deployment-strength
`LacunaModel` from scratch and measure its gap to the **β₁′-profiled MAR-null ceiling** (run-2
findings). SPEC ONLY — code is not yet committed; NOTHING is trained until this audit is approved.
The numbers below are grounded in an executable smoke (see §10).*

## Design in one line
For each regime, train the full `LacunaModel` to discriminate the **exact two distributions the
profiled oracle compared** — H1 = MNAR(δ, β₁=1) vs H0 = the profiled best-fit MAR(β₁′, β₂=0) — using
the **recorded fixed params** of that cell, so the model's achievable error is directly comparable to
the recorded profiled Bayes error (the Bayes-optimal error for that exact pair).

## 1. Exact selected parameter cells (the four regimes + a null control)
Synthetic-X (`ConditionalGaussian.synthetic(ρ)`), β₁=1.0, rate=0.1, 2 columns (col0 predictor
always observed; col1 target censored). Params are the **exact recorded values** from
`profiled_20260603_100549`:

| regime | ρ | δ | n | ceiling (profiled Bayes err ± SE) | H1: (β₀,β₁,β₂) | H0: (β₀′,β₁′,0) |
|---|---|---|---|---|---|---|
| near-chance control | 0.9 | 0.5 | 128 | 0.453 ± 0.011 | (−2.928, 1.0, 0.5) | (−2.925, 1.462, 0) |
| boundary | 0.9 | 1.0 | 512 | 0.251 ± 0.010 | (−3.344, 1.0, 1.0) | (−3.274, 1.864, 0) |
| moderate | 0.3 | 1.0 | 512 | 0.104 ± 0.007 | (−3.048, 1.0, 1.0) | (−2.675, 1.162, 0) |
| strong | 0.0 | 1.0 | 2048 | 0.004 ± 0.001 | (−2.877, 1.0, 1.0) | (−2.523, 0.926, 0) |
| **null control** | 0.9 | 0.0 | 512 | 0.500 (by construction) | (−2.93, 1.0, 0) | (−2.93, 1.0, 0) — *identical to H1* |

Masks are sampled with **fixed** params (not per-dataset rate-solving), so the generative process is
exactly the oracle's hypotheses. The null control's two classes are identical ⟹ true error 0.5 (a
leakage tripwire, §9).

## 2. Number of datasets per regime
Each "example" = one n-row, 2-column dataset; label ∈ {MAR(H0), MNAR(H1)}, balanced 50/50.
**Train 8000 / Val 2000 / Test 4000** datasets per regime (cheap: §10 shows ~0.03 s/step on GPU).

## 3. Train/validation/test split
Synthetic i.i.d. data ⟹ splits are **independent draws with disjoint RNG seeds** (train seed-block,
val seed-block, test seed-block; no example reused). Early stopping monitors **val binary error**;
the **test** set is touched once, after training, for the reported number. Seeds recorded in the
manifest. (No row/column or dataset overlap is possible — fresh generation per split.)

## 4. Model target / loss
- **Model:** the full `LacunaModel` via `create_lacuna_model` (hidden=128, evidence=64, 4 layers,
  4 heads, dropout 0.1, `learn_evidence_attenuation=True`), `max_cols=2`. **899,467 params.**
- **Labels:** H0→MAR (class id 1), H1→MNAR (class id 2), via the existing tokenizer `class_mapping`.
- **Loss:** negative log-likelihood / cross-entropy on the model's class posterior `p_class`.
- **Binary decision / error:** predict MNAR iff `p_class[MNAR] > p_class[MAR]`; `model_error` =
  fraction misclassified on the test set (directly comparable to the equal-prior binary Bayes error).
- **Training loop:** a minimal from-scratch loop (Adam, lr 3e-4, batch 32, grad-clip 1.0, early stop
  on val error, patience 10, max ~3000 steps). **We do NOT use the legacy `Trainer`** — its
  accuracy/confusion machinery assumes the 3-class objective (audit §2.5); a clean binary loop avoids
  that and is lower-risk. The MODEL is the full LacunaModel; only the training *loop* is minimal.

## 5. Expected runtime
Validated GPU step time **0.03 s** (B=32, n≤512, max_cols=2; §10). With data-gen + early-stopped
training, **~3–5 min per regime ⇒ ~15–25 min total** for all five regimes (strong's n=2048 data-gen
is the only heavier part; still minutes). CPU fallback would be ~30–50× slower; the run targets CUDA.

## 6. Checkpoint policy
**No checkpoint is ever loaded.** Each regime's model is freshly initialized (random) and trained
end-to-end. The best-val model per regime is *saved* to the artifacts dir for provenance only.
Manifest records `checkpoint_loaded=false`. (A v1.0 transfer ablation is explicitly NOT in this run.)

## 7. Manifest fields (required before any metric is read)
`arm="model"`, `kind="main"`, `checkpoint_loaded=false`, `all_layers_trainable=true`,
`trainable_param_count=899467`, `split_scheme` (seed blocks + train/val/test sizes), `grid` (the five
regimes + exact params + recorded ceilings), `xmodel` (synthetic ConditionalGaussian per ρ),
`generator_path`, `delta_formula`, `beta0_solver` (here: fixed recorded β₀, not solved),
`wall_clock_seconds`, `metrics`, `calibration`. `validate_manifest` enforces: kind=main ⟹ no
checkpoint AND all_layers_trainable=true (charter §4.9). **Metrics are not interpreted unless the
manifest validates.**

## 8. Metrics to report (per regime)
- `model_error` (test) ± binomial SE; `profiled_bayes_error` (ceiling) ± its MC SE.
- **`gap = model_error − profiled_bayes_error`** with a combined CI.
- Per-class realized missing rate (rate-match evidence).
- Best-val error + epochs-to-stop; trainable_param_count; checkpoint_loaded.
- Light **calibration**: mean predicted P(MNAR) vs empirical, and a 10-bin ECE (reported, not
  optimized).

## 9. Leakage checks (first-class; a failed check voids the regime's result)
1. **Null control** (regime 5, identical classes): `model_error` must be ≈ 0.5 (CI contains 0.5). If
   < 0.5 beyond MC tol ⟹ **leakage** (the model found structure the generative model says isn't
   there — e.g. a tokenization/ordering/padding artifact). Hard stop + investigate.
2. **Near-chance control gap:** `model_error` must not fall below the 0.453 ceiling beyond MC
   tolerance. A clearly **negative gap** here is a red flag for leakage/mismatch, **not** a victory
   (per your interpretation rules). Reported but does not dominate the feasibility conclusion.
3. **Rate-match:** mean realized missing rate of H0 and H1 must overlap (no rate cue, the known
   confound). Reported per regime.
4. **Provenance asserts:** `checkpoint_loaded=false`, `all_layers_trainable=true`, param count =
   899,467 — asserted in code and recorded in the manifest.
5. **Negative-gap flagging:** any regime with gap < −2·SE is flagged in the report as
   leakage/mismatch, never as the model beating the Bayes ceiling.

**Interpretation (your rules):** small positive gap = good (model approaches ceiling); large positive
gap = model leaving signal on the table; negative gap (esp. control) = leakage/mismatch. The main
feasibility question is sensible gap-to-ceiling behavior in **boundary / moderate / strong**; the
near-chance regime is a control.

## 10. Smoke evidence (already executed; no training of record)
- Full `LacunaModel` builds at max_cols=2: **899,467 params, all trainable** (`nparam==ntrainable`),
  no checkpoint.
- Forward+backward runs at n∈{128,512}, B=32: **0.03 s/step on CUDA**.
- Synthetic H0/H1 data-gen + `tokenize_and_batch` produces valid `[B,n,2,4]` token batches with
  `class_ids ∈ {MAR,MNAR}`; realized missing rates matched across classes (H1 0.11 vs H0 0.094 at a
  spot check — the rate-match check will quantify this per regime).

## 11. Non-goals for this run (explicit)
No RF/MLP/handcrafted baselines mixed into the main result; no v1.0 transfer ablation; no richer
MAR-null profiling (predictor/link-family) — deferred until after this first model-vs-ceiling
bracket, per your instruction.

## Approval gate
On approval I will: implement `lacuna/feasibility/model_arm.py` (data-gen + minimal loop) + tests →
suite green → run the five-regime model arm on GPU → present the gap-to-ceiling table with the
leakage checks. Nothing trains until you approve this audit.
