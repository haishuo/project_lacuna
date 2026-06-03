# Feasibility Model Arm — P1 Pilot (1 seed) Findings

*Run `modelarm_pilot_20260603_190551` (GPU, 55.9 min, seed 42). Full `LacunaModel` (899,275 params)
trained FROM SCRATCH per regime; binary H0-vs-H1 test (MCAR-safe renormalized MAR/MNAR head); fixed
oracle params; gap measured vs the recorded β₁′-profiled Bayes ceiling. **Manifest validated.**
Artifacts: `/mnt/artifacts/project_lacuna/feasibility/modelarm_pilot_20260603_190551/`.*

## Validity (leakage + provenance — all pass)
- **Manifest valid**; `checkpoint_loaded=False`, `all_layers_trainable=True`, 899,275 params, fixed
  oracle params, MCAR-safe binary head.
- **Null control = 0.500** (identical H0/H1) → no leakage.
- **Per-class realized rates matched** at ~0.10 in every regime → no rate cue (the known confound).
- **No suspicious negative gaps**; `valid=True`.

## Gap-to-ceiling table (1 seed)

| regime | ρ | δ | n | ceiling | model_error ± SE | gap | epochs | ECE |
|---|---|---|---|---|---|---|---|---|
| near-chance control | 0.9 | 0.5 | 128 | 0.453 | 0.498 ± 0.008 | **+0.046** | 13 | 0.007 |
| boundary | 0.9 | 1.0 | 512 | 0.251 | 0.473 ± 0.008 | **+0.222** | 15 | 0.024 |
| moderate | 0.3 | 1.0 | 512 | 0.104 | 0.127 ± 0.005 | **+0.023** | 39 | 0.022 |
| strong | 0.0 | 1.0 | 2048 | 0.004 | 0.009 ± 0.002 | **+0.006** | 33 | 0.005 |
| null control | 0.9 | 0.0 | 512 | 0.500 | 0.500 ± 0.008 | +0.000 | 12 | 0.049 |

## Reading (per the approved interpretation rules)
- **Strong (+0.006) and moderate (+0.023): the model essentially reaches the ceiling.** The full
  LacunaModel, trained from scratch, recovers the profiled-ceiling signal where it is not too subtle.
  This is the core positive result: the architecture CAN learn the non-absorbable MNAR signature.
- **Near-chance control behaves correctly**: model ≈ chance (0.498), does NOT beat the ≈chance ceiling
  (small positive gap). No leakage — the control did its job.
- **Boundary is the failure: gap +0.222.** The ceiling says 0.251 is achievable, but the model sat
  near chance (0.473) and **early-stopped at epoch 15** — it plateaued in a chance basin. By the
  interpretation rules this is a **large positive gap = the model leaving signal on the table**, NOT
  leakage and NOT a fundamental limit (the oracle proves the signal exists). It is concentrated in the
  **high-ρ (0.9)** regime, where the non-absorbable signature is subtlest.

## Interpretation
The pipeline is valid and the model is capable: it tracks the ceiling in 3 of 4 signal regimes
(control trivially, moderate, strong). The one real shortfall is the **high-ρ boundary**, where the
model got stuck at chance. The two candidate explanations:
1. **Optimization/seed sensitivity** — early stop at epoch 15 suggests a chance basin a different
   seed / schedule might escape. (Most likely; testable by multi-seed.)
2. **A genuine learning difficulty** at high ρ (near-collinear columns) that this architecture/training
   recipe doesn't crack — which would be a real, scoped finding (not a contradiction of the ceiling).

## Next step (pre-authorized): 3-seed follow-up
Per the approval ("if promising or borderline, immediately follow with ≥3 seeds for boundary/moderate/
strong"), run those three regimes across seeds {42, 43, 44} and report mean ± std of model_error / gap.
If boundary recovers on some seeds → optimization sensitivity (fixable by seed/schedule). If boundary
robustly fails across seeds → a genuine high-ρ learning gap to characterize. Moderate/strong expected
to remain near-ceiling (variance check). No stronger claim is made until the multi-seed is in.

## Scope reminder
This is the minimum-viable β₁′-profiled ceiling (single fit, β₁′-only, synthetic-X). A richer MAR null
could lower the ceiling; the model-vs-ceiling story here is conditional on that ceiling, as documented.
