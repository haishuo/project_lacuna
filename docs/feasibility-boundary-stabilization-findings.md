# P1 Boundary Stabilization — Findings (SUCCESS)

*Run `boundary_stab_20260603_230213` (GPU, 188 min, manifest VALID). 10 restarts (seeds 100–109) of
the full `LacunaModel` from scratch on the FIXED boundary H0/H1 pair (ρ=0.9, δ=1.0, n=512), recipe:
lr 1e-4 / 300-step warmup / cosine→1e-5 / patience 20 / max 80 epochs. One shared fresh held-out test
set (seed 9000); checkpoints selected by validation only. Ceiling held fixed at 0.251.*

## Result: stabilization SUCCEEDS (and far exceeds the pre-declared bar)

| seed | test_err ± SE | best_val | gap | epochs | escaped | ECE |
|---|---|---|---|---|---|---|
| 100 | 0.279 ± 0.007 | 0.266 | +0.028 | 57 | ✓ | 0.034 |
| 101 | 0.276 ± 0.007 | 0.248 | +0.025 | 68 | ✓ | 0.027 |
| 102 | 0.279 ± 0.007 | 0.262 | +0.028 | 80 | ✓ | 0.029 |
| 103 | 0.274 ± 0.007 | 0.252 | +0.023 | 80 | ✓ | 0.026 |
| 104 | 0.275 ± 0.007 | 0.264 | +0.024 | 50 | ✓ | 0.041 |
| 105 | 0.278 ± 0.007 | 0.263 | +0.027 | 65 | ✓ | 0.037 |
| 106 | 0.280 ± 0.007 | 0.259 | +0.029 | 49 | ✓ | 0.055 |
| 107 | 0.273 ± 0.007 | 0.254 | +0.022 | 61 | ✓ | 0.030 |
| 108 | 0.276 ± 0.007 | 0.277 | +0.025 | 73 | ✓ | 0.027 |
| 109 | 0.273 ± 0.007 | 0.241 | +0.022 | 70 | ✓ | 0.040 |

**Aggregate:** escape fraction **10/10**; test error **0.276 ± 0.0024** (mean ± std); median **0.276**;
best (val-selected, seed 109) **0.273**; gap mean **+0.025**, median **+0.025**; ECE mean **0.035**;
realized rates H0/H1 **0.101 / 0.102** (matched); **no suspicious negative gaps**; manifest valid.

## Against the pre-declared success criteria
- Escape majority (≥6/10 with err < 0.40): **met — 10/10.**
- Median val-selected test error ≤ 0.32: **met — 0.276.**
- No negative gaps / leakage: **met** (min gap +0.022; rates matched).
- Manifest valid: **met.**
⇒ `stabilization_success = True`.

## Before vs after (the optimization fix, isolated)
| | escape | test error per/agg | gap |
|---|---|---|---|
| baseline (seeds 42/43/44, lr 3e-4, no warmup/decay, pat 10) | 1/3 | 0.309 / 0.480 / 0.458 (0.416 ± 0.076) | +0.165 |
| stabilized (10 restarts, lr 1e-4, warmup+cosine, pat 20) | **10/10** | **0.276 ± 0.0024** | **+0.025** |

The std collapsed ~30× (0.076 → 0.0024) and the gap fell from +0.165 to +0.025. The model now **reaches
the profiled Bayes ceiling at high ρ, reliably, across every restart.**

## Scoped conclusion
The high-ρ boundary instability was an **optimization-procedure** problem, fully resolved by
training-procedure changes alone — lower LR + warmup + cosine decay + longer patience — **with no
architecture, data-regime, generator, or oracle changes, and no checkpoint loaded.** Combined with the
earlier moderate/strong results (gaps +0.014 / +0.007), the full LacunaModel now recovers the
non-absorbable MNAR signal near the β₁′-profiled ceiling across the entire easy→hard span, including the
hardest high-ρ boundary. Leakage controls remained clean throughout.

## Standing caveat (unchanged)
The ceiling is the minimum-viable β₁′-profiled null (single fit, β₁′-only, synthetic-X). A richer MAR
null (predictor/link or per-replicate GLRT) could lower it; real-survey-X profiling is not yet run.
Those remain the documented next robustness stages — deferred per the current plan.
