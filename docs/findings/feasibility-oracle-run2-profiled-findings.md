# Feasibility Oracle — Run 2: Point-null vs β₁′-Profiled Ceiling

*Run `profiled_20260603_100549` (CPU, no training, no checkpoint). 192 synthetic cells, point-null
AND β₁′-profiled Bayes error each. Bayes errors are MC estimates of the theoretical Bayes error of
the optimal LLR test, float64, SE/CI reported. Artifacts:
`/mnt/artifacts/project_lacuna/feasibility/profiled_20260603_100549/`.*

**Status: profiled (deployment-relevant) ceiling presented for review. NO model training run.**

## 1. Headline: the point-null "ρ helps" was largely a refittable-MAR artifact — profiling confirms it

Run-1's surprise (predictor–target correlation ρ *improves* MNAR-vs-MAR distinguishability) does **not
survive** a re-fitting MAR null. The β₁′-profiled null re-fits its predictor slope to absorb the
ρ-induced apparent coupling, and it absorbs most of the run-1 signal. The selected β₁′ rises with ρ
exactly as the absorption story predicts (β₁=1.0 cells, averaged): β₁′ ≈ 0.83 (ρ=0) → 1.11 (ρ=0.3) →
1.46 (ρ=0.6) → 1.94 (ρ=0.9). So the high-ρ "shadow help" was, in large part, slope a MAR can mimic.

**The corrected ceiling restores the Molenberghs intuition: higher ρ makes MNAR HARDER to distinguish.**
(β1=1.0, rate=0.1, point → profiled Bayes error):

| n | δ | ρ=0 | ρ=0.3 | ρ=0.6 | ρ=0.9 |
|---|---|---|---|---|---|
| 512 | 0.5 | 0.251→0.258 | 0.234→0.264 | 0.182→**0.303** | 0.106→**0.391** |
| 512 | 1.0 | 0.088→0.096 | 0.083→0.104 | 0.038→**0.121** | 0.012→**0.251** |
| 128 | 0.5 | 0.373→0.374 | 0.351→0.367 | 0.328→0.383 | 0.270→**0.453** |

At ρ=0 the profiled error ≈ the point error (a MAR cannot absorb a signal that has no predictor
channel). As ρ→0.9 the profiled error climbs toward chance. δ=0 floor behaves correctly: profiled
≈0.46–0.50 with β₁′≈1.0 (recovers the true MAR).

## 2. But the signal SURVIVES — there is a genuine, non-absorbable MNAR signature

Profiling does **not** push the error to chance except in one corner. **142 of 144 δ>0 cells remain
distinguishable** (profiled 95%-CI upper < 0.45). The own-value truncation of the observed marginal is
a signature no β₁-MAR can reproduce, and it is:
- **Fully genuine at ρ=0** (zero absorption): e.g. n=2048, δ=0.5: profiled 0.097; δ=1.0: 0.004.
- **Strong at moderate ρ / moderate δ**: n=512, ρ=0.3, δ=1.0: profiled 0.104; n=2048, ρ=0.6, δ=1.0: 0.015.
- **Recoverable with more data even at high ρ**: n=2048, ρ=0.9, δ=1.0: profiled 0.098 (vs near-chance
  at n=128).

**The one near-collapse corner** (where abstention should trigger): high ρ × small δ × small n —
e.g. ρ=0.9, δ=0.5, n=128: profiled 0.453 (CI [0.43, 0.47]). This is the boundary, mapped.

## 3. Verdict: the feasibility gate PASSES in the deployment-relevant sense

Under the β₁-flexible MAR null — the correct ceiling for what the deployment model faces — a genuine,
non-identifiable-by-MAR MNAR signal exists across a large, well-characterized region (low/moderate ρ,
δ≥0.5, n≥512; all of ρ=0). The "easy everywhere" picture of the point-null was an artifact; the real
ceiling is harder and ρ-graded, but **not** a global collapse. Proceeding to the model arm is
justified — to ask whether the real `LacunaModel` recovers this profiled-ceiling signal where it
exists, and whether it correctly degrades/abstains in the near-collapse corner.

## 4. Caveats (honest scope of the claim)

1. **Minimum-viable null.** This profiles over β₁′ ONLY, with a single β₁′ per cell (not a
   per-replicate GLRT, not profiling over MAR predictor choice or link family). A richer MAR null
   could absorb more, so **the surviving signal is an UPPER bound on identifiability** — the true
   deployment ceiling is at or below this. Widening the MAR family is the documented next robustness
   stage, to run IF we want a more conservative ceiling before trusting the model arm in marginal
   regions. (Charter discipline: do not overclaim identifiability.)
2. **Synthetic-X only.** Real-survey-X profiling not yet run (the fitted-Gaussian assumption would
   add a second caveat there).
3. **δ=0 floor** (~0.47–0.50) is the single-β₁′ design's known floor; read the profiled surface
   against its own δ=0 baseline, not against 0.5 exactly.

## 5. Recommended model-arm regimes (for approval — NOT yet run)

Four representative cells (β1=1.0, the genuine MAR null), spanning the profiled ceiling:

| regime | cell (ρ, δ, n, rate) | profiled Bayes error (ceiling) |
|---|---|---|
| near-chance control | ρ=0.9, δ=0.5, n=128, r=0.1 | 0.453 (≈ chance) |
| boundary | ρ=0.9, δ=1.0, n=512, r=0.1 | 0.251 |
| moderate | ρ=0.3, δ=1.0, n=512, r=0.1 | 0.104 |
| strong | ρ=0.0, δ=1.0, n=2048, r=0.1 | 0.004 |

The model arm (full `LacunaModel`, retrained from scratch, no checkpoint, all layers trainable) would
train/eval against these regimes and report the **gap to the profiled ceiling**: it should approach
the ceiling in moderate/strong, and — importantly — it should NOT beat the ≈chance ceiling in the
control (if it does, that signals leakage/confound, not skill). HOLDING for approval before any
training.
