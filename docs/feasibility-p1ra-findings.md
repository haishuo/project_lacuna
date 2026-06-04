# P1R-A — Predictor-Choice Richer MAR Null — Findings

*Branch `experiment/p1r-richer-mar-null`. Run `p1ra_20260604_091846` (CPU, 0.5 min, manifest VALID).
3-column synthetic-X [z_p, z_a, z_t]; restricted (z_p-only) vs richer (predictor-choice over {z_p, z_a})
profiled MAR null on the SAME 3-column data; 4 P1 regimes × ρ_a ∈ {0, 0.3, 0.6, 0.9, 0.95, 0.99}.*

## Headline: predictor-choice does NOT absorb the P1 signal — P1 survives this richer null

**Δ = E_richer − E_restricted = 0.000 in all 24 cells; z_p is the least-favorable (hardest-to-distinguish)
MAR predictor in every cell, even at ρ_a=0.99; no monotonicity violations.** Predictor choice does not
raise the Bayes error toward chance anywhere in the P1 regimes.

| regime | ρ_a=0.0 | 0.3 | 0.6 | 0.9 | 0.95 | 0.99 | (2-col P1 ceiling) |
|---|---|---|---|---|---|---|---|
| control | 0.424 | 0.445 | 0.417 | 0.412 | 0.394 | 0.368 | 0.453 |
| boundary | 0.273 | 0.250 | 0.233 | 0.167 | 0.132 | 0.115 | 0.251 |
| moderate | 0.103 | 0.064 | 0.025 | 0.005 | 0.003 | 0.004 | 0.104 |
| strong | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 |

(Values are E_restricted = E_richer since Δ=0 everywhere.) Two things to read:
- **The added proxy column LOWERS the Bayes error (makes MNAR EASIER to detect), not higher.** As ρ_a
  rises, `z_a` becomes a shadow variable that helps the *discriminator* reconstruct the censored target
  and detect the truncation. The richer single-predictor MAR null **cannot exploit it to mimic better**.
- At ρ_a=0 (z_a is pure noise) E_restricted ≈ the 2-column P1 ceiling (sanity: the 3-col-with-noise case
  reduces to 2-col P1). The 2-col ceiling is secondary context only, not the monotonic comparison.

## Why (the analytical reason, confirmed)
In every P1 regime β₁=1 ≥ δ, and H1's missingness depends on **both** the designated driver z_p (β₁)
**and** the own value z_t (δ). No single observed column captures both: z_p captures the driver but
misses the own-value (MNAR) residual; a proxy z_a captures the own-value part (to extent ρ_a) but misses
the driver. The best single MAR predictor (least-favorable null member) is therefore z_p whenever β₁≥δ —
even for a near-perfect proxy ρ_a=0.99. So single-predictor CHOICE over a candidate set containing the
true driver cannot absorb the signal. (z_a is selected only in own-value-dominated regimes, β₁<δ — unit-
tested separately; not the P1 regimes.)

## A bug the monotonicity tripwire caught (recorded for honesty)
The first run (`p1ra_20260604_091451`) showed **negative Δ at ρ_a=0.99** (control −0.129, boundary
−0.039) — flagged by `any_monotonicity_violation=true`. Root cause: the richer null's predictor was
selected by **mask-fit likelihood** (the stated KL objective), computed on the missingness indicator
only. At ρ_a→1 the mask-fit prefers z_a (missingness correlates with z_a≈z_t), but the *observed-data*
discriminator exploits z_a as a shadow, making MAR-on-z_a MORE distinguishable — so reporting z_a's Bayes
error gave E_richer < E_restricted, which is impossible for a superset null. **Fix:** select the richer
null's member as the **least-favorable = maximum-Bayes-error candidate** (the correct composite-null
criterion), which guarantees E_richer ≥ E_restricted. The mask-fit selection is retained only as a
diagnostic. After the fix, no violations; z_p selected everywhere.

## Scoped conclusion
**The P1 conclusion survives the P1R-A predictor-choice richer MAR null:** giving MAR a free choice
among observed single predictors does not collapse the MNAR signal in any P1 regime, even with a
near-perfect observed proxy of the censored target. If anything, a correlated observed column makes the
mechanism MORE identifiable (shadow). The P1 headline stands, now with the tested qualifier that
**single-predictor choice is not a threat.**

## What this does NOT rule out — and the recommended next step
P1R-A tests a MAR null restricted to ONE observed predictor. The genuine absorption risk lives in a
**multi-predictor MAR null** that uses a *linear combination* of z_p AND z_a (β₁′·z_p + β₂′·z_a): that
could capture the driver AND the own-value proxy simultaneously, which single-predictor choice provably
cannot. This is strictly richer than P1R-A and is the experiment that could actually move the P1
headline. **Recommendation: prefer multi-predictor MAR over P1R-B (link-family).** Link-family profiling
(probit/cloglog) only changes the link shape on a single predictor — like predictor choice, it cannot
capture the own-value residual, so it is very unlikely to absorb where predictor choice did not.
