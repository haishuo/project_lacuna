# P1R-C — Multi-Predictor MAR Null — Findings (P1 SURVIVES; governance boundary = Var(z_t|obs))

*Branch `experiment/p1r-richer-mar-null`. Run `p1rc_20260604_100943` (CPU, 2.8 min, manifest VALID,
no monotonicity violations, no grid edges). Restricted single-predictor (β₂′=0) vs multi-predictor MAR
`σ(β₀′+β₁′z_p+β₂′z_a)` on the same 3-column data; least-favorable null by MAX Bayes error; 4 P1 regimes
× ρ_a ∈ {0, 0.3, 0.6, 0.9, 0.95, 0.99}; ρ_pa = ρ_orig·ρ_a. This is the final pre-P2 robustness gate.*

## Headline: the multi-predictor null GENUINELY absorbs (unlike P1R-A), but P1 still survives

The multi-predictor MAR uses z_a as a proxy (selected β₂′ grows with ρ_a), and Δ = E_multi − E_restricted
grows with ρ_a — real absorption that single-predictor choice (P1R-A, Δ≡0) could not produce. **But the
signal collapses toward chance ONLY as ρ_a→1, and only the already-near-chance control regime actually
reaches chance.** The substantive regimes stay distinguishable even at ρ_a=0.99.

**E_multi (= least-favorable multi-predictor Bayes error):**

| regime | ρ_a=0 | 0.3 | 0.6 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| control (ceil 0.453) | 0.447 | 0.422 | 0.433 | 0.453 | 0.437 | 0.476 |
| boundary (0.251) | 0.252 | 0.257 | 0.273 | 0.306 | 0.341 | 0.409 |
| moderate (0.104) | 0.092 | 0.101 | 0.144 | 0.265 | 0.336 | 0.407 |
| strong (0.004) | 0.003 | 0.005 | 0.012 | 0.110 | 0.190 | 0.341 |

- **No collapse at moderate ρ_a (≤0.6):** boundary 0.273, moderate 0.144, strong 0.012 — all well below
  chance. `collapse_at_moderate_ρ_a = False`.
- **Collapse emerges only at ρ_a ≥ 0.9**, and only the **control** regime (which began at chance) reaches
  E_multi ≥ 0.45. Boundary/moderate/strong stay distinguishable (0.34–0.41) even at ρ_a = 0.99.
- Monotonicity clean (Δ ≥ 0 within MC; a couple of −0.007/−0.010 cells are MC noise, none flagged).

## The result is governed by Var(z_t | z_p, z_a) — the North Star boundary

Absorption tracks the **conditional variance of the censored target given the observed predictors**
(the unabsorbable residual β₂·ε has variance β₂²·Var(z_t|z_p,z_a)):

| | ρ_a=0 | 0.6 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| Var(z_t\|z_p,z_a), ρ_orig=0.9 | 0.190 | 0.172 | 0.105 | 0.069 | 0.018 |
| Var(z_t\|z_p,z_a), ρ_orig=0.3 | 0.910 | 0.602 | 0.186 | 0.097 | 0.020 |

As Var(z_t|obs)→0 (an observed column nearly reconstructs the censored target), E_multi→chance — MNAR
becomes indistinguishable from a multi-predictor MAR. When substantial residual variance remains
(Var ≳ 0.1, i.e. ρ_a ≲ 0.9), the own-value signal survives. Detectability also scales with n (strong,
n=2048, is harder to collapse at equal Var than boundary/moderate, n=512).

**This is exactly the governance dial the North Star calls for:** the proxy-aware abstention rule is
*Var(z_t | observed predictors)* — abstain / widen uncertainty when a suspected-MNAR column is nearly
reconstructible from observed columns; trust the MNAR signal when it is not.

## Verdict against the pre-declared interpretation rules
- "Collapse only near ρ_a ≈ 1 ⇒ P1 robust enough; proceed to P2 with a near-proxy abstention caveat."
  **— This is the outcome, and it is stronger than the bar:** only the near-perfect-proxy regime (ρ_a→1,
  Var→0) collapses, and only the already-near-chance control fully reaches chance.
- "Collapse at moderate ρ_a ⇒ revise headline." **— Not triggered** (no moderate-ρ_a collapse).
- "Moderate/strong survive ⇒ P1 substantially stronger." **— Yes:** moderate/strong survive a
  multi-predictor MAR with a strong proxy up to ρ_a≈0.9; P1 is strengthened, not weakened.

**No catastrophic collapse.** Per the agreed framing, we proceed to the Lacuna re-architecture (P2).

## Consequence for P2 (the product)
P1R-C converts the earlier hand-wave ("absent a strong observed proxy") into a **quantitative,
implementable feature**: the re-architected δ-prior / sensitivity-prior Lacuna must compute (or estimate)
**Var(z_t | observed)** per suspected-MNAR column and use it to drive **proxy-aware abstention /
uncertainty widening**. This is now a core, empirically-grounded requirement, not an afterthought.

## Scope / caveats (unchanged discipline)
Semi-synthetic, Gaussian X, single censored column, own-value self-censoring axis; ρ_pa = ρ_orig·ρ_a
(one modeling choice for the predictor–predictor correlation). The least-favorable null is a finite-grid
MAX Bayes error ⇒ a conservative lower bound on the true least-favorable (so true absorption could be
marginally higher, which would only sharpen the same Var-based abstention rule). Per the plan, oracle
expansion stops here.
