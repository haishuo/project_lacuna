# P1R-C — Multi-Predictor MAR Null — Implementation Audit (for approval BEFORE coding)

*Branch `experiment/p1r-richer-mar-null`. The first richer null that can GENUINELY absorb the P1
signal: a MAR using a linear combination of observed predictors. CPU oracle only; NO model training;
NO Lacuna-model or production-generator changes; NO baselines. SPEC ONLY — no code until approved.
Vocabulary: only the analytic Bayes oracle, under its stated X-model, is a "ceiling."*

## 0. Why this one can absorb (the analytical structure — read first)

H1's missingness logit is `β₀ + β₁·z_p + β₂·z_t`. Decompose the censored target into its projection on
the observed predictors plus an orthogonal residual: `z_t = a·z_p + b·z_a + ε`, with `ε ⟂ (z_p, z_a)`.
Then `H1 logit = (β₁ + β₂a)·z_p + (β₂b)·z_a + β₂·ε + β₀`. A **multi-predictor MAR**
`σ(β₀′ + β₁′·z_p + β₂′·z_a)` can match the z_p and z_a parts **exactly** (set β₁′=β₁+β₂a, β₂′=β₂b),
leaving the **unabsorbable residual `β₂·ε`** — the own-value dependence on the part of z_t that **no
observed column predicts**. So:

> **The MNAR signal that survives a multi-predictor MAR null is governed by `Var(z_t | z_p, z_a)` — the
> conditional variance of the censored target given all observed predictors.** As ρ_a→1, `z_a`→`z_t`,
> `Var(z_t|z_p,z_a)→0`, the residual vanishes, and MNAR becomes indistinguishable from MAR (Bayes
> error → chance). This is the sharpest form of the Molenberghs floor: MNAR is identifiable only to the
> extent the censored variable has variance unexplained by observed columns.

For the default 3-col structure (§2), `Var(z_t|z_p,z_a) = 1 − (ρ_orig²+ρ_a²−2ρ_orig²ρ_a²)/(1−ρ_orig²ρ_a²)`.
E.g. boundary (ρ_orig=0.9): single-predictor leaves Var=0.19; multi-predictor at ρ_a=0.9 leaves 0.105;
at ρ_a=0.99, ≈0.02. So we EXPECT graded absorption that single-predictor choice could not produce. This
is precisely the decision-relevant test.

## 1. Hypotheses
- **H1 (MNAR), fixed = the P1 cell:** `σ(β₀ + β₁·z_p + β₂·z_t)`, β₂=δ, β₁=1, β₀ recorded. Unchanged.
- **H0 (multi-predictor MAR):** `σ(β₀′ + β₁′·z_p + β₂′·z_a)`, β₀′ rate-matched, (β₁′, β₂′) profiled.
- **Restricted null (the comparator) = the P1R-A restricted null:** the same with **β₂′ forced to 0**
  (single predictor z_p). It is the β₂′=0 slice of the multi-predictor family ⇒ a true subset.

## 2. X design (same 3-column MVN as P1R-A)
`MultivariateGaussianX` over [z_p, z_a, z_t], unit-diagonal, **PSD-checked (Cholesky)**.
`corr(z_p,z_t)=ρ_orig` (P1 regime value), `corr(z_a,z_t)=ρ_a` (**swept knob**),
`corr(z_p,z_a)=ρ_pa`. **Default ρ_pa = ρ_orig·ρ_a** (the conditional-independence value: z_p ⟂ z_a | z_t
— two independent noisy readings of the target), which is always PSD and is the natural "two proxies"
interpretation. **ρ_pa materially affects absorption** (it sets how much *joint* information the two
predictors carry about z_t, hence Var(z_t|z_p,z_a)); it is recorded in the manifest and is sweepable in
a later stage. Reuses `build_p1ra_corr` (already PSD-tested).

## 3. Profiling objective — MAX Bayes error, NOT max mask-likelihood (the P1R-A lesson)
The composite null's ceiling is the **least-favorable member = the (β₁′, β₂′) that MAXIMIZES the Bayes
error** (minimizes distinguishability from H1). P1R-A proved that max-mask-likelihood (= min KL(H1‖H0))
can DISAGREE with Bayes error under a shadow column and produce a monotonicity violation; therefore the
selection MUST be by Bayes error, not by mask-fit. The KL/mask-fit point is used only as a **search
center / diagnostic**, never as the selection criterion. We record the selected (β₁′, β₂′) AND its
Bayes error.

## 4. Optimization (how β₁′, β₂′ are profiled; monotonicity guaranteed by construction)
1. **Center:** the analytical projection coefficients (β₁′₀, β₂′₀) = (β₁+β₂·a, β₂·b), where (a, b) are
   the regression coefficients of z_t on (z_p, z_a) from the X-model (closed form: `Σ_oo⁻¹ Σ_ot`). This
   is the min-KL center where absorption is expected.
2. **Search set:** a 2-D grid over (β₁′, β₂′) around the center (e.g. ±100% in each, 7×7), **PLUS the
   restricted point (β₂′=0, β₁′ = the P1R-A single-predictor profile)**. β₀′ is rate-matched at EVERY
   grid point (§ below).
3. **Evaluate Bayes error (MC) at every point; E_multi = MAX over the set; selected coeffs = argmax.**
   Because the restricted point is in the set, **E_multi ≥ E_restricted by construction** (monotonicity
   cannot be violated). Refine the argmax with a higher-n_mc Bayes-error estimate.
4. **β₀′ rate-matching (per candidate (β₁′,β₂′)):** the linear index `β₁′z_p+β₂′z_a` is Gaussian with
   variance `s² = β₁′²+β₂′²+2β₁′β₂′ρ_pa`; solve β₀′ so `E_{ξ~N(0,s²)}[σ(β₀′+ξ)] = target_rate` by
   Gauss–Hermite + bisection (monotone in β₀′). Fail loud if unbracketable.
5. **Bounds / failure handling:** coefficient grid bounded to a documented box (e.g. |β| ≤ 8); skip/raise
   on non-PSD X (can't occur with default ρ_pa); β₀′ solve guarded.
6. **Honesty about the grid:** a finite grid gives an *approximate* least-favorable null, so reported
   E_multi is a **lower bound** on the true least-favorable Bayes error. Direction of the bias is
   conservative for "P1 survives" claims (we could under-state absorption), so the argmax is refined and,
   if it lands on a grid edge, the grid is expanded. This is stated in the findings.

## 5. Comparison (headline vs context)
- **Headline (monotonic):** restricted single-predictor MAR vs multi-predictor MAR **on the same
  3-column data** ⇒ `Δ = E_multi − E_restricted ≥ 0`. Report per (regime, ρ_a) with MC SE/CI.
- **Secondary context only:** E_restricted vs the 2-column P1 ceiling (the shadow-detection shift),
  clearly labeled non-monotonic context, never the headline.

## 6. Tests (before any run)
- **Reduction:** forcing β₂′=0 reproduces the P1R-A restricted-null Bayes error (regression check).
- **Superset monotonicity:** `E_multi ≥ E_restricted − 2·SE` on constructed cells; the restricted point
  is in the search set so a violation beyond MC tol fails the test (bug/MC-noise tripwire).
- **PSD / correlation:** non-PSD raises; default ρ_pa always PSD; sampled covariance matches spec.
- **Rate-matching:** β₀′ hits target_rate for representative (β₁′, β₂′) incl. large |β₂′|.
- **δ=0 floor:** δ=0 ⇒ Bayes error ≈ 0.5 (within the single-fit floor) for both nulls.
- **Coefficient recording:** selected (β₁′, β₂′, β₀′) and Bayes error present in the cell output.
- **Known-absorption case:** a high-ρ_a cell (e.g. ρ_orig=0.9, ρ_a=0.95, β₁=δ=1) where multi-predictor
  MAR absorbs MORE than single-predictor choice did (E_multi > E_restricted by a clear margin, and >
  the P1R-A Δ=0) — directly contrasting with the P1R-A null result.

## 7. Initial run plan
- CPU oracle only; no training; no production-generator changes.
- Representative regimes first: **control, boundary, moderate, strong** (same P1 cells).
- Sweep the same interpretable grid **ρ_a ∈ {0, 0.3, 0.6, 0.9, 0.95, 0.99}**.
- Default ρ_pa = ρ_orig·ρ_a. Report the table (E_restricted, E_multi, Δ, selected β₁′/β₂′, Var(z_t|z_p,z_a))
  and the monotonicity check, then STOP before any wider (ρ_pa or finer) surface.

## 8. Interpretation (pre-declared)
- **Δ ≈ 0 until ρ_a near 1:** P1 is robust except when the table contains an almost-direct observed
  proxy for the censored item; the P1 headline stands with that qualifier.
- **Moderate ρ_a (≤ ~0.6) collapses signal (Δ large, E_multi → chance):** the P1 ceiling was optimistic
  for realistic tables with correlated proxy variables — the P1 headline must be **revised**, and the
  future Lacuna product needs explicit **proxy-aware abstention** (flag when an observed column is a
  strong predictor of a suspected-MNAR column).
- **Moderate/strong regimes survive (Δ small, E_multi well below chance) across ρ_a:** P1 becomes
  **substantially stronger** — even a multi-predictor MAR with a good proxy cannot absorb the own-value
  residual there.
- A negative Δ beyond MC tolerance remains a bug/MC-noise flag (cannot occur given the restricted point
  is in the search set; if it appears, investigate).

## Approval gate
On approval: implement the multi-predictor H0 (generalize the `profiled_oracle_mv` LLR/rate-match to a
coefficient vector over [z_p, z_a]; new module, P1R-A code untouched) + tests → suite green → run the
CPU representative-regime ρ_a sweep → present the table with the monotonicity check and the
Var(z_t|z_p,z_a) column. Nothing runs until approved.
