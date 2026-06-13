# P1R — Richer MAR Null — Implementation Audit (for approval BEFORE coding)

*Branch `experiment/p1r-richer-mar-null` (from tag `p1-feasibility-passed`). Goal: test whether the
P1 conclusion survives a richer MAR null than the β₁′-only profiled null. CPU oracle only; NO model
training; NO Lacuna-model changes; NO production-generator changes; NO baselines. SPEC ONLY — no code
until approved. Vocabulary held: only the analytic Bayes oracle, under its stated X-model, is a
"ceiling."*

## 0. The key question, answered first

**Does predictor-choice profiling require moving from the 2-column synthetic-X setup to multi-column?
YES — it is impossible and meaningless in 2 columns.** In the P1 setup there is exactly one observed
column (the predictor `z_p`); the target `z_t` is the censored column and cannot be its own MAR
predictor. So "choose among candidate observed predictors" has a candidate set of size 1 — there is no
choice. **P1R-A therefore requires a minimal multi-column X (3 columns).** Design in §2.

**Consequence that governs the whole experiment (state up front, do not bury):** adding an observed
column changes the *observed data itself*, not just the null hypothesis set. A new observed column
correlated with the censored target is **a shadow variable**: it simultaneously (i) lets the MAR null
mimic MNAR better (raising Bayes error) AND (ii) gives the *discriminator* more power to detect the
truncation (lowering Bayes error). Therefore the clean, theorem-backed monotonic comparison —
"a richer null can only make the problem harder or equal" — holds **only between the richer null and
the restricted (single-predictor) null evaluated on the SAME multi-column data**, NOT between the
richer-null ceiling and the original 2-column P1 ceiling (0.251 etc.), which differ because the
observed data differs. The experiment computes BOTH ceilings on the same 3-column data and reports the
richer-vs-restricted gap as the monotonic result, with the change-from-2-column-P1-ceiling as
secondary context.

## 1. Hypotheses (H1 unchanged from P1; only the null is enriched)

- **H1 (MNAR), identical to the P1 regime cell:** `P(miss_t) = σ(β₀ + β₁·z_p + β₂·z_t)`, β₂=δ, β₁=1,
  β₀ the recorded P1 value. Depends on the original predictor `z_p` and the target's own value `z_t`;
  does NOT depend on the added column. (So β₀ rate-match is unchanged — it never involved the added
  column — and H1 is literally the P1 H1, just embedded in a 3-column table.)
- **H0 (richer MAR):** `P(miss_t) = σ(β₀′ + β₁′·z_j)`, β₂=0, for a CHOSEN observed predictor
  `j ∈ candidates`, with β₀′ rate-matched and β₁′ profiled — selected to best fit H1 (objective §5).
- **H0 (restricted MAR) = the P1 null:** the same but with `j` fixed to the original predictor `z_p`.

## 2. Minimal multi-column synthetic-X design (3 columns)

`MultivariateGaussianX` over `[z_p, z_a, z_t]` (predictor, added column, target), standard-normal
marginals, correlation matrix
```
        z_p    z_a    z_t
 z_p [  1     ρ_pa   ρ_orig ]
 z_a [ ρ_pa   1      ρ_a    ]
 z_t [ ρ_orig ρ_a    1      ]
```
- **ρ_orig** = the P1 regime's target–predictor correlation (boundary 0.9, moderate 0.3, strong 0.0,
  control 0.9) — so col_p reproduces the P1 predictor exactly.
- **ρ_a** = the added column's correlation with the target — **the new swept knob** (the strength of
  the alternative shadow/predictor).
- **ρ_pa** = predictor–added correlation. Default the 1-factor-consistent / a documented value; the
  matrix is **PSD-validated by Cholesky (fail loud if not PSD)**. (Alternative always-PSD construction:
  a 1-factor latent `z_j = λ_j f + √(1−λ_j²)ε_j`; either is acceptable — implementation picks one and
  records it.)
- Minimal candidate count **K=2** (`{z_p, z_a}`); extensible to K>2 later (more added columns).

**Why ρ_a is the scientific knob.** As ρ_a→1, `z_a`→`z_t`, so "MAR on z_a" → "missingness driven by a
near-copy of the censored value" → it approaches the H1 (MNAR) mechanism itself, and the richer null
can absorb almost everything (Bayes error → chance). The decision-relevant question P1R-A answers is
therefore precise: **how strong must an observed proxy (ρ_a) be before predictor-choice profiling
collapses the MNAR signal toward chance?** If only a near-perfect proxy collapses it, the P1 conclusion
is robust; if a modest proxy collapses it, the P1 ceiling was optimistic for real tables that contain
proxy columns.

## 3. H1 target column definition
`z_t` (the censored column). MNAR mechanism = the recorded P1 regime params (β₀, β₁=1 on `z_p`, β₂=δ on
own value `z_t`). The added `z_a` is a bystander under H1 (observed, correlated with `z_t`, not driving
missingness). β₀ is the P1-recorded value (rate-match unaffected by `z_a`).

## 4. H0 candidate predictor choices
Candidate set = `{z_p, z_a}` (all observed columns except the target). **The original predictor `z_p`
IS an allowed selection** — required, because the richer null's hypothesis set must ⊇ the restricted
null's, which is what guarantees `E_richer ≥ E_restricted` (harder-or-equal). Each candidate yields its
own (β₀′, β₁′).

## 5. β₀′ rate-matching and β₁′ profiling objective
- **β₀′ rate-match (per candidate j, per β₁′):** solve β₀′ so the population marginal missing rate
  `E_{z_j}[σ(β₀′ + β₁′·z_j)] = target_rate` (bisection on the monotone mean-sigmoid; `z_j` marginal is
  standard normal). Same method as P1, applied per candidate.
- **β₁′ profiling objective (state explicitly):** for each candidate j, select
  `β₁′*(j) = argmax_{β₁′} E_{H1}[ log L_{H0(j,β₁′)}(observed data) ]`
  = `argmin_{β₁′} KL( H1_observed ‖ H0(j,β₁′)_observed )` (expected observed-data negative
  log-likelihood of the MAR family under H1 data — the KL-projection of H1 onto the MAR-(j,β₁′)
  family). Then the **richer null selects the candidate**
  `j* = argmax_j  E_{H1}[ log L_{H0(j,β₁′*(j))} ]` (best-fitting predictor overall).
  Optimizer: grid + golden-section over β₁′ (reusing the P1 `profile_beta1` machinery), evaluated on a
  fixed H1 observed-data fit-sample; outer argmax over the K candidates. Deterministic given the RNG.
- The Bayes error is then computed (n-sample MC of the optimal LLR test) between H1 and `H0(j*, β₁′*)`,
  and separately between H1 and the restricted `H0(z_p, β₁′_restricted)` — both on the SAME 3-column
  observed data. The oracle LLR generalizes the P1 form: the missing-target term integrates `z_t` over
  the conditional `p(z_t | z_p, z_a)` (multivariate-Gaussian conditional); the observed/MAR terms use
  the chosen predictor. (The discriminator sees `z_p` and `z_a` always; `z_t` when observed.)

## 6. Expected runtime
CPU, no training. Per cell: K=2 candidate profiles (grid+golden, ~tens of likelihood evals on a fit
sample) + 2 Bayes-error MC estimates (richer + restricted). The 4 representative regimes × a small ρ_a
set (e.g. {0, 0.3, 0.6, 0.9, 0.95, 0.99}) ≈ 24 cells ⇒ **a few minutes**. Optional small (regime × ρ_a)
surface adds minutes. Target < 15 min CPU.

## 7. Tests (before any run)
- `MultivariateGaussianX`: conditional `p(z_t|z_p,z_a)` mean/variance match the closed-form Gaussian
  conditioning; sampling reproduces the specified correlation matrix; **non-PSD matrix raises**.
- Oracle reduces to the P1 2-column result when the candidate set is `{z_p}` and `z_a` is marginalized
  / absent (regression check against `profiled_oracle`).
- **Monotonicity:** on a constructed cell, `E_richer ≥ E_restricted − MC_tol`. A violation beyond
  tolerance fails the test (flags bug/MC noise) — directly encoding the user's rule.
- Predictor selection picks `z_a` when ρ_a > ρ_orig and it fits H1 better; picks `z_p` when ρ_a ≤ ρ_orig.
- Per-candidate β₀′ rate-match hits target rate; δ=0 ⟹ Bayes error ≈ 0.5 (within the single-fit floor).

## 8. Manifest fields
`arm="oracle"`, `kind="main"`, `checkpoint_loaded=false`, `all_layers_trainable=null`,
`trainable_param_count=null`, `xmodel` (MultivariateGaussianX descriptor + correlation matrix + PSD ok),
`grid` (regimes, ρ_orig per regime, ρ_a sweep, ρ_pa, candidate set, recipe), `beta0_solver`, per-cell:
selected predictor `j*`, selected β₁′ (richer & restricted), `E_richer`±SE, `E_restricted`±SE,
`KL_richer`, `KL_restricted`, `E_richer − E_restricted`, and `E_richer − P1_2col_ceiling` (context),
`wall_clock_seconds`, `metrics`. `validate_manifest` enforced; metrics not interpreted unless valid.

## 9. Success / failure interpretation
- **Monotonic (clean) comparison:** `E_richer ≥ E_restricted` on the same 3-column data, always.
  Report `Δabsorb = E_richer − E_restricted ≥ 0` per (regime, ρ_a). **If `Δabsorb < −2·SE`, treat as a
  bug or MC noise and investigate — never as "easier."**
- **P1 conclusion SURVIVES** if `E_richer` stays well below chance (signal not collapsed) for all but
  near-perfect proxies — i.e. predictor-choice profiling does not materially raise the ceiling except
  when an observed near-copy of the target exists (ρ_a → 1). Then the P1 headline stands, with the
  added, honest qualifier "absent a strong observed proxy for the censored item."
- **P1 conclusion AT RISK** if a *modest* ρ_a (say ≤ 0.6) pushes `E_richer` toward chance. That would
  mean the minimum-viable P1 ceiling was optimistic for realistic tables (which often contain
  correlated items), and the model-arm gaps should be re-read against the richer ceiling. This is the
  outcome the experiment exists to catch.
- **Context (not the monotonic claim):** report `E_restricted` vs the 2-column P1 ceiling to show how
  much the *added shadow column itself* shifted the baseline (the detection-side effect), kept separate
  from the null-enrichment effect.

## 10. Staging (per the approved plan)
- **P1R-A (this audit): predictor-choice profiling only.** Implement and run only this.
- **P1R-B (deferred): link-family profiling** (probit/cloglog added to the MAR null) — only if P1R-A
  does NOT collapse the signal AND the implementation is clean+tested. Not started before P1R-A review.
- **P1R-C (deferred): per-replicate / finite-sample GLRT-style profiling** — optional later; not before
  A/B review.

## Approval gate
On approval I will implement `lacuna/feasibility/xmodel_mv.py` (MultivariateGaussianX) + extend the
oracle/profiling to multi-predictor (new module, production generators and the Lacuna model untouched)
+ tests → suite green → run the CPU P1R-A representative-regime sweep → present `E_richer` vs
`E_restricted` (and vs the P1 ceiling) with the monotonicity check. Nothing runs until approved.
