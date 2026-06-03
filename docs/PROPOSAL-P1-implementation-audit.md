# P1 Feasibility — Implementation Audit (for approval BEFORE any coding)

*Governed by `docs/NORTH-STAR.md` (esp. §4.9 no-weak-proxy-kills, §2 identification line) and
`docs/PROPOSAL-survey-rewrite.md` (§5.1 DECISION, §6 P1). This document specifies exactly what the P1
feasibility experiment will do. **Status: SPEC ONLY. No code exists yet. No training will run until
this audit is approved.***

## Normative vocabulary (used precisely throughout)

- **Ceiling** — reserved *exclusively* for the analytic Bayes oracle (§6 below), and only under its
  stated X-model assumption. Nothing else may be called a ceiling.
- **Baseline** — RF / MLP / handcrafted-footprint-feature models. May contextualize a positive result;
  a baseline *negative is never evidence the problem is impossible* (§4.9).
- **Transfer ablation** — any frozen-encoder / stapled-head / checkpoint-loaded run (e.g. the optional
  v1.0 control). Not Lacuna, not a ceiling, not the feasibility estimator.
- **Deployment-strength estimator** — the real `LacunaModel` architecture, retrained end-to-end from
  scratch. The only learned instrument whose *negative* counts (§4.9).

## What this experiment IS and IS NOT

IS: a controlled identifiability/feasibility probe on **one** mechanism axis (own-value
self-censoring), asking — at matched missingness rate — *how much observed-data signal δ produces
(the ceiling), and how much of it the deployment-strength model recovers (the achieved).*
IS NOT: the Lacuna rewrite, a multi-idiom manifold, an accuracy claim on real data, or anything that
touches production generators/model code. Single censored column per dataset. Scalar δ only.

---

## 1. Generator path

New, isolated, **tested** package — production generators are NOT modified (that is P2). Reuses only
the existing `_zscore_columns` (predictor view) and the catalog X-loaders.

```
lacuna/feasibility/
  __init__.py
  delta_generator.py   # δ self-censoring generator + β₀ rate-matching solver   (<300 LOC)
  xmodel.py            # XModel ABC + ConditionalGaussian + GaussianCopula        (<300 LOC)
  oracle.py            # analytic Bayes-oracle (LLR, KL/Chernoff, n-sample error) (<350 LOC)
  sweep.py             # grid orchestration + manifest writing                    (<350 LOC)
tests/unit/feasibility/
  test_delta_generator.py   # rate-matching hits target; δ=0 ⟹ own-value-independent mask
  test_xmodel.py            # conditional Gaussian matches closed form; copula round-trips
  test_oracle.py            # LLR=0 when H0==H1; Bayes-error→0.5 as δ→0; monotone in δ; MC≈Chernoff
scripts/
  run_feasibility_oracle.py # oracle arm (CPU)
  run_feasibility_model.py  # model arm (GPU, from scratch)
```
Determinism: every stochastic step takes an injected `RNGState` (Rule 6). The oracle is analytic
(no RNG except the integration MC, which is seeded and is *numerical integration of a known integral,
not estimation* — see §6).

## 2. δ generator formula

On real survey X, with `Z = _zscore_columns(X_complete)`; choose target column `t`, predictor `p≠t`.
**Only column `t` is censored** (single-column probe); all other columns fully observed. Observed
values are handed downstream on the original scale (`X_complete * R`), per v1.0.

```
η_i      = β₀ + β₁·Z[i,p] + β₂·Z[i,t]
P(R[i,t]=0 | Z) = σ(η_i)                 # probability column t is MISSING in row i
R[i,t] ~ Bernoulli(1 − σ(η_i))           # 1 = observed
```
- **δ ≡ β₂** (dependence on the column's *own*, about-to-be-hidden value).
- **MAR ⟺ β₂ = 0** (missingness depends only on the *observed* predictor Z[:,p]).
- MCAR ⟺ β₁ = β₂ = 0;  MNAR ⟺ β₂ ≠ 0.
- β₁ = observed-coupling **nuisance** (swept, not the target of inference).

## 3. β₀ rate-matching solver

Each hypothesis is **independently** matched to the same target marginal rate `r*`, so the two are
indistinguishable by rate alone (defuses the confound, charter §4.7).
```
realized_rate(β₀) = mean_i σ(β₀ + β₁·Z[i,p] + β₂·Z[i,t])     # monotone ↑ in β₀, deterministic
solve  β₀  s.t.  realized_rate(β₀) = r*   via Brent on [−50, 50]   (scipy.optimize.brentq)
```
Matches the **expected** rate exactly; the sampled realized rate fluctuates ~Binomial and is recorded.
Fails loud (Rule 1) if brentq does not bracket (should never, since mean σ spans (0,1)).

## 4. β₁ / δ / rate sweep grid

| axis | symbol | values | role |
|---|---|---|---|
| departure | δ ≡ β₂ | {0, 0.25, 0.5, 1.0, 1.5, 2.0} | 0 = MAR anchor → strong MNAR |
| observed coupling | β₁ | {0.0, 0.5, 1.0, 2.0} | nuisance (β₁=0 makes δ=0 a pure-MCAR null) |
| matched rate | r* | {0.1, 0.2, 0.3, 0.4} | the known confound — swept, never free |
| predictor–target corr | ρ | {0.0, 0.3, 0.6, 0.9} | **the Molenberghs knob** (synthetic-X only) |
| sample size | n | {128, 512, 2048} | how distinguishability scales with evidence |
| seeds (sampling) | — | 5 fixed seeds | sampling variance (model arm + realized-rate) |

Synthetic-X oracle sweeps the full ρ axis. Real-X oracle and the model arm use the **survey pool**
(10 train / 2 val survey datasets); target/predictor columns chosen by a fixed, recorded rule
(highest-variance numeric columns; ρ measured, not set). δ=0-vs-δ\* comparisons are always at equal
(β₁, r*, dataset/ρ, n).

## 5. X-model interface (assumption made explicit, never hidden)

```
class XModel(ABC):
    def fit(self, Z) -> "XModel"                      # no-op for the exact synthetic model
    def conditional_logpdf(self, z_t, z_p) -> Tensor  # log p(z_t | z_p)   (oracle needs this)
    def conditional_moments(self, z_p) -> (mean, var) # for Gauss–Hermite nodes
    def sample_conditional(self, z_p, rng) -> z_t      # for the n-sample Bayes-error MC
```
- **Synthetic-X (exact ceiling):** `ConditionalGaussian` with the *true* generating (ρ) — no fit, no
  assumption. This is the clean mathematical ceiling for the mechanism family.
- **Real-survey-X (model-dependent):** `ConditionalGaussian` **fitted** (mean/cov → closed-form
  conditional) for v1, with `GaussianCopula` as the pluggable second option. The fitted X-model is the
  *only* assumption in the real-X oracle and is reported in the manifest. Interface is pluggable so a
  nonparametric/learned conditional can be swapped in later. **The X-model assumption is always
  reported alongside any real-X ceiling; an oracle is only a "ceiling" under its stated X-model.**

## 6. Oracle calculation (the only "ceiling")

Per-row observed-data log-likelihood-ratio between H1 (δ=δ\*) and H0 (δ=0), each independently
rate-matched. The predictor marginal and `p(z_t|z_p)` cancel in the *observed-target* term; the
X-model enters the *missing-target* term and the sampling law:
```
target observed (z_t seen):   LLR_i = log(1−σ(η₁ᵢ)) − log(1−σ(η₀ᵢ))
target missing:               LLR_i = log m₁(z_p) − log m₀(z_p),
                              mₕ(z_p) = ∫ p(z_t|z_p)·σ(β₀ʰ+β₁z_p+β₂ʰ z_t) dz_t   (Gauss–Hermite)
```
Outputs per grid cell:
- **Per-row information:** KL(H1‖H0), KL(H0‖H1), Chernoff information C.
- **n-sample Bayes error** of the optimal LLR test (sign of Σᵢ LLRᵢ, equal priors) — computed by
  drawing rows from the X-model under each hypothesis and evaluating the *known* LLR. This is
  **numerical evaluation of a known integral, not learning**; by Neyman–Pearson it is the optimal test,
  hence the ceiling. Cross-checked against the asymptotic ½·exp(−nC).
- The decisive object: the **Bayes-error( δ, β₁, ρ, r*, n ) surface**, especially (i) does it fall
  below 0.5 as δ grows at matched rate, and (ii) how it degrades as ρ→1 (the Molenberghs collapse).
**A negative here (Bayes error ≈ 0.5 across the realistic δ range at matched rate) is an unconfounded,
assumption-stated kill.**

## 7. Model target / loss (the deployment-strength arm)

Primary task = **binary discrimination** δ=0 vs δ=δ\*, so the model's ROC is directly comparable to
the oracle's on the *same* data.
- Input: probe-generated `(X_observed, R)` tokenized by the existing tokenizer; label = δ-bucket.
- Model: the **current `LacunaModel` architecture**, instantiated fresh, with the 3-class head
  **removed** and replaced by a **single-logit binary head** on the dataset-level evidence. Re-objectived,
  not adapted.
- Loss: binary log-loss (a proper scoring rule); secondary optional δ-regression with NLL/CRPS as a
  B2 preview (reported separately, not the gate).
- Eval: **out-of-family / held-out** (held-out survey datasets and held-out target/predictor columns,
  and a held-out link form for the §6c LOFO step), reporting achieved AUC / log-loss and the
  **gap to the oracle ceiling** per cell. Achieved-below-ceiling localizes "signal left on the table";
  achieved≈ceiling-≈chance confirms the kill is fundamental.
- Optional **baselines** (clearly labeled, never a ceiling): a footprint-feature RF/logistic for
  context. Optional **transfer ablation** (clearly labeled): the v1.0 control under `--v1-control`.

## 8. Are all layers trainable?

**YES.** Every parameter of the freshly-instantiated `LacunaModel` (encoder + backbone + new binary
head) has `requires_grad=True`. Nothing is frozen. A frozen-encoder run is permissible ONLY as an
explicitly-labeled transfer ablation, and never as the feasibility result (§4.9, §5.2).

## 9. Is any checkpoint loaded?

**NO.** Random initialization, end-to-end from scratch. No v1.0 checkpoint, no pretrained encoder, no
old generator path. The **only** exception is an explicit `--v1-control` run, which loads v1.0 and is
recorded as a *transfer ablation* in the manifest — not the feasibility estimator, not a ceiling.

## 10. Expected runtime

- **Oracle arm (CPU):** analytic + quadrature + seeded integration MC. Full grid (≈ 6 δ × 4 β₁ × 4 ρ ×
  4 r* × 3 n for synthetic; survey-pool × 6 δ × 4 β₁ × 4 r* × 3 n for real) ≈ **< 10 min, CPU**.
- **Model arm (GPU):** does NOT train one model per grid cell. Trains **N_models ≈ 5–8** deployment-
  strength models (held-out folds × a few representative (β₁, regime) settings), each a from-scratch
  full-MoE fit ~15–30 min (early-stopped, like the v1.0 repro's ~10 min) → **≈ 1.5–4 h GPU total**.
  The exact N_models and fold scheme are listed in the run manifest and are part of what you approve.

## 11. Saved artifacts & manifest fields

Output root: `/mnt/artifacts/project_lacuna/feasibility/<run_id>/`
- `manifest.json` — `run_id`, `git_commit`, `charter_commit`, `proposal_commit`, `timestamp` (injected
  clock), `arm` (`oracle`|`model`), `task` (`binary`|`delta_regression`), full grid spec (§4),
  `xmodel` (`type`, fitted params or `exact_synthetic`), `target_predictor_rule`, `seeds`,
  **`checkpoint_loaded: false`**, **`all_layers_trainable: true`**, `is_transfer_ablation: false`,
  `is_baseline: false`, code/file hashes.
- `oracle_results.parquet` — per cell: KL both directions, Chernoff C, n-sample Bayes error
  (synthetic-exact and real-fitted), realized rate, ρ.
- `model_results.parquet` — per run/cell: config, achieved AUC / log-loss / Bayes-error, **gap-to-
  ceiling**, train/val curve refs; checkpoints + `metrics.jsonl` via existing infra.
- `oracle_vs_model.csv` — the bracket table (ceiling vs achieved per cell).
- `baselines.parquet` (if run) and `transfer_ablation.parquet` (if `--v1-control`), each tagged so they
  can never be mistaken for the ceiling or the feasibility estimator.
- `figures/` — Bayes-error surfaces over (δ, r*, ρ); oracle-vs-model overlay; ρ-collapse curves.
- `report.md` — the pass/kill read against the gate, in the normative vocabulary above.

---

## Approval gate

I will not write any of the code in §1, and will not run the oracle or any training, until this audit
is approved. On approval I propose to build and run in this order, pausing for your review between:
**(i)** `lacuna/feasibility/` + its tests (green suite) → **(ii)** oracle arm (CPU, no training) →
**(iii)** present the oracle ceiling surface → **(iv)** only then the model arm (from scratch, GPU).
If the oracle (ii–iii) returns a fundamental kill, we stop there and report — no model arm needed.
