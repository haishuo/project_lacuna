# G1 Pre-Registration — MAR-Imputation Counterfactual Channel (truth-level information test)

*Pre-registration — committed **before** the channel module is built or any example is generated
(PI go 2026-06-08: "Proceed with G1 exactly as written, preserving the gates. Treat it as evaluation
instrumentation, not architecture."). Implements gate **G1** of
`PROPOSAL-MAR-imputation-counterfactual-channel.md` (§5.A, §8). **No proxies, no runtime channels, no
deployment pathways** are designed or evaluated — G1 is solely about whether truth-level information
beyond φ exists. Thresholds below are locked and may not be revised after results exist.*

---

## 1. Question

> On semi-synthetic role-B cells, leave-domain-out, does the **paired-difference imputation channel**
> (truth-based, eval-only) (a) carry mechanism/δ information, (b) **add** information beyond the frozen
> φ/marginal baseline, and (c) separate the idioms — stably across ≥2 imputer classes?

## 2. Protocol (locked)

**Corpus & domains** (continuous bases, as in D3/E-study): **labor** = cps1985, cps1988, psid1976,
psid7682, workinghours (cont-only) · **nhanes** = rb_nhanes_weight/poverty/income (cont-only) · **hmda** ·
**wealth** = rb_scf2022_wealth_cont. All continuous target columns (card ≥ 30) per base.
**Splits:** block-aware **leave-one-domain-out** over the 4 domains; logistic-regression (LR-level, the
`transfer_features` gate discipline — no neural training); **pooled OOF** predictions across the 4 splits
→ one OOF AUC per task (precedent: the 0.65 no-training gate).

**Examples.** Per (dataset, target, idiom, δ): **24 examples**, max_rows 384, matched rate 0.3,
β₁ = 1.0, **δ ∈ {0.0, 2.5}** (the established binary slice), idioms {top_coding (τ_q=0.70), own_value}.
Deterministic per-cell RNG (fixed base seed, sorted enumeration). For **each example** a **paired
matched-rate MCAR mask** on the same column (same row subsample) is generated; every imputer is fit/
applied identically on both masks.

**Imputer classes (fixed configs, no tuning):**
1. `linear` — OLS conditional mean on observed rows; `F̂` = Gaussian(μ̂, σ̂_resid).
2. `mice_lite` — 2 rounds of chained linear imputation across predictors, then target regression;
   Gaussian `F̂`.
3. `rf` — RandomForestRegressor(n_estimators=50, fixed seed); `F̂` from per-tree spread + residual floor.
4. `gbm` — GradientBoostingRegressor(default depth, 100 estimators, fixed seed); Gaussian `F̂`.
5. `nn` — MLP(32, ReLU), 200 epochs, lr 1e-3, fixed seed; Gaussian `F̂`.

**Channel feature vector (per example, per imputer — 6 paired differences, mech − MCAR):**
`ΔB` (signed mean bias) · `ΔPIT_loc` (mean PIT − 0.5) · `ΔPIT_tail` (frac PIT > 0.9) ·
`ΔB_top` (signed bias over the top truth-quintile cells) · `ΔW1` (W₁ imputed-vs-truth within mask) ·
`Δcov80` (central-80% interval coverage). All values standardized within the observed column first.

**Baseline:** the frozen **17 marginal consequence features** (`compute_consequence_features`) on the
mechanism-masked example (φ-level marginal footprint; the established baseline).

## 3. Tasks & locked thresholds

LR (`sklearn LogisticRegression`, default C, standardized features), pooled OOF AUC over the 4
leave-one-domain-out splits:

| id | task | features | locked threshold |
|---|---|---|---|
| **G1-a (information)** | **own_value** δ=0 vs δ=2.5 | channel-only (6) | pooled OOF AUC ≥ **0.65** |
| **G1-b (increment)** | **own_value** δ=0 vs δ=2.5 | baseline+channel (23) vs baseline (17) | increment ≥ **+0.05** pooled OOF AUC |
| **G1-c (idiom separation)** | top_coding vs own_value at δ=2.5 | channel-only (6) | pooled OOF AUC ≥ **0.65** |

- Each criterion must hold for **≥ 2 of the 5 imputer classes**, and G1-a/b must hold for the **same**
  ≥2 classes (a channel that informs with one imputer and increments with a different one is not stable).
- **Primary task is own_value** because that is where φ is weak (the flat idiom) and where the channel's
  conditional content is non-redundant; top_coding versions of (a)/(b) are computed and reported as
  secondary (expected: channel informative, increment small because φ is already strong).
- **G1 PASSES iff G1-a ∧ G1-b ∧ G1-c** (per the proposal §8: separation **and** φ-increment).
- **G1 FAILS** otherwise — recorded outcomes: a-fail = no transferable truth-level signal (channel dead);
  b-fail with a-pass = channel duplicates the marginal footprint (φ-redundant — the §5.C concern);
  c-fail = no mechanism-family content (δ-sensitivity only).

## 4. Pre-registered predictions

P1. top_coding channel features will be strongly informative (localized bias/PIT spike) — but that is
    *not* the gate.
P2. own_value: `ΔB`/`ΔPIT_loc` negative-shifted at δ=2.5 (truth above MAR prediction); the open question
    is whether it **transfers** leave-domain-out at the 0.65 bar (its ancestor `transfer_features` failed
    at 0.523).
P3. Imputer stability: `linear`/`mice_lite` and `rf`/`gbm` behave as two within-pair-similar families.
P4. Wealth-domain rows will be the noisiest (heavy tails); the LODO split with wealth held out is the
    hardest.

## 5. Guardrails (binding)

Evaluation instrumentation only — **no architecture change, no model training (LR gate only), no
proxies, no runtime pathway, no natural-missingness labels, no SCF-imputed values as truth** (SCF cont
base values are retained-observed values; the punched-cell truth is ours). No tuning of imputers, LR, or
thresholds after results. Failures and ambiguities reported as-is; the gate verdict uses only the table
in §3.

---

*Committed before the channel module exists. The run script must consume these constants; the findings
document must score exactly these criteria.*
