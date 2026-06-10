# G1 — MAR-Imputation Counterfactual Channel: Findings (truth-level information test)

*Result of gate G1, executed exactly per the locked pre-registration
(`PREREGISTRATION-G1-imputation-channel.md`, commit `6f1bff9` — committed before any channel code or run).
Evaluation instrumentation only: LR-level, no model training, **no proxies, no runtime pathways, no
architecture** (PI constraint). Code: `lacuna/survey/imputation_channel.py` (+6 tests; 252 survey tests
green), `scripts/run_g1_imputation_channel.py`; artifact `runs/g1_imputation_channel.json` (3,840 example
rows: 40 targets × 2 idioms × δ∈{0, 2.5} × 24 examples, paired matched-rate MCAR mask each).*

---

## 0. Verdict

> **G1 PASSES — decisively.** All three locked criteria pass for **all five** imputer classes (bar: the
> same ≥2 for a/b). Truth-level information beyond φ exists, transfers leave-domain-out, and separates
> the idioms. **The channel is justified as EVAL-ONLY instrumentation.** Per the PI constraint and the
> proposal's own gates, **nothing about runtime follows from this** — G2 (proxy transfer) remains
> untested and its prior remains pessimistic.

## 1. Results against the locked criteria

Baseline (frozen 17 marginal consequence features): own_value pooled OOF AUC **0.574** (the flat idiom,
as established), top_coding **0.633**.

| imputer | **G1-a** own_value channel-only (≥0.65) | **G1-b** increment over φ (≥+0.05) | **G1-c** idiom sep (≥0.65) | tc channel | tc increment |
|---|---|---|---|---|---|
| linear | 0.900 | +0.302 | 0.681 | 0.885 | +0.292 |
| mice_lite | 0.900 | +0.302 | 0.681 | 0.885 | +0.292 |
| rf | 0.989 | +0.421 | **0.834** | 0.947 | +0.289 |
| gbm | **0.993** | **+0.424** | 0.824 | 0.943 | +0.306 |
| nn | 0.994 | +0.379 | 0.799 | 0.883 | +0.262 |

**Per-held-out-domain detail (own_value, channel-only)** — the transfer content of the result:

| held-out | base17 | linear | rf | gbm | nn |
|---|---|---|---|---|---|
| hmda | 0.685 | 1.000 | 1.000 | 1.000 | 1.000 |
| labor | 0.588 | 0.986 | 0.999 | 0.999 | 0.995 |
| nhanes | 0.615 | 0.999 | 1.000 | 1.000 | 0.994 |
| **wealth** | 0.652 | **0.596** | **0.977** | 0.979 | 0.985 |

**Pre-registered predictions:** P1 ✓ (top_coding strongly informative); P2 ✓ (signed directions exactly
as predicted: `d_B` −0.19 → **−1.99** from δ=0 to δ=2.5 for own_value, `d_PIT_loc/tail` positive and
growing); P3 partially ✓ (linear ≡ mice_lite — see §3; rf/gbm pair as expected); P4 ✓ (wealth is the
hardest split — and it **breaks only the linear family** (0.596), while flexible imputers hold 0.98).

## 2. What the result means (and does not mean)

1. **The own-value "flat" idiom is flat at the *marginal-footprint* level, not at the truth-conditional
   level.** With truth access but **no mechanism knowledge**, a generic MAR imputer's paired failure
   signature recovers own-value δ at 0.90–0.99 OOF — across domains, including the regime-isolated
   wealth domain. The selection shift is *statistically* large and estimator-accessible; what deployment
   lacks is **truth, i.e., identification — not signal**. This sharpens the lab-coat statement: the
   information needed to govern sensitive-item sensitivity analysis *exists in the counterfactual*, and
   is missing at runtime for identification reasons only.
2. **The three-levels hierarchy gains a quantified middle rung** (binary δ0-vs-2.5, this corpus): oracle
   ≈ 1.0 (knows mechanism) → **imputation counterfactual 0.90–0.99 (knows truth values only)** →
   marginal-feature/φ 0.57–0.63 → learned transferable channel ≈ base-rate (P2.2c). Almost the entire
   oracle-to-channel gap survives the removal of mechanism knowledge and collapses only when **truth**
   is removed — locating the bottleneck squarely at identification, not estimator class.
3. **The increment over φ is large precisely where φ is weak** (own_value +0.30 to +0.42) — the channel
   is *not* a re-parameterization of the marginal footprint (the §5.C redundancy concern is answered).
4. **What it does NOT mean:** no runtime signal exists until **G2** shows a truth-free proxy tracks
   these quantities OOF — prior pessimistic (`transfer_features` 0.523; the E-study divergence). The
   truth-level strength was *partially expected* (the proposal's honest prior called it
   near-tautological at the truth level); the gate's real content — **leave-domain-out transfer incl.
   wealth, and the φ-increment** — is what passed decisively.

## 3. Honest caveats

- **`mice_lite` ≡ `linear` exactly:** with complete predictors (role-B), the chained pass is a no-op
  smoothing, so the class effectively contains **4 distinct** imputers, not 5. Stability still far
  exceeds the ≥2 bar; recorded for accuracy.
- **Linear-family fragility on heavy tails:** the wealth split drops the Gaussian-residual linear imputer
  to 0.596 — the channel's robustness on regime-outlier domains comes from the flexible imputers
  (rf/gbm/nn). Any future use should report per-imputer, never pooled-only.
- **δ=0 reference is MAR(β₁), not MCAR**, so the paired difference at δ=0 is small-but-nonzero
  (`d_B` ≈ −0.19; imperfect conditional fit). The LR uses this null honestly; it is not contamination.
- Binary δ slice (0 vs 2.5) — the extreme; intermediate-δ resolution untested at G1 (not in the gate).
- 24 examples/cell; pooled OOF per the precedent (no per-seed SEs at the LR level — splits shown
  instead).

## 4. Status & what would come next (PI decisions; nothing started)

- **Channel status: EVAL-ONLY instrumentation, earned via G1.** Usable (when the PI directs) for:
  enriching the lab-coat-fraction decomposition (the identification-vs-signal split in §2.1), eval-time
  mechanism/δ instrumentation over `P_prior`, and as the truth-side anchor that any future **G2** proxy
  study would be validated against.
- **G2 (proxy transfer)** is the next gate *if* runtime relevance is ever pursued — explicitly **not
  designed here** (PI constraint). **G3** (runtime admission review) remains behind G2.
- No architecture, model, or manifest change accompanies this result; the channel lives in
  `lacuna/survey/imputation_channel.py` + the G1 script as instrumentation.

---

*Executed exactly per the locked pre-registration; no thresholds, features, imputers, datasets, or
criteria altered after results. Stopped for PI review.*
