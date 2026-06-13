# P2 — Lacuna-Survey δ-Prior Re-architecture — Implementation Audit / Spec

*Branch `p2/delta-prior-rearchitecture` (off the P1R tip; carries the validated backbone, the
`lacuna/feasibility/` package, and the P1R governance result). Governed by `docs/NORTH-STAR.md` and
`docs/proposals/PROPOSAL-survey-rewrite.md`. **SPEC ONLY — no code, no training, no oracle runs until approved.***

## 0. What P2 is (and is not)

**Is:** Lacuna-Survey as a **governance layer for missing-data assumptions** — a calibrated, inspectable
**prior over the sensitivity parameter δ** per (suspected-MNAR) column, with **first-class
uncertainty/abstention** (including **proxy-aware** abstention keyed on `Var(z_t|observed)`), feeding a
**deterministic estimand-side sensitivity report** (δ-prior × estimand → tipping-point curve).

**Is not:** a mechanism *measurement* (Molenberghs — charter §2), a 3-class MCAR/MAR/MNAR classifier, or
the temporary binary H0-vs-H1 feasibility probe. P2 replaces the probe head with the real output system.

**Charter compliance check (must hold throughout):** never claim to beat non-identifiability (the δ-prior
is a *prior*, reported with uncertainty); calibration over accuracy (§4.2); abstain, don't confabulate
(§4.3); semi-synthetic is the only ground truth (§4.4); leakage guard + matched rate (§4.5, §4.7); run
the REAL model, retrained from scratch (§4.6, §4.9); human-parity — give the regime and the estimand,
don't infer them (§4.8); estimand enters at the reporting layer (§3¾).

---

## 1. Backbone reuse (what P1 justifies)

| v1.0 `LacunaModel` component | P2 disposition | Justification |
|---|---|---|
| `LacunaEncoder` (set-transformer over (value, observed, mask-type, feature-id) tokens) | **KEEP architecture; retrain from scratch** | Objective-agnostic representation (codebase audit §2.4). The P1 binary probe used exactly this encoder, trained from scratch, and reached the profiled Bayes ceiling in moderate/strong/boundary → it has the capacity to extract the δ-relevant signal. v1.0 (val 0.9163 at d≤48) shows it scales to real survey widths. |
| Tokenization / batching, catalog / ingestion, normalization, RNG | **KEEP** | Objective-agnostic data infra (audit §2.1–2.2). |
| Generator framework (`apply_to` contract, registry) | **REUSE + REPARAMETERIZE** to (rate, δ) on survey idioms; **start with own-value self-censoring** (P1-validated). The `lacuna/feasibility` δ-generator + β₀ rate-solver are the seed. | P1 validated this exact axis. |
| MoE gating MLP + temperature scaling | **KEEP as plumbing / calibration seam** | Reusable; temperature scaling is the calibration hook. |
| Missingness feature extractor (footprint stats) | **KEEP**; also feeds the OOD/manifold + proxy signals | Label-free observable statistics (audit §2.2). |
| Reconstruction heads (MCAR/MAR/MNAR world-models) | **DROP** (or, optional, repurpose as residual features) | Exist only for the dead 3-class gate. |
| 3-class head `get_class_posterior` + `BayesOptimalDecision` (loss matrix) | **REMOVE/REPLACE** | The dead objective (charter §10 forbids reuse as main). |
| Binary feasibility head (`model_arm`) | **REMOVE** | Temporary probe. |
| **NEW:** δ-prior head, abstention/OOD head, proxy-score, estimand reporting layer | **BUILD** | §2, §5–7. |

**Seam:** cut after `evidence = encoder(...)` (the clean seam from the codebase audit). Keep encoder +
missingness features; replace everything downstream.

## 2. δ-prior output (the target)

- **Output = a calibrated distribution over ORDERED δ bins** (per user preference; not raw scalar
  regression). Own-value self-censoring axis: δ = β₂ ≥ 0, MAR ⇔ δ=0.
- **Bins (P2.0 decision):** ~7 ordered bins with **δ=0 (MAR) as bin 0**, e.g. {0} | (0, 0.25] | (0.25,
  0.5] | (0.5, 1.0] | (1.0, 1.5] | (1.5, 2.0] | (2.0, ∞). Edges interpretable; finalize against the δ
  grid the generator samples.
- **Derived, reported quantities:** `E[δ]` (bin-center expectation), **entropy** (uncertainty),
  **credible/plausible range** (central interval covering a stated mass, e.g. 80%), and `P(δ=0)` (the
  "is this plausibly MAR?" mass).
- **Granularity:** v1 = per-**(dataset with one censored target column)** δ-prior (the P1 setup, on REAL
  survey X with the other columns observed). Extend to **per-column** on fully-missing tables in a later
  P2 stage. (Flag.)

## 3. Training target (semi-synthetic survey data only)

- **Real survey X** (the v1.0 survey pool / catalog) + **synthetic δ-parameterized missingness** on a
  chosen target column, using the z-scored predictor view (reuse `semisynthetic._zscore_columns` +
  the feasibility δ-generator generalized to real multi-column X).
- **Known δ** from the generator = the supervised label (answer sheet). Sample δ from the δ-grid
  (including δ=0 / MAR), β₁ (observed coupling) as a swept nuisance, predictor column chosen/recorded.
- **Matched missingness rate** via the β₀ solve (defuse the rate confound — charter §4.7).
- **Answer sheet saved per example:** δ (and bin), β₀/β₁, target & predictor column ids, realized rate,
  dataset name, generator-family id, seed — for auditability and the eval (proxy/abstention tests).
- **Start with own-value self-censoring** (P1-validated) before expanding idioms (skip logic, etc.).

## 4. Loss and calibration

- **Primary loss = a proper scoring rule for ORDERED bins:** Ranked Probability Score (RPS, the discrete
  CRPS over the δ-bin CDF) — proper and order-aware — or ordinal log-score. **NOT** the old 3-class
  cross-entropy (charter §10).
- **Calibration:** post-hoc temperature scaling (reuse v1.0 `calibration.py` scaffolding) on the bin
  logits; report **before/after**.
- **Tracked metrics (calibration-first, §4.2):** reliability diagrams, **ECE** (bin probabilities),
  **interval coverage** (does the X% credible interval contain true δ X% of the time?), **δ-bin
  accuracy + E[δ] error / RPS**. Accuracy is secondary to calibration.

## 5. Abstention / uncertainty (first-class)

**Define the abstention criterion BEFORE implementation.** Lacuna abstains (or widens the prior) when ANY
trigger fires; the output records the abstain flag **and the reason**:

1. **High predictive uncertainty** — δ-distribution entropy > τ_H (near-chance / unidentifiable
   footprint; the profiled-oracle near-chance regions manifest as high entropy here).
2. **Near-proxy absorption (P1R-C)** — proxy-score (operational `Var(z_t|observed)`, §6) < τ_proxy ⇒ the
   censored column is nearly reconstructible from observed columns ⇒ MNAR≈MAR-on-proxy ⇒ do not make a
   confident δ claim.
3. **Out-of-manifold footprint** — OOD-score (distance of the dataset's missingness footprint from the
   training-footprint manifold) > τ_OOD ⇒ outside the trained span (charter §6 coverage boundary).

Thresholds (τ_H, τ_proxy, τ_OOD) are **calibrated on held-out data** to a target abstention–accuracy
operating point; recorded in the manifest. Abstention is evaluated as a first-class metric (§8), not a
side effect. **A confident δ claim in an abstain-region is the worst output (charter §4.3).**

## 6. Proxy-aware governance (P1R-C made operational)

- **Operational proxy-score for `Var(z_t|observed)`:** regress the **observed** values of the target
  column on the other observed columns ⇒ R²; proxy-score ≈ `1 − R²` (standardized) — small ⇒ strong
  proxy. Computable from observed data alone.
- **Caveat (recorded):** under MNAR the observed target values are truncated, so R² estimated on observed
  data is biased; the proxy-score is therefore a *conservative operational signal*, not the true
  `Var(z_t|z_t-unobserved)`. v1 uses it as an abstention trigger (§5.2) with this caveat documented;
  a debiased estimator is a later refinement.
- **Behavior:** when proxy-score is small (near-reconstructible), Lacuna **widens the prior toward δ=0 /
  abstains** rather than asserting MNAR — directly implementing the P1R-C finding that a strong observed
  proxy makes the MNAR claim unreliable.

## 7. Estimand layer (separation confirmed)

- The **network learns the δ-prior only** — estimand-free (charter §3¾). It never sees the downstream
  estimand.
- A **thin, deterministic reporting layer** takes (δ-prior, user-stated estimand) → **tipping-point
  curve / sensitivity sweep** via δ-adjustment / pattern-mixture (B2: canonical mean-shift first). When
  the δ-prior is an abstain, the report says "insufficient evidence for a δ-specific claim; here is the
  full uninformative sweep."
- **Confirmed:** no estimand inside the network; estimand enters only at the reporting layer.

## 8. Evaluation (the RIGHT metrics — charter §5)

- **Out-of-mechanism-FAMILY δ-prior calibration** (train on some families, test calibration on unseen) —
  the headline number.
- **Leave-one-FORM-out** generalization (train on link forms, hold one out).
- **OOD / manifold detection** (flags family-novel footprints?).
- **Abstention quality:** abstention–accuracy/calibration curve; does abstaining improve calibration on
  the kept set; does it abstain in the right places (near-chance, near-proxy, OOD)?
- **Proxy-awareness tests using `Var(z_t|observed)`:** construct held-out cells sweeping Var (à la
  P1R-C); confirm the model widens/abstains as Var→0 and is confident when Var is large.
- **Null δ=0 behavior:** MAR data ⇒ mass on the δ=0 bin / wide prior, not a confident MNAR claim.
- **Matched-rate leakage controls:** rates matched (no rate cue); a δ=0-vs-δ=0 null sanity ⇒ chance-level
  discrimination; no negative-gap-style anomalies.

## 9. Manifest & audit (every P2 run)

Extend `lacuna/feasibility/manifest.py` (or a P2 manifest) to require: generator family + δ grid/bins +
rate-matching method, model architecture + trainable_param_count, **checkpoint_loaded (must be False for
main)**, all_layers_trainable (True for main), loss spec, calibration metrics (ECE/coverage/RPS),
abstention metrics + thresholds, proxy/OOD config, split scheme, wall-clock, and **kind ∈
{main, ablation, smoke}**. **No metric is interpreted unless the manifest validates** (fail-loud).

## 10. Forbidden shortcuts (hard constraints)

No RF/MLP-on-handcrafted-stats as the main model; no frozen-encoder + stapled-δ-head as a main run (full
retrain only); no old v1.0 generator path; no metric without manifest validation; **no collapse back to
a binary/3-class MAR-vs-MNAR objective as the main P2 objective.** (Ablations/ smokes may use small
configs but must be labeled `kind` accordingly and never reported as the main result.)

---

## 11. Proposed module plan (new code; production v1.0 model untouched until built)

- `lacuna/survey/delta_generator.py` — δ-parameterized self-censoring on **real** survey X (generalizes
  `feasibility.delta_generator`; (target, predictor, β₁, δ, rate); answer sheet).
- `lacuna/survey/delta_head.py` — ordered-δ-bin head on `evidence` (+ per-column pooling for later).
- `lacuna/survey/abstention.py` — uncertainty + proxy-score + OOD triggers; abstain decision.
- `lacuna/survey/proxy_score.py` — operational `Var(z_t|observed)` estimator (§6).
- `lacuna/survey/manifold_ood.py` — footprint-manifold density / distance (reuses `missingness_features`).
- `lacuna/survey/loss.py` — RPS / ordinal proper scoring + calibration metrics.
- `lacuna/survey/train.py` — from-scratch training loop (full LacunaModel encoder + δ head); no checkpoint.
- `lacuna/survey/eval.py` — the §8 harness (out-of-family calibration, LOFO, OOD, abstention, proxy,
  null, leakage).
- `lacuna/survey/report.py` — deterministic estimand × δ-prior → tipping-point (B2).
- Tests for every module (CLAUDE.md Rule 7); all files ≤ 500 LOC (Rule 4); determinism via injected RNG
  (Rule 6); fail-loud at boundaries (Rule 1).

## 12. Phasing (each phase: audit-confirmed scope, suite green, manifest-valid; review between)

- **P2.0 — paper:** finalize δ-bins, the proxy-score definition, the abstention thresholds' calibration
  protocol, and the answer-sheet schema. (No code.)
- **P2.1 — data:** δ-generator on real survey X + answer sheet + matched rate + tests.
- **P2.2 — model + loss:** δ-bin head on the (from-scratch) encoder; RPS loss; minimal training loop;
  calibration tracking. First out-of-family calibration number.
- **P2.3 — abstention + governance:** proxy-score, OOD/manifold, abstention gate; proxy-awareness tests.
- **P2.4 — eval harness:** the full §8 suite.
- **P2.5 — reporting layer:** δ-prior × estimand → tipping-point (B2).

## 13. Non-goals (this pass)

Multi-axis/per-idiom δ beyond own-value self-censoring (skip logic, attrition, unit-nonresponse — later,
per PROPOSAL §5.1 revision trigger); B1 rich-mechanism posterior (B2 first); per-cell multi-column
missingness (single censored target first); real-time/online use; non-survey regimes.

## 14. Open decisions for review (before P2.1)

1. δ-bin edges + count (§2) — propose 7 bins as above; confirm or adjust against the δ-grid.
2. Output granularity v1: single-censored-target per dataset (start) vs per-column now — propose start.
3. Proxy-score estimator: observed-R² (simple, biased-conservative) vs a debiased variant — propose
   observed-R² v1 with the caveat.
4. Loss: RPS (propose) vs ordinal log-score vs a hybrid.
5. Abstention thresholds: a single operating point vs a reported abstention–calibration curve — propose
   the curve, pick an operating point from it.

## Approval gate
This is the spec. On approval I will start at **P2.0/P2.1** (paper decisions + the δ-generator on real
survey X with tests), get the suite green, and report — then proceed phase-by-phase with review between.
No code until approved.
