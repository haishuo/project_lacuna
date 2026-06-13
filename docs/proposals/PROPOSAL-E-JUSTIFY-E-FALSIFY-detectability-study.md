# E-JUSTIFY / E-FALSIFY — Pre-Registered Detectability Study (specification)

*Specification & pre-registration only — **no training, no probing, no runs, no architecture change**
until PI approves this document (binding; PI 2026-06-07). Implements §7–§8 of
`PROPOSAL-detectability-analysis.md`. Governed by `NORTH-STAR.md` and
`MASTER-lacuna-survey-architecture.md`. This is an **architectural review** of the Stage-2 detectability
head (`PROPOSAL-Level1-design-spec.md` §3), not a metric-improvement exercise.*

---

## 0. Framing — null vs challenger

| role | candidate | composition |
|---|---|---|
| **H₀ (null) — the derived stack** | coverage gate + posterior info-gain | (B) coverage-distance (D2 metric, model-independent) gates; (A-estimate) `info_gain = KL(posterior ‖ prior_marginal)` from the **existing** δ-prior head's calibrated output — **zero new learned parameters** |
| **H_A (challenger) — the learned head** | representation→detectability probe | a small regressor from the frozen model's pooled representation to the oracle informativeness target — evaluated as a **post-hoc probe** (the §8-representation-probe precedent), never wired into the model |

**Burden of proof is on the challenger.** The Stage-2 head is **struck from the design** unless H_A meets
the pre-registered justification criteria (§6). A tie or an ambiguous result defaults to H₀ (the
no-new-parameters position). The off-manifold **gate role is not adjudicated here — it is already closed
to learned components** (self-reference argument, detectability analysis §4); this study adjudicates only
the **in-manifold informativeness estimator**.

## 1. The three questions, mapped to measurements

1. **In-manifold validity of the derived estimator:** does posterior info-gain agree with **oracle
   informativeness** out-of-family, and reproduce the known idiom spectrum (top-coding ≫ own-value) at
   matched coverage? → §5 F1–F2.
2. **Gate validity:** does coverage-distance, thresholded by a rule locked before training, flag the
   D3 failure regime (wealth-like) while passing covered domains — and does the D3 calibration pathology
   concentrate in the gated-out region? → §5 F3–F4.
3. **Challenger value-add:** does the probe beat the derived estimator **leave-domain-out** by the locked
   margin, where it matters (idiom separation, oracle agreement)? → §6 J1–J3.

## 2. Study design

**Models (the "subjects").** Level-1 φ-spine models trained with the **frozen D3 configuration**
(`Level1Config` exactly as `run_curve_tightened._cfg`; binary δ0-vs-δ2.5 scheme; no tuning), 8 seeds,
**leave-domain-out**: for each held-out domain D ∈ {NHANES-cont, HMDA}, train on the moderate-continuous
pool minus D (labor ± nhanes as in D3's matched arms). Wealth is never trained on — it is the
**uncovered control**. Reuses the D3 training recipe verbatim; no new model variants.

**Cells (the unit of oracle comparison).** A cell = (idiom, dataset, target column, δ, rate 0.3 fixed):
- idiom ∈ {**top_coding**, **own_value_self_censoring**} — both, because idiom separation is the question;
- datasets = the held-out domains' continuous bases (rb_nhanes_weight/poverty/income, survey_hmda,
  rb_scf2022_wealth_cont) + 2 in-family training datasets as sanity anchors;
- δ ∈ {0.0, 0.75, 1.25, 2.5} (subset of the canonical grid; includes the trained binary endpoints and two
  interior points to test informativeness off the trained grid);
- ≥ 200 examples per cell (the D3 eval size), fixed eval seed 777.

**Per-cell quantities (all from existing machinery):**
- **Oracle informativeness** `I_oracle` — top_coding via `lod_oracle_cell` (profiled-MAR-null Bayes-error);
  own-value via the feasibility profiled oracle. **Locked transform:** `I_oracle = 1 − 2·BE ∈ [0, 1]`.
  *Caveat recorded:* fitted-Gaussian X-model ⇒ necessary-not-sufficient; claims are relative to the
  profiled-MAR comparison class.
- **Derived estimator** `I_gain` — mean over the cell's examples of `KL(p ‖ prior_marginal)` from the
  temperature-calibrated posterior (binary: `prior_marginal = (0.5, 0.5)`, max 1 bit).
- **Probe estimator** `I_probe` — ridge regression (primary; depth-0 challenger) **and** one 2-layer MLP
  (secondary; depth-1 challenger) from the model's per-example pooled φ representation `e_col` to
  `I_oracle`, trained on **training-domain cells only**, evaluated on held-out-domain cells
  (leave-domain-out). Probe hyperparameters fixed in advance: ridge α=1.0; MLP 32-unit, 200 epochs, lr 1e-3,
  no tuning. *(Two capacities so a ridge failure alone cannot be blamed on probe weakness; if the MLP wins
  only in-distribution, J3 still kills it.)*
- **Coverage-distance** `cov(column)` — the D2 metric of the cell's target column against the model's own
  training pool (per-model, since pools differ by held-out domain).

**Gate threshold (locked by RULE before training; computed by a descriptive script, D3-precedent):**
`θ_gate` = the **95th percentile of within-pool self-coverage** (each training column's distance to the
rest of the training pool, leave-one-column-out). One θ per trained pool. A column is **gated out**
(UNKNOWN) iff `cov > θ_gate`. The rule — not a hand-picked number — is what is pre-registered; the script
(`preregister_detectability_gate.py`) runs and commits its output **before any training**.

## 3. Pre-registered predictions (locked now)

P1. Wealth columns: `cov ≫ θ_gate` (gated out). NHANES/HMDA continuous columns: mostly `cov ≤ θ_gate`.
P2. `I_oracle`: top_coding ≫ own_value at δ > 0 (the known spectrum); both → 0 at δ = 0.
P3. If H₀ suffices: gated-in `I_gain` tracks `I_oracle` across cells (F1) and separates idioms (F2).
P4. The D3 pathology (ECE↑ entropy↓) concentrates in gated-out cells (F4).

## 4. Pre-registered metrics

- **Agreement:** Spearman rank correlation between an estimator and `I_oracle` over **held-out-domain,
  gated-in cells** (primary), pooled across the two held-out domains and 8 seeds (per-seed corr → mean ± SE).
- **Idiom separation:** Δ_idiom = mean estimator value (top_coding, δ>0 cells) − (own_value, δ>0 cells),
  at gated-in, coverage-matched cells; significance = >2·SE over seeds.
- **Gate rates:** hit = fraction of wealth columns gated out; false-flag = fraction of held-out NHANES/HMDA
  continuous columns gated out.
- **Gated calibration:** ECE on gated-in vs gated-out cells (the D3 §5 measurement, split by the gate).

## 4½. Pre-execution note — scientific interpretation of F1 (PI, 2026-06-07, added before any run)

> **Spearman(`I_gain`, `I_oracle`) ≥ 0.50 means the derived estimator is sufficiently aligned with oracle
> distinguishability to support governance use as a RANKING signal across domains and idioms, subject to
> the coverage gate.** It is not a claim of point-accurate informativeness estimation; the governance
> deliverable (column triage — which columns deserve sensitivity analysis first) needs reliable *ordering*
> within footprint support, and F1 is calibrated to exactly that bar.

*Pre-hoc computational note (recorded before any training; no criterion altered):* F4's "gated-out cells
contain the worst-ECE tertile" is computed as — pool all eval cells (held-out + control), rank by per-cell
ECE, take the worst tertile; the clause holds iff **a majority (> 50%) of that tertile is gated-out**.

## 5. E-FALSIFY — criteria under which the head is struck (H₀ sufficient)

ALL of:
- **F1 (oracle agreement):** gated-in OOF Spearman(`I_gain`, `I_oracle`) ≥ **0.50** (mean − 1·SE above 0.50).
- **F2 (idiom spectrum):** Δ_idiom(`I_gain`) > 0 at **>2·SE**, same sign as Δ_idiom(`I_oracle`).
- **F3 (gate):** wealth hit rate ≥ **0.90** AND held-out covered false-flag ≤ **0.20**.
- **F4 (honesty composition):** gated-in ECE ≤ **0.20** on held-out domains AND gated-out cells contain the
  worst-ECE tertile (the D3 monotone pattern disappears once gated).

**If F1–F4 all hold:** detectability = **derived metric + governance reporting**; the Stage-2 learned head
is **removed from the design of record** regardless of probe performance (no-new-parameters preference at
parity or better).

## 6. E-JUSTIFY — criteria under which the head survives review

The head survives **only if** the derived stack fails where the probe succeeds — ALL of:
- **J1 (margin):** gated-in OOF Spearman(`I_probe`, `I_oracle`) exceeds Spearman(`I_gain`, `I_oracle`) by
  ≥ **0.15** (mean − 1·SE above the margin), for the **ridge** probe or (if only the MLP clears it) with J3
  satisfied.
- **J2 (decisive capability):** F2 **fails** for `I_gain` AND Δ_idiom(`I_probe`) > 0 at >2·SE — i.e. the
  probe separates idioms at matched coverage where the free estimator cannot.
- **J3 (transfer, the P2.2c lesson):** J1/J2 hold **leave-domain-out**, not merely in-distribution.
  An in-distribution-only probe win = **NOT justified**.

**Survival is narrow:** even on J1–J3 passing, the head is approved only for the **in-manifold
informativeness estimator role**, behind the (derived) coverage gate — never as the gate.

## 7. Outcome table (all branches named in advance)

| F1–F4 | J1–J3 | verdict |
|---|---|---|
| pass | — | **Head struck.** Detectability = derived metric (gate + info-gain), shipped as governance fields. |
| F1/F2 fail | pass | **Head survives review** — in-manifold estimator role only, gated, built as Stage-2 with its own spec. |
| F1/F2 fail | fail | **Neither estimator is trustworthy in-manifold.** Runtime detectability **deferred**: report oracle-validated detectability at eval time only; the manifest carries coverage-state + UNKNOWN, no info-gain claim. (An honest "we cannot yet estimate detectability at deployment" is a recordable result.) |
| **F3 fails** (gate broken) | — | **Finding against the D2 metric's column-level sufficiency** — redirect to the support metric (e.g. mask-topology axes), **not** to a head; both estimators blocked on a valid gate. |
| F4 fails alone | — | gate flags support but not calibration honesty ⇒ treat as partial F3; same redirect, with the ECE decomposition as the diagnostic. |

## 8. Guardrails (binding)

No architecture change · no HP tuning (frozen D3 config; fixed probe HPs) · probes are post-hoc analysis
artifacts, never wired or shipped · no new data / no acquisition · no metadata · no Level-2 deviation ·
gate threshold locked by rule via a committed descriptive script **before** any training · all thresholds
in §5–§6 are locked by this document and may not be revised after results exist · oracle claims name the
comparison class (profiled-MAR-on-predictors, fitted-Gaussian caveat) · the study's product is the §7
verdict + a findings doc, **not** a metric improvement.

## 9. Execution plan (on PI approval — nothing started)

1. `scripts/preregister_detectability_gate.py` — descriptive; computes & commits θ_gate per pool +
   per-column coverage (locks P1 numerically). **Run before any training.**
2. `scripts/run_detectability_study.py` — trains the leave-domain-out subjects (frozen recipe), computes
   per-cell `I_oracle` / `I_gain` / `I_probe` / gate state, evaluates §4 metrics, prints the §7 verdict row.
3. `docs/findings/E-JUSTIFY-E-FALSIFY-findings.md` — results against every locked criterion, honest caveats, the
   architectural verdict on the Stage-2 head. Stop for PI review.

Estimated scope: ~16 model trainings (2 held-out domains × 8 seeds; CPU, D3-scale) + oracle sweeps over
~40–60 cells + two probes. No GPU dependency.

---

*No training, probing, runs, or architecture changes are performed by this document. It locks the null
(derived stack), the challenger (probe), every threshold, and every outcome branch — including the ones in
which the Stage-2 head is struck, survives narrowly, or the question is deferred. Execution awaits PI
approval.*
