# Pre-Registration — Is the Neural Network Load-Bearing? (T1–T3 Architectural Review)

*Pre-registration — **committed before any test code or run exists** (PI-authorized 2026-06-11). This
review supersedes the standing architecture freeze: it IS the authorized architecture phase, bounded to
exactly the three tests below. Its outcome is binding via the locked verdict table (§6): it determines
whether Lacuna proceeds with a neural network as its load-bearing estimator, proceeds under a narrower
neural role, or is recommended for scrap/pivot. Thresholds, feature lists, architectures, and the
verdict table may not be altered after results exist. Discipline as established: matched rates,
block-aware leave-one-domain-out, bit-identical examples across arms, comparison-class-named claims,
honest reporting of every failure.*

---

## 0. The question and the stakes

The June arc established that at vocabulary-2, within-column δ-strength estimation, a **shallow model on
specifiable statistics matches the neural channel**. The PI's dissertation requires a neural network to
be **genuinely load-bearing** — not packaging. Three named roles remain where the network could earn
that status; each gets one pre-registered test:

| test | role under review | the question | prior odds (recorded) |
|---|---|---|---|
| **T1** | mask-topology learner (the v1.0 axis) | at **matched rate**, does a network beat enumerable cross-column mask statistics for mechanism-class discrimination? *(the showdown v1.0 never ran)* | ~50–60% |
| **T2** | vocabulary-scale generalizer | does the enumerable-feature ceiling crack at **vocabulary-3 + mixtures** (skip-logic added)? | ~60% |
| **T3** | amortized estimator (PFN role) | can **one forward pass** reproduce the explicit multi-model pipeline's per-column posterior OOF? | ~85% |

**Burden of proof is on the network in T1/T2** (challenger vs the feature null, as in E-JUSTIFY) and
**on parity in T3** (the network must match its explicit teacher). The review's product is the §6
verdict row — including the scrap row.

## 1. Shared protocol (all tests)

- **Corpus:** the 4-domain continuous role-B corpus (labor / nhanes / hmda / wealth-cont) + the full
  multi-column tables for topology tasks. **Block-aware leave-one-domain-out**; pooled OOF arbiter.
- **Matched overall missingness rate = 0.3** for every mechanism within a comparison (the rate-confound
  guard — the flaw measured in v1.0 must not be reproducible here). Rate is always included as a feature
  so its (lack of) signal is auditable.
- **Bit-identity discipline:** both arms (network, features) consume **identical generated examples**
  (shared seeds; checksum verification as in the empty-cell run).
- **Feature arms are FROZEN by this document** (§2/§3/§4 lists). The anti-handicap guard: the lists were
  assembled to be the *strongest* enumerable baselines we can construct — everything from v1.0, the June
  arc, and new topology statistics. No additions after any network result exists. Shallow learners:
  **LR (primary) and gradient boosting on the same features (strong-shallow, secondary)** — the network
  must beat the **better** of the two.
- **Network training:** 5 seeds; mean − SE must clear the bar (degenerate seeds reported, never
  silently dropped). Calibration (ECE) reported alongside AUC everywhere.
- **No tuning after results.** Architecture configs are fixed in this document; one config each.

## 2. T1 — The mask-topology showdown (the test v1.0 never ran)

**Task.** Mechanism-class discrimination at matched rate, on **multi-column** semi-synthetic examples
over real survey X-bases. Generator vocabulary (all matched to overall rate 0.3; per-mechanism params
drawn from fixed ranges):

| class | generators |
|---|---|
| MCAR | uniform-cell MCAR · **rotated-booklet** (column-block × row-assignment) |
| MAR | single-predictor logistic MAR · multi-predictor MAR · **module-skip** (observed gate column drives a row-aligned block) |
| MNAR | own-value self-censoring · top-coding · **module-refusal** (row-aligned block driven by the block's own latent values) |

Primary metric: pooled OOF **macro one-vs-rest AUC** over the 3 classes. Two named hard-pair
secondaries (reported, not gated): MCAR-vs-not AUC; **module-skip vs module-refusal** AUC (the PHQ-9
pair — the literature-grounded identifiability boundary; we *expect* this pair to be hard for both arms).

**Feature arm (FROZEN; ~32 statistics).**
- v1.0 survivors: per-column missing-rate stats (mean/var/range/max); cross-column missingness-corr
  stats (mean |corr|, max |corr|, frac > 0.5).
- Block/row structure: top-eigenvalue share of the R-correlation matrix; row missing-count dispersion
  vs binomial (Fano factor); fraction of rows all-or-none missing within detected column blocks;
  column missing-rate bimodality (dip-statistic proxy: variance of rates + frac ∈ {0, 1}).
- Gate signatures (skip detection): max over observed columns of AUC(column → block missingness);
  max point-biserial |corr(R_j, observed col k)| (mean/max over pairs).
- Value-conditional: per-column observed-vs-missing-row mean shift on correlated columns (max/mean);
  the 17 consequence features of the highest-missing-rate column.
- Anchors: overall rate; n; d.

**Network arm (one fixed config).** The v1.0 architecture family retrained from scratch on this
matched-rate vocabulary: the existing encoder + MoE gate (current-code-compatible configuration), no
hand-fed feature vector (the network must earn the statistics from the mask+values itself — that *is*
the question). Frozen training recipe (v1.0 reproduction settings; no tuning).

**Locked bar:** the network **passes T1** iff pooled OOF macro-AUC (mean − SE over 5 seeds) exceeds the
better shallow arm by **≥ +0.05**, and does not lose on MCAR-vs-not. Margin < +0.05 or any reliance on
rate (auditable via the rate feature's shallow-arm weight and a rate-shuffled control) ⇒ **fail**.

**Effort:** ~3–5 days (generators mostly exist; rotated-booklet/module pair ported from v1.0's registry;
one network retrain × 5 seeds).

## 3. T2 — The vocabulary break (skip-logic + mixtures)

**Task.** Per-column **joint** mechanism-family + strength output at **vocabulary-3 with mixtures**:
{top_coding, own_value, **skip_logic** (observed gate → noisy block missingness; routing parameter
ranges), + matched-rate MAR null}, where each table draws **1–2 mechanisms simultaneously** on
different columns (mixtures — the realism step that breaks single-mechanism enumeration). Labels per
flagged column: family ∈ {null, predictor-driven, self-driven, skip} and δ-bin (binary slice).
Primary metric: pooled OOF macro-AUC over family × the binary-δ task, per held-out domain.

**Feature arm (FROZEN):** the **union of everything built** — 17 consequence features + 16 H/S/R
no-truth conditional features + the T1 topology list (computed per column where applicable) — LR and
GBM. This is the full enumerable ceiling the project can construct today.

**Network arm (one fixed config).** The design-of-record Level-1 completed with its **Stage-2
mask-topology stream** (φ per-column + mask-topology encoder + fusion + family/δ heads — the
architecture MASTER §5 always specified; built once, frozen config, 5 seeds).

**Locked bar:** **pass** iff network (mean − SE) beats the better shallow arm by **≥ +0.05** pooled OOF
macro-AUC **and** the margin appears specifically on cells involving skip/mixtures (the combinatorial
content — verified by a no-mixture ablation slice where parity is acceptable). Fail otherwise.

**Effort:** ~1–2 weeks (skip-logic generator + mixtures support + the mask stream build + runs).

## 4. T3 — Amortization (the PFN role)

**Task.** One network, one forward pass per column, trained on semi-synthetic labels to reproduce what
the **explicit pipeline** (5 imputers × 3 fits + H/S/R statistics + LR — ~seconds per column) achieves
OOF. Teacher/reference numbers (already measured, locked): own_value 0.74–0.80; top_coding 0.81–0.83.

**Network arm (one fixed config).** A **conditional-φ**: per-row tokens [predictor values; target value
or MISSING flag] → row-set encoder (DeepSets/attention, bounded size) → column embedding → calibrated
family/δ head. (The current φ is target-marginal-only by design; T3's network must ingest predictors —
this is the one genuinely new architectural element of the review, and it is exactly the conditional
representation the mid-rung result licenses testing.)

**Locked bar:** **pass** iff, on bit-identical eval examples, the network's pooled OOF AUC ≥
**(explicit pipeline − 0.02)** for **both** idioms, with single-forward-pass inference (wall-clock
per column reported; expected ≥ 100× faster — reported, not gated). Fail if it cannot reach
pipeline-parity OOF.

**Effort:** ~1 week (conditional-φ + training on the existing G1/empty-cell example machinery; the
teacher numbers and eval splits already exist).

## 5. Leak/fairness red-team (locked)

- **Anti-handicap:** feature lists frozen here, assembled maximally; any post-hoc "the features were
  weak" objection is answered by this document's timestamp.
- **Anti-rate-confound:** matched rate everywhere + rate included as an auditable feature + a
  rate-shuffled control on any passing network.
- **Anti-cherry-picking:** all three tests run regardless of early results; all reported; degenerate
  seeds reported; per-domain splits shown (no pooled-only claims).
- **Bit-identity:** shared example seeds + checksum verification across arms (empty-cell precedent).
- **No truth leakage:** T2/T3 feature and network inputs are observed-view-only (the L1–L8 guards
  inherited); the H/S/R features remain eval-validated instruments used identically in both arms.

## 6. Verdict table (locked; the decision the PI asked for)

| T1 | T2 | T3 | verdict |
|---|---|---|---|
| pass | — | — | **Network load-bearing on the deliverable's own axis.** Lacuna 2.0 proceeds: Stage-A topology network (matched-rate retrained, rate-confound killed) + Stage-B per-column governance. Strongest dissertation position. |
| fail | pass | — | **Network load-bearing at vocabulary scale.** Proceed vocabulary-first (skip-logic chapter is the heart); Stage-A may remain shallow with the network carrying the joint task. |
| fail | fail | pass | **Network load-bearing as the amortized estimator (PFN framing).** The network IS the deployed product (one-pass inference reproducing a multi-model statistical pipeline, posterior under a named prior — a recognized ML contribution class). **PI decision point:** accept this as the dissertation's neural heart, or pivot. |
| fail | fail | fail | **SCRAP/PIVOT RECOMMENDATION.** No defensible load-bearing neural role exists in Lacuna on current evidence; the honest system is the explicit pipeline, which does not meet the degree constraint. Recommend the PI pivot. |

Joint prior probability of the scrap row (recorded honestly): **~15–20%.**

In **every** non-scrap branch, the four PI objectives bind the build: explain what happens to the
matrix; defend each operation (named prior, matched rate, comparison class, coverage gate); explain
the output (per-column {coverage state, MCAR-departure, mechanism-family evidence, δ-prior, abstain}
with stated semantics); the network at the heart per the passing role. The v1.0 epistemic corrections
(rate confound; consensus-label limits; posterior-under-named-prior semantics) apply in every branch.

## 7. Sequencing & timeline (decision ≈ 4 weeks out)

| week | work | gate |
|---|---|---|
| 1 | T1 build + run (generators port, feature arm, network retrain ×5) | T1 verdict |
| 2 | T3 build + run (conditional-φ; teacher evals exist) — parallel start during T1 runs | T3 verdict |
| 2–4 | T2 build + run (skip-logic generator, mixtures, mask stream) | T2 verdict |
| 4 | findings doc + §6 verdict row + Report 6 implications | **PI go/no-go on Lacuna** |

Execution is gated on PI approval of this document. Each test produces its own findings section in one
review document; the verdict row is computed mechanically from the locked bars.

---

*Spec only — no code, no runs. Thresholds, feature lists, architectures, sequencing, and the verdict
table are locked as of this commit. The review answers one question: does the neural network earn its
place at the heart of Lacuna — and if not, it says so in time to pivot.*
