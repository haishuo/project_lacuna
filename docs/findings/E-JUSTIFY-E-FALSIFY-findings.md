# E-JUSTIFY / E-FALSIFY — Findings & Architectural Verdict

*Result of the pre-registered detectability study (`PROPOSAL-E-JUSTIFY-E-FALSIFY-detectability-study.md`,
PI-approved; gate locked before training, commit `b9b0c8d`). Executed exactly as specified: no tuning, no
new estimators, no threshold changes, no post-hoc reinterpretation. Code:
`scripts/preregister_detectability_gate.py` (θ_gate, committed pre-training),
`scripts/run_detectability_study.py`; artifacts `runs/detectability_gate.json`,
`runs/detectability_study.json` (3,008 cell-rows). 16 subjects (2 pools × 8 seeds, frozen D3 recipe,
top-coding-trained per the verbatim-recipe rule), 240 cells, 180 oracle MC cells, ridge + MLP probes.*

---

## 0. The architectural answer

> **Does the Stage-2 detectability head earn its place in the architecture? — NO.**

No branch of the locked outcome table grants survival: the probes **decisively fail J2** (no idiom
separation: +0.006 ± 0.006) and reach only **0.17 absolute** oracle agreement OOF (vs the 0.50 governance
bar). **But the derived stack did not win either** — F1 fails decisively (info-gain ⊥ oracle, Spearman
≈ 0). Two pre-registered failure rows apply simultaneously (reported per the table, ambiguity flagged):
**runtime detectability is DEFERRED** (neither estimator trustworthy in-manifold) **and** the gate's
column-level letter failed (F3) → support-metric work, **not** a head. Behind the mechanical verdict sits
one deep scientific finding (§4): **oracle distinguishability and learned-channel reliability genuinely
diverge — they rank both cells and idioms differently — so the oracle is the right identifiability ground
truth but the wrong runtime calibration target.** The head, as specified (a regressor onto oracle
informativeness), was aimed at the wrong target; so was F1's premise for `I_gain`.

## 1. Criteria — strict (locked letter) and valid-seed sensitivity

One subject (hmda-pool, **seed 7**) was **degenerate**: 22/24 of its HMDA-cell outputs were NaN (training
instability — echoing the D3 diverse-pool seed collapses). The locked letter computes means over all 8
seeds → NaN-poisoned; the sensitivity column drops only that seed (n=7). **No pass/fail flips between the
two readings** except J1 (noted below, and moot given J2).

| criterion (locked) | strict | valid-seed (n=7) | verdict |
|---|---|---|---|
| **F1** Spearman(`I_gain`,`I_oracle`) gated-in OOF ≥ 0.50 | NaN | **−0.034 ± 0.021** (per-seed −0.148…+0.016) | **FAIL — decisive** |
| **F2** idiom sep of `I_gain` > 2·SE, sign = oracle | NaN | +0.031 ± 0.012 (>2·SE) **but oracle Δ = −0.083 — SIGN MISMATCH** | **FAIL** (see §4) |
| **F3** wealth hit ≥ 0.90 ∧ false-flag ≤ 0.20 | hit **0.833**, ff **0.000** | same (gate is pre-training-locked) | **FAIL on the letter** (see §3) |
| **F4** gated-in ECE ≤ 0.20 ∧ worst-ECE tertile majority gated-out | ECE **0.117 ± 0.028** ✓; tertile gated-out frac **0.30** ✗ | same | **FAIL** (second clause) |
| **J1** probe margin ≥ 0.15 (mean−SE) | NaN | ridge **+0.199 ± 0.033** (mean−SE 0.167) → would pass; MLP +0.08 fails | strict FAIL / sensitivity pass — **moot** (J2) |
| **J2** probe separates idioms where `I_gain` fails | +0.006 ± 0.006 | same | **FAIL — decisive** |
| **J3** OOF (leave-domain-out) | by construction | — | satisfied but moot |

Absolute OOF oracle agreement: `I_gain` **−0.03**, ridge probe **0.166 ± 0.017**, MLP 0.047 ± 0.033.

## 2. Verdict per the locked outcome table

- **Head survival** requires (F1/F2 fail) ∧ J1 ∧ (J2 ∨ F2) ∧ J3 → **J2 fails and F2 fails ⇒ the head does
  not survive under any reading** (including the J1-favorable sensitivity reading).
- **"Head struck (derived stack wins)"** requires F1–F4 pass → F1 fails decisively ⇒ not this row either.
- Two failure rows match simultaneously (the table did not define precedence — reported honestly):
  - **Row "F1/F2 fail + J fail":** *neither estimator is trustworthy in-manifold ⇒ runtime detectability
    DEFERRED — report oracle-validated detectability at eval time only; the manifest carries
    coverage-state + UNKNOWN, no runtime info-gain claim.* **Adopted as the operative consequence.**
  - **Row "F3 fails":** *finding against the D2 metric's column-level sufficiency ⇒ redirect to the
    support metric, not a head.* **Adopted for the gate workstream** — with the §3 qualification.

**Net architectural outcome:** the Stage-2 detectability head is **not approved** and is removed from the
build plan (it may only re-enter via a new spec against a *deployed-channel* target, §5 — a PI decision,
not scheduled). Runtime detectability claims are **deferred**; detectability remains an **eval-time,
oracle-validated, comparison-class-named** reporting quantity. The coverage gate continues as a
**governance/OOD** signal (its D3 validation stands) but gains no detectability authority from this study.

## 3. The F3 letter-fail, honestly

Strictly: wealth hit rate 5/6 = 0.833 < 0.90 ⇒ FAIL. The single miss is **`wealth.age`** — gated **in** at
coverage 0.26–0.55. As recorded **before training** (commit `b9b0c8d`): age is genuinely a moderate-regime
column; gating it in is arguably **correct** behavior, and the criterion's denominator (all base columns)
charges the gate for it. Functionally the gate did its job: **5/5 wealth dollar aggregates gated out**
(cov 3.1–3.4), **0/13 false flags** on covered held-out columns. We apply the letter (FAIL) and do not
reinterpret — but the *substantive* gate problem is F4, not F3: **30% of the worst-calibrated tertile is
gated-out, i.e. most badly-calibrated cells are gated IN.** Coverage catches the *off-regime* failure mode
(D3) but does **not** capture in-manifold reliability variation. That is the real column-level
insufficiency the redirect should address.

## 4. The deep finding — oracle distinguishability ≠ learned-channel reliability

The study's most important scientific output was not a criterion number:

- **The oracle ranks the idioms opposite to the learned channel.** At n = 384, matched rate, profiled-MAR
  null, held-out cells: `I_oracle`(own_value) = 0.75 / 0.90 / 0.98 at δ = 0.75/1.25/2.5 vs
  `I_oracle`(top_coding) = 0.60 / 0.79 / 0.99 — **own-value is MORE oracle-distinguishable** (Δ = −0.083).
  This is consistent with P1 (the own-value signal *survives* the profiled MAR at the Bayes-optimal level)
  and with P2.2c's three-levels memo: own-value is flat at the **learned/transferable** level, not at the
  oracle level. Meanwhile `I_gain` shows the **learned** spectrum (+0.031, top_coding > own_value) — the
  spectrum every prior result (Stage-0, D3) shows for the deployed channel.
- **Consequence:** F2's "sign must match the oracle" requirement could not have been satisfied by a
  *faithful* estimate of either quantity — the two quantities genuinely order the world differently. The
  same divergence explains F1: the calibrated posterior's information gain tracks what the *trained
  channel* extracts, which is **rank-uncorrelated** with what the *Bayes-optimal* observer could extract
  (Spearman ≈ 0; the probe recovers only 0.17 of it from the φ representation).
- **Architectural moral:** the design-of-record's detectability head (regress `z` → oracle informativeness)
  and the derived `I_gain`-vs-oracle validation were both aimed at the **identifiability ceiling**, but the
  governance consumer needs the **deployed channel's reliability**. The spec's recorded caveat ("oracle is
  necessary-not-sufficient") understated the problem: the oracle is not merely an upper bound — **its
  ranking disagrees with deployed reliability**. Any future detectability estimator must be calibrated to a
  deployed-channel target (e.g., held-out calibration/accuracy of the δ-prior itself), with the oracle
  retained for *identifiability* claims only. That re-definition is a PI decision (§5); nothing is built
  here.

## 5. What follows (PI decisions; nothing started)

1. **Stage-2 head: not approved** (this review). Strike from the build plan; any future head proposal must
   target a deployed-channel reliability quantity, not oracle informativeness, and re-enter through a new
   pre-registered spec.
2. **Runtime detectability: deferred.** Manifest/report carry {δ-prior, coverage-state, UNKNOWN} +
   eval-time oracle-validated detectability (comparison class named). No runtime info-gain claims.
3. **Gate workstream:** keep coverage as the OOD/governance gate (D3 validation stands); the F4 result
   defines the open problem — in-manifold reliability variation that coverage does not capture. Candidate
   directions (unscoped): mask-topology axes; per-cell held-out reliability maps over `P_prior`.
4. **Definitional update for MASTER (recommended):** split "detectability" into **identifiability**
   (oracle, eval-time, comparison-class-relative — solid) and **deployed-channel reliability** (the thing a
   runtime signal must estimate — currently unsolved). The lab-coat-fraction headline survives intact: it
   is an *eval-time* measurement and never depended on a runtime head.

## 6. Honest limitations

- Subjects are **top-coding-trained** (D3 recipe verbatim); `I_gain`'s idiom behavior partly reflects
  footprint-specificity of that training. A mixed-idiom subject was not licensed by the frozen spec.
- One degenerate seed (NaN subject) — training instability remains an open reliability issue (third
  sighting: SCF diverse collapses, D3 diverse seed, now hmda-pool seed 7). No verdict hinges on it.
- Oracle cells use the fitted-Gaussian X-model (recorded caveat); oracle SE ≈ 0.012 at n_mc = 800.
- The oracle-vs-learned divergence is established **at this n (384), rate (0.3), and δ-grid**; the gap
  could narrow at other regimes. The direction of the architectural conclusion (calibrate to the deployed
  channel) does not depend on the gap's exact size.

---

*Executed exactly per the frozen specification; all criteria evaluated against locked thresholds; both
strict and degenerate-seed-sensitivity readings reported; no criterion, threshold, estimator, or dataset
was altered after results existed. Stopped for PI review.*
