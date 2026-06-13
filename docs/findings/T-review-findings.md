# Network Load-Bearing Review — Findings (T1 / T2 / T3) and Verdict

*Findings for the pre-registered review `docs/proposals/PREREGISTRATION-network-load-bearing-review.md`
(locked at commit `7714f0c`, PI-authorized 2026-06-11). All thresholds, feature lists,
architectures, and the verdict table were frozen before any result existed. Implementations were
committed before results (T1 `fd87151`, T3 `31be57d`, T2 `0a7e394`). This document reports every
number produced, every rig repair, and the mechanically computed §6 verdict row. Runs executed
2026-06-12, 00:30–01:55.*

---

## 0. Summary of verdicts (computed mechanically from the locked bars)

| test | locked bar | result | verdict |
|---|---|---|---|
| **T1** mask-topology showdown | network mean−SE ≥ best-shallow + 0.05 (= **0.988**) AND MCAR-vs-not ≥ shallow − 0.02 (= **0.980**) | network seeds 0.844, 0.858 (run stopped at 2/5 — verdict mathematically forced, §1.4); MCAR 0.877–0.885 | **FAIL** |
| **T2** vocabulary break | network mean−SE ≥ best-shallow + 0.05 (= **0.889**), margin specifically on skip/mixture cells | network mean−SE **0.840** vs shallow 0.839 (margin **+0.001**); skip/mix margin **+0.009** | **FAIL** |
| **T3** amortization | pooled OOF AUC ≥ teacher − 0.02 for BOTH idioms (top_coding ≥ **0.802**, own_value ≥ **0.751**) | top_coding mean−SE **0.690**; own_value mean−SE **0.699** | **FAIL** |

**Verdict row (prereg §6, locked):** fail / fail / fail ⇒ **SCRAP/PIVOT RECOMMENDATION** — "No
defensible load-bearing neural role exists in Lacuna on current evidence; the honest system is the
explicit pipeline, which does not meet the degree constraint. Recommend the PI pivot." The joint
prior recorded for this row before any run was ~15–20%.

**Scope of the verdict (binding, §5 of this doc):** this is a verdict about the *numeric-matrix
channel of the survey vocabulary under the named prior* — networks vs the frozen feature null on
the testable (semi-synthetic, known-δ) distribution. It licenses no claim about other data
domains (e.g., EHR/longitudinal), and no claim about channels outside the matrix (e.g., item
semantics/metadata — see the companion pivot proposal).

---

## 1. T1 — the mask-topology showdown (the test v1.0 never ran)

### 1.1 Setup (locked pre-results)

8 generators across 3 classes (MCAR: uniform + rotated-booklet; MAR: 1-pred logistic,
multi-pred, module-skip; MNAR: own-value, top-coding, module-refusal), all matched to overall
rate 0.3; 4-domain corpus (labor / nhanes / hmda / wealth), block-aware leave-one-domain-out;
128-row × all-column examples; train 2400 / val 480 / test 480 per split; both arms consume
bit-identical example objects. Feature arm: frozen 37 topology statistics, LR + HistGBM,
better-of. Network arm: v1.0 LacunaModel family (hidden 128 / evidence 64 / 4 layers / 4 heads),
`use_missingness_features=False` (the network must earn the statistics itself — that is the
question), CE + reconstruction loss, 5 seeds.

### 1.2 Rig-repair history (documented, as the prereg requires)

1. **Booklet K-fallback** (`32d7642`): rotated-booklet had no K-multiple admitting rate 0.3 at
   some widths (d=8/K=3); fixed with a documented K-fallback, verified d=3..11 × 8 generators
   under runner retry semantics. Pre-results harness fix.
2. **FIRST RUN VOID — unstandardized inputs** (`bf2562d`; artifacts preserved as
   `runs/t1_showdown_VOID_unstandardized.*`): the harness fed raw dollar-scale values; the
   reconstruction loss (~8.6e6) drowned the CE gradient and the network sat at chance *on its own
   training set* — a harness bug by the pre-declared diagnostic (train-fit ≈ 0.5 ⇒ rig, not
   result). Repair: per-column standardization by observed-cell statistics (runtime-computable;
   v1.0 always trained on standardized inputs; the scale-free feature arm is unaffected).
   Verification before relaunch: `runs/diag_t1_fit2.log` — the repaired network reaches VAL
   macro-AUC 0.915 by epoch 12 on 1/5-size data. **The network learns the task; the showdown
   measures transfer, not trainability.**

No tuning was performed at any point; both repairs are harness-class and were committed with
documentation before the scored run.

### 1.3 Results

**Feature arm (frozen 37):**

| arm | pooled OOF macro-AUC | MCAR-vs-not | skip-vs-refusal |
|---|---|---|---|
| LR | 0.932 | 1.000 | 0.836 |
| GBM | **0.938** | 1.000 | 0.887 |

**Network arm** (run stopped after 2 of 5 seeds; see §1.4):

| seed | pooled OOF macro-AUC | MCAR-vs-not | skip-vs-refusal |
|---|---|---|---|
| 2026 | 0.844 | 0.885 | 0.883 |
| 7 | 0.858 | 0.877 | 0.896 |

### 1.4 Verdict mechanics — why 2/5 seeds already decide it

The bar requires mean−SE ≥ 0.988 over 5 seeds. With seeds 1–2 at 0.844 and 0.858, even if the
three remaining seeds scored a perfect 1.000, the mean would be 0.940 < 0.988 before subtracting
SE. Independently, the MCAR-vs-not secondary requires ≥ 0.980; the observed 0.877–0.885 fails it
with no path to recovery (it is a pooled metric, not a best-of). **T1 = FAIL, with the network
not merely short of +0.05 but ~0.09 *below* the shallow arm on held-out domains.** The
rate-shuffled control is required only on a pass and was therefore not run. The remaining 3 seeds
were not completed (run stopped); completing them would change reported means, not the verdict —
they can be rerun on request for record completeness.

### 1.5 Feature decomposition (descriptive, not gated — `runs/t1_feature_decomposition.log`)

| feature set | LR | GBM |
|---|---|---|
| anchors only (rate, n, d) | 0.493 | 0.616 |
| v1.0 legacy 7 (rates + xcorr) | 0.862 | 0.899 |
| 10 new topology stats | 0.946 | 0.940 |
| 17 consequence (value-marginal) | 0.692 | 0.695 |
| legacy7 + topology | 0.945 | 0.950 |
| full frozen 37 | 0.932 | 0.940 |

Findings: (i) the 0.94 ceiling is **feature engineering, not the learner** — plain LR on the 10
new topology statistics already reaches 0.946; GBM adds ~nothing. (ii) **Anchors-only LR at
0.493 validates the rate-confound guard** — at matched rate, rate carries no linear signal.
*Honest footnote:* anchors-only GBM reaches 0.616 via a rate-**precision** fingerprint
(deterministic block mechanisms hit exactly 0.300 while stochastic ones jitter around it) — a
generator artifact worth knowing about, not a mechanism signal; it is unavailable to the
matched-rate examples' class boundary at the resolution the locked task uses, and the full-37
arms dominate it regardless. (iii) The frozen-37 list slightly *understates* the achievable
feature ceiling (legacy7+topology 0.950 > full-37 0.940), so the locked bar was conservative
**toward the network**.

### 1.6 Reading

The v1.0 question ("from the whole table, should you worry?") is answerable at matched rate —
0.94 macro, MCAR-vs-not 1.000 — but by enumerable statistics. The cross-column structure the
BERT/MoE apparatus was built to capture (block alignment, gate correlations, eigenstructure) is
exactly what 17 cross-column statistics capture; asked to earn those statistics itself, the
network reaches 0.85 in-family and transfers worse than the feature arm. The "see the entire
dataset" advantage was real as a *task requirement* and empty as a *neural requirement*.

---

## 2. T2 — the vocabulary break (skip-logic + mixtures)

### 2.1 Setup (locked)

Vocabulary-3 with mixtures: {top_coding, own_value, skip_logic, matched-rate MAR null}, 1–2
mechanisms per table on different columns; per-flagged-column joint family + binary-δ task;
1921 flagged examples over the 4 domains (d_max 9), leave-one-domain-out. Feature arm: union of
ALL built features (17 consequence + 16 H/S/R no-truth + topology list per column), LR + GBM.
Network arm: design-of-record Level-1 φ + mask-topology stream + fusion + family/δ heads (the
build the prereg authorized), one fixed config, 5 seeds. Primary = pooled OOF macro-AUC over
family × binary-δ.

### 2.2 Results

| arm | primary | family | δ | skip/mix slice |
|---|---|---|---|---|
| shallow LR | 0.788 | 0.666 | 0.910 | 0.783 |
| shallow GBM | **0.839** | 0.727 | 0.951 | 0.837 |
| network s2026 | 0.846 | 0.727 | 0.965 | 0.851 |
| network s7 | 0.846 | 0.717 | 0.975 | 0.848 |
| network s99 | 0.842 | 0.718 | 0.966 | 0.847 |
| network s13 | 0.834 | 0.707 | 0.960 | 0.835 |
| network s41 | 0.845 | 0.727 | 0.964 | 0.848 |

Network mean−SE **0.840** vs best shallow 0.839 ⇒ margin **+0.001** (bar: ≥ +0.05). Skip/mixture
slice: network 0.846 vs shallow 0.837 ⇒ **+0.009** (bar: margin specifically here ≥ +0.05).
**T2 = FAIL.** All 5 seeds completed; none degenerate; seed spread 0.834–0.846 (tight).

### 2.3 Reading

The hypothesis was that mixtures + skip-logic create combinatorial structure that breaks
enumeration. They do not: the full enumerable union (GBM) tracks the network to within noise on
the exact cells where the break was predicted. Parity, not victory — the burden of proof was on
the network and +0.05 was the locked materiality threshold. Notably both arms agree the *family*
sub-task is the hard part (0.71–0.73) and δ-binning is easy (0.95+) — consistent with the wider
arc: strength is footprint-visible, mechanism identity is the identification-limited part.

---

## 3. T3 — amortization (the PFN role)

### 3.1 Setup (locked)

Conditional-φ: per-row tokens [predictor values; target value-or-MISSING flag] → row-set encoder
→ column embedding → calibrated family/δ head; one fixed config; trained on the G1/empty-cell
machinery; eval on bit-identical examples (3840; 100% identity-checked) against the explicit
pipeline teacher (5 imputers × 3 fits + H/S/R + LR). Teacher numbers measured on the locked
splits: top_coding mean 0.822 (per-imputer 0.813–0.830), own_value 0.771 (0.743–0.796). Bars =
teacher − 0.02: **0.802 / 0.751**.

### 3.2 Results

| idiom | seeds (OOF AUC) | mean | SE | mean−SE | bar | verdict |
|---|---|---|---|---|---|---|
| top_coding | 0.723, 0.668, 0.725, 0.698, 0.691 | 0.701 | 0.011 | **0.690** | 0.802 | FAIL |
| own_value | 0.734, 0.720, 0.670, 0.757, 0.692 | 0.715 | 0.015 | **0.699** | 0.751 | FAIL |

ECE range 0.16–0.33 (poorly calibrated throughout). Single-pass inference 0.511 ms/column vs
~15 model fits/column for the explicit pipeline — the speed claim holds (≥100×), but speed was
reported-not-gated and parity was the bar. **T3 = FAIL** (bar requires BOTH idioms; top_coding
misses by 0.112, own_value by 0.052).

### 3.3 Reading

The amortizer could not match its own teacher on bit-identical examples — it does not reproduce
the explicit pipeline's information extraction, it approximates it lossily. own_value is the
narrow miss (0.699 vs 0.751); top_coding is decisive. Under this prereg the PFN role is failed;
any future amortization attempt is a new, separately pre-registered effort with a redesigned
conditional representation, not a reopening of this one.

---

## 4. The detectability map (context the verdict sits in)

From the June arc + `runs/detectability_study.log` (240 cells; oracle = truth-access):

- **MCAR-vs-not: 1.000** (features) — the identifiable gate works perfectly.
- **Top-coding / skip-logic: data-detectable** (truncation spike; gate-column correlation) — and
  feature-capturable (0.81–0.95 across tasks).
- **Own-value self-censoring: the flat idiom.** Near-chance OOF for features AND networks at
  realistic δ; best-ever 0.74–0.80 via the explicit imputation pipeline at the extreme δ=2.5.
  On heavy-tailed wealth columns (SCF networth/asset) even the truth-access oracle collapses
  (I_oracle 0.12–0.14) — the information is absent, not merely hard.
- **Detectability scoring itself** (predict which columns are detectable, OOF): Spearman ≈ 0.00
  vs oracle — fails for ridge and MLP alike.
- **Skip-vs-refusal (the PHQ-9 pair):** 0.84–0.90 both arms — separable only through the
  identifiable shadow (observed-gate correlation), residual error real.

The consistent pattern across the entire review: **wherever signal exists in the matrix it is
low-dimensional and feature-capturable; wherever features fail, networks fail identically, and in
the hardest cells the truth-oracle itself fails** — the binding constraint is identification, not
representation.

## 5. Scope, limitations, honest caveats

1. **Comparison-class relativity.** Every MNAR-vs-MAR number is relative to the *named plausible
   MAR class* (logistic-on-observed-predictors, gates, routing). The Molenberghs exact-mimic MAR
   is deliberately excluded: including it makes the labels ill-posed (identical observed-data law
   under both labels ⇒ Bayes error 0.5 for any method). Claims must never be read as "MNAR is
   identifiable."
2. **No graded-mimic ladder.** The vocabulary lacks a ρ-graded MAR-on-proxy family (footprint
   continuously approaching own-value as ρ→1). Absolute detectability margins are therefore
   optimistic against a hostile-but-plausible MAR world; the *comparative* (network vs feature)
   verdict is unaffected, since both arms face the same competitors.
3. **Semi-synthetic circularity.** All results are statements about the named prior P_prior (our
   generators on real X). They are the only testable distribution (real missingness has no δ
   truth), and the feature lists were partly reverse-engineered from the same vocabulary — which
   is precisely why the locked margins put the burden on the network. The verdict is: *on the
   most favorable testable distribution we could construct, the network adds nothing.* There is
   no evidence path to a stronger claim for the network elsewhere in the matrix channel.
4. **T1 stopped at 2/5 seeds** (PI stop). The verdict is mathematically forced (§1.4); the
   remaining seeds affect reported means only. Rerun available on request.
5. **Domain scope.** Survey vocabulary on cross-sectional numeric matrices only. Nothing here
   measures EHR/longitudinal/event data, where mechanism footprints are plausibly
   higher-dimensional (informative presence, workflow cascades, temporal structure).

## 6. What the review establishes (the constructive reading)

The judgment "income is missing not at random" — which essentially every statistician would make
— decomposes, on these measurements, into **~all prior and ~no likelihood**: at matched rate the
matrix is nearly silent on self-driven informativeness (§4), and what little the matrix does say
is captured by enumerable statistics (§§1–3). The information that drives the human judgment
lives in the *semantics of the item* ("income is sensitive"), a channel this entire review —
by deliberate construction ("strip the column names") — never gave any model. That measured
decomposition is the empirical basis for the companion proposal
(`docs/proposals/PROPOSAL-semantic-channel-pivot.md`): the load-bearing-ML question moves to the channel
where the information demonstrably lives.

*Verdict row computed mechanically from locked bars; no threshold, feature list, or architecture
was altered after results existed. Review complete; awaiting PI decision.*
