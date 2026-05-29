# Stage 4 (full) — Fine-tuned, multi-seed per-column subtype detectability

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006, Stage 4 (full version). Script: `scripts/stage4_subtype_detectability.py`.
- **Predecessor:** `2026-05-29-stage4-subtype-detectability.md` (the frozen-probe check). This is the
  handoff run from that doc's "Handoff" section.
- **Setup:** full registry (`lacuna_tabular_110`, 113 gens: MCAR 32 / MAR 40 / MNAR 41), one generator
  per dataset (single-mechanism, full diversity), each missing-bearing column labelled by its
  generator's class. **Fine-tune** the baseline encoder + per-column head + deployable features
  (`--fine-tune --epochs 40 --eval-batches 200`), Adam lr=1e-3. Baseline encoder/config:
  `/mnt/artifacts/project_lacuna/runs/stage0_general_baseline/`.
- **Seeds:** 7 (`--seed 1..7`), each a full independent train+eval. Outputs:
  `stage4_subtype_finetune_s{1..7}.json`; aggregate `stage4_subtype_finetune_aggregate.json`.
- **Status:** complete. The headline is a **negative / cautionary** result: fine-tuning does **not**
  yield a stable balanced per-subtype map — it yields a *seed-dependent collapse*.

## Headline — per-class recall is dominated by seed-to-seed attractor instability

Per-class recall pooled over all supervised columns per seed, then mean ± sd across the 7 seeds
(sd is population sd; sample sd in parentheses):

| class | mean ± sd | range over seeds | frozen probe (prior doc) |
|---|---|---|---|
| **MCAR** | 0.807 ± 0.240 (0.26) | [0.27, 0.98] | 0.222 |
| **MAR**  | 0.292 ± 0.237 (0.26) | [0.02, 0.66] | 0.177 |
| **MNAR** | 0.383 ± 0.194 (0.21) | [0.15, 0.79] | 0.806 |
| overall accuracy | **0.596 ± 0.051** | [0.51, 0.66] | ≈ 0.39 |

Per-seed by-class (the raw distribution — this is the result, not the mean):

| seed | MCAR | MAR | MNAR | dominant attractor |
|---|---|---|---|---|
| 1 | 0.27 | 0.54 | 0.79 | MNAR-leaning |
| 2 | 0.68 | 0.66 | 0.34 | MCAR+MAR balanced |
| 3 | 0.95 | 0.02 | 0.31 | MCAR-collapse |
| 4 | 0.87 | 0.35 | 0.50 | MCAR-leaning |
| 5 | 0.98 | 0.07 | 0.22 | MCAR-collapse |
| 6 | 0.96 | 0.37 | 0.36 | MCAR-leaning |
| 7 | 0.95 | 0.03 | 0.15 | MCAR-collapse |

**Only overall accuracy is stable (sd 0.05).** Every per-class recall has sd ≈ 0.20–0.24 — as large as
the means themselves. Each fine-tuning run collapses toward over-predicting **one** class, and *which*
class is essentially set by the seed. The most common landing point is **MCAR-collapse** (5 of 7 seeds
have MCAR ≥ 0.85, with MAR and MNAR sacrificed). This is the same optimizer/non-identifiability
instability Stage 3b found on self-censoring mixtures (the MAR↔MNAR split had sd ≈ 0.23), now seen
across **all three** classes once the full encoder is fine-tuned on the full registry.

## Answers to the three handoff questions

**(a) Does fine-tuning rebalance the frozen probe's cross-class bias (do MCAR & MAR recover)?**
No — not into a balanced model. The frozen probe was MNAR-biased (MNAR .81 / MCAR .22 / MAR .18,
overall ≈ .39). Fine-tuning raises overall accuracy to ≈ .60 and, *on average*, flips the bias from
MNAR-dominant to **MCAR-dominant** (MCAR .81 / MNAR .38 / MAR .29). But the average hides the truth:
no single seed is balanced. MCAR recovers in the sense that it becomes the easy default; MAR's mean
(.29) is barely above the frozen probe and is entirely attractor-driven (it is .54–.66 in the three
non-collapse seeds and .02–.07 in the MCAR-collapse seeds). So "rebalanced" is the wrong word — the
bias **moved and became seed-dependent**, it did not resolve.

**(b) Does the subtype stratification persist under balanced training?**
Within any single seed, yes — recall still ranges the full [0,1] across subtypes. But it does **not**
persist as a stable *absolute* per-subtype map across seeds: most generators swing by ±0.4 depending
on which class the seed's optimizer favored (e.g. MNAR-ThreshTwoSided seeds = [.97, 0, .12, .75, .09,
.33, 0]; MAR-Section = [.8, 1.0, 0, .8, .25, 1.0, 0]; MNAR-Truncation = [.04, 0, .14, .96, .8, .92,
.22]). The frozen probe's clean, sharp stratification is therefore **not** what a fine-tuned deployable
model delivers. What *does* survive across seeds is a small, genuinely seed-invariant set (below).

**(c) Does missing-by-design MAR (skip-logic, sections) recover from the frozen probe's 0.00?**
Conditionally yes. In the seeds that do not collapse MAR, structural/skip-logic MAR recovers strongly
(MAR-Section, -ColBlocks, -SkipLogic, -MultiCol reach 0.7–1.0 in seeds 1/2/4/6). This confirms the
prior doc's hypothesis that the frozen 0.00 was the MNAR-lean swamping it, not genuine undetectability.
But the recovery is **not reliable**: those same generators go to 0.00 in the MCAR-collapse seeds
(3/5/7). So missing-by-design MAR is *detectable in principle* but *not dependably detected* by this
fine-tuned model.

## The seed-invariant signal (the deployable core)

Generators appearing with n ≥ 20 in ≥ 4 of 7 seeds, classified by their min/max recall **across**
seeds — i.e. what holds regardless of which attractor a given training run lands in:

**Reliably DETECTED (min recall ≥ 0.5 across appearing seeds):** mostly MCAR.
- MCAR: Pareto (0.97±0.04), Bernoulli-30 (0.93±0.07), BatchFx (0.95), RowGauss (0.95), RowMixture
  (0.93), ColGauss (0.88), ColOrdered (0.82±0.18), ColClustered (0.78±0.19), RowColInt (0.87).
- MAR: RowBlocks / Distance / Kernel = 1.0 (but each appears in only 4–5 seeds — strong but thin).

**Reliably MISSED (max recall ≤ 0.35 across appearing seeds):**
- MAR: DemoGated-Sparse (0.01±0.03, 7/7), CrossClass (0.07±0.09, 6/7) — sparse / conditional MAR.

**No MNAR subtype is in either set.** Every MNAR generator's recall swings widely with the attractor
(class mean .38 ± .19, per-generator sds ≈ 0.3–0.4). So under fine-tuning, MNAR detectability is
entirely attractor-dependent — the opposite of the frozen probe, where threshold/detection/quantile
MNAR were the *most* reliably detected (0.9–1.0). **Fine-tuning destroyed the frozen encoder's sharp,
deployable MNAR-subtype footprint** (e.g. MNAR-Truncation: frozen 0.93 → fine-tuned 0.44 ± 0.40;
MNAR-ThreshTwoSided: frozen 0.98 → 0.32 ± 0.36).

## Interpretation

The frozen probe and the fine-tuned model answer different questions and disagree on the deployable
takeaway:

- **Frozen probe** (head + deployable features on a *fixed* baseline encoder): MNAR-biased overall,
  but with a **sharp, stable** per-subtype stratification — threshold/detection/quantile MNAR detected
  at 0.9–1.0, structural MAR swamped to 0. The stratification is trustworthy *because the encoder is
  fixed*; only the small head varies.
- **Fine-tuned model**: higher overall accuracy (.60 vs .39) but **unstable** — perturbing the whole
  encoder lets each run fall into a different winner-take-all attractor (usually MCAR, the plurality
  class by column count). The sharp subtype map dissolves into ±0.4 seed noise.

This is a non-identifiability × class-imbalance × full-model-fine-tuning interaction. MCAR is the
cheap attractor (most columns, easiest to fit by defaulting); MAR and MNAR sit on the non-identifiable
boundary with no stable gradient to anchor them (cf. Stage 3b), so the optimizer trades them off
near-randomly per seed. lr=1e-3 on the full encoder almost certainly amplifies this (the per-epoch CE
oscillated in [0.5, 0.9] all through training rather than converging) — a lower lr / frozen-encoder
regime is the natural follow-up, but changing it here would have broken single-variable attribution
against the pre-registered Stage 4 spec.

## Deployable product implication (revised)

The honest deployable spec is **not** "fine-tune for a balanced per-subtype map." It is:

- The **frozen probe** is the better deployable lens for the per-subtype *map* (stable, even if
  MNAR-leaning) — report its stratification with the MNAR-lean caveat.
- Under fine-tuning, the only seed-robust capability is **MCAR triage** plus a couple of strong-MAR
  forms; MNAR and most MAR are attractor-dependent and must be reported as **calibrated-uncertain**,
  not confidently flagged.
- If a fine-tuned model is shipped, it needs **ensembling/seed-averaging or a stability fix** (lower
  lr, frozen encoder, class-balanced loss, or explicit calibration) before any per-class number is
  trustworthy. A single fine-tuned seed is not reportable (seed 1 alone would have told a triumphant
  "MAR recovers, MNAR holds" story that seeds 3/5/7 demolish).

## Critical caveats

- **Seed variance is the result.** Do not read the mean per-class recalls as point estimates; the
  distribution (sd ≈ 0.2, range ≈ 0.7 wide) is the finding. n = 7 establishes the instability robustly
  but the *mean* of an unstable quantity is still only loosely estimated.
- **Single-mechanism eval, not mixtures.** All missing columns in a dataset share one mechanism (the
  Stage-4 design). Not directly comparable to the Stage 0–3 per-column-mixture numbers.
- **Eval-sampling noise is small relative to the effect.** Per-generator n's are in the hundreds
  (binomial se ≈ 0.03); the observed 0→1 swings are ~10× larger, so the cross-seed variance is
  training-attractor, not eval sampling.
- **lr / fine-tune-depth not ablated** here (held at the pre-registered Stage 4 settings for
  attribution). They are the obvious next variable.
- Current-code baseline encoder (10-feature gate), deployable per-column features (5).

## Files

- Per-seed: `runs/stage0_general_baseline/stage4_subtype_finetune_s{1..7}.json`
- Aggregate (per-class & per-generator mean ± sd): `runs/stage0_general_baseline/stage4_subtype_finetune_aggregate.json`
- Frozen-probe reference: `runs/stage0_general_baseline/stage4_subtype_frozen.json`
