# Stage 5 — Full-diversity TRUE mixtures: a deployment-grade per-column estimate

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006. Follow-up to Stage 4b (`2026-05-29-stage4b-diverse-mnar-mixtures.md`).
- **Script:** `scripts/stage5_true_mixtures.py` (reproducible runner; Stage 4b was hand-driven).
- **Question:** Stage 4b showed the per-column estimand is viable and *stable* once the mixture's
  MNAR diversity matches reality — but with only an 8-subtype MNAR pool AND a single MAR family
  (`MARLogistic`). Does the stable, deployable, frozen-probe per-column classifier hold up at
  **FULL subtype diversity on TRUE mixtures** — diverse MAR subtypes AND diverse MNAR subtypes
  coexisting across the columns of one dataset? That balanced multi-seed number is the
  deployment-grade headline.

## Design (one diversity axis added at a time)

The lens is the one the arc established as deployable (Stage 3a/4): a **frozen** baseline encoder +
per-column readout head + **deployable** distributional features (observed-values-only; no
fine-tune — Stage 4 showed fine-tuning collapses seed-dependently — and no oracle). Train AND eval
on the **same** arm distribution (matched regime ⇒ a fair deployment estimate, not a ceiling).
6 seeds, 15 epochs, frozen probe. The **only** variable changed between arms is which diversity
axis is switched on:

| arm | MNAR columns | MAR columns | = prior stage |
|---|---|---|---|
| **monoculture** | self-censoring only | logistic only | Stage 0–3 control |
| **diverse_mnar** | 25-subtype pool | logistic only | Stage 4b, expanded pool |
| **full_diversity** | 25-subtype pool | 8-subtype pool | **the Stage 5 headline** |

Enablers (committed first): per-column `target_col_idx` on **all** MNAR families (bit-identical
defaults, guarded by a registry hash regression test); a curated 25-subtype diverse MNAR pool
(`mnar_column_pool`, own-value clean-MNAR only) and a new 8-subtype diverse MAR pool
(`mar_column_pool`, single clean predictor); `compose_mixed_missingness` gained `mar_diverse`;
`MixedBatch` now carries the realised per-column subtype so eval can break detectability down by
subtype. Per-column miss-rate held ≈ 0.25 across all three classes in every arm (confound control;
see realised rates below).

## Result (frozen-probe + deployable features, n = 6 seeds; mean ± sd)

| metric | monoculture | diverse_mnar | **full_diversity** |
|---|---|---|---|
| per-column accuracy | ‹MONO_ACC› | ‹DM_ACC› | **‹FD_ACC›** |
| **MNAR recall** | ‹MONO_MNAR_R› | ‹DM_MNAR_R› | **‹FD_MNAR_R›** |
| **MNAR precision** | ‹MONO_MNAR_P› | ‹DM_MNAR_P› | **‹FD_MNAR_P›** |
| **MAR recall** | ‹MONO_MAR_R› | ‹DM_MAR_R› | **‹FD_MAR_R›** |
| MAR precision | ‹MONO_MAR_P› | ‹DM_MAR_P› | ‹FD_MAR_P› |
| MCAR recall | ‹MONO_MCAR_R› | ‹DM_MCAR_R› | ‹FD_MCAR_R› |
| per-column ECE | ‹MONO_ECE› | ‹DM_ECE› | ‹FD_ECE› |
| composition L1 | ‹MONO_L1› | ‹DM_L1› | ‹FD_L1› |

Per-seed distributions (the distribution is the result, per the project's rigor standard):

```
MNAR recall  monoculture:    ‹MONO_MNAR_SEEDS›
MNAR recall  diverse_mnar:   ‹DM_MNAR_SEEDS›
MNAR recall  full_diversity: ‹FD_MNAR_SEEDS›
MAR  recall  monoculture:    ‹MONO_MAR_SEEDS›
MAR  recall  diverse_mnar:   ‹DM_MAR_SEEDS›
MAR  recall  full_diversity: ‹FD_MAR_SEEDS›
```

Confound check — realised per-class miss rate (mean over seeds), target 0.25:

```
                MCAR        MAR         MNAR
monoculture     ‹MONO_MR›
diverse_mnar    ‹DM_MR›
full_diversity  ‹FD_MR›
```

## Per-subtype detectability (full_diversity arm, mean ± sd over seeds, n ≥ 20)

‹PER_SUBTYPE_TABLE›

## Findings

‹FINDINGS›

## Comparison against the prior stages

- **(a) vs the self-censoring monoculture (Stage 0–3 control = this run's `monoculture` arm).**
  ‹CMP_MONO›
- **(b) vs single-mechanism-diversity (Stage 4b = this run's `diverse_mnar` arm, now at the
  expanded 25-subtype pool).** ‹CMP_4B›

## Interpretation

‹INTERPRETATION›

## Critical caveats

- **Matched train+eval regime.** Each arm trains on the same MNAR/MAR diversity it is tested on —
  the right question for "is per-column classification inherently hard, or was the monoculture the
  problem." It is **not** a train-on-X / deploy-on-Y mismatch claim (a separate question).
- **Still single-mechanism-per-column, clean-MAR, semi-synthetic.** Real data has no per-column
  ground truth (ADR-0006). The pools are curated to clean per-column mechanisms: MNAR = 25 own-value
  subtypes (row/other-column/sequence/latent subtypes excluded — they have the targeting capability
  but would inject MAR-adjacent label noise); MAR = 8 single-clean-predictor subtypes. This is
  "near-full" diversity, not literally all 41 MNAR / 40 MAR registry generators.
- **Frozen probe, no fine-tune, no oracle.** Current-code 10-feature baseline encoder; deployable
  features (5). Fine-tuning was shown (Stage 4) to re-introduce the seed-dependent collapse this
  regime avoids, so it is deliberately not used.
- **n = 6 seeds. Report the distributions, not the means.**

## Files

- Runner: `scripts/stage5_true_mixtures.py`
- Per-run: `runs/stage0_general_baseline/stage5_true_mixtures/{monoculture,diverse_mnar,full_diversity}_s{1..6}.json`
- Aggregate: `runs/stage0_general_baseline/stage5_true_mixtures/aggregate.json`
