# Stage 5 — Full-diversity true mixtures expose a miss-rate confound behind the Stage 4b stability

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006. Follow-up to Stage 4b (`2026-05-29-stage4b-diverse-mnar-mixtures.md`).
- **Script:** `scripts/stage5_true_mixtures.py` (reproducible; Stage 4b was hand-driven).
- **Question (as posed):** Stage 4b concluded the per-column estimand is viable and *stable* once
  mixture MNAR diversity matches reality — but with only an 8-subtype MNAR pool and a single MAR
  family (`MARLogistic`). Does the stable, deployable, frozen-probe per-column classifier hold up at
  **FULL subtype diversity on TRUE mixtures** (diverse MAR *and* diverse MNAR across columns)?
- **Short answer:** **No — and pursuing *why* uncovered that the Stage 4b stability was substantially
  a MISS-RATE CONFOUND.** Once per-column miss rate is held *truly* constant across all three
  mechanisms (the proper confound control), the frozen-probe per-column classifier is **seed-unstable
  across every diversity regime** — monoculture, diverse-MNAR, and full-diversity alike. The
  per-column MAR↔MNAR↔MCAR non-identifiability the arc kept hitting reasserts itself the moment the
  spurious miss-rate cue is removed.

## Design

Lens (the arc's deployable one, Stage 3a/4): **frozen** baseline encoder + per-column readout head +
**deployable** distributional features (observed-values-only; no fine-tune, no oracle). Train AND
eval on the SAME arm distribution (matched regime). 6 seeds, 15 epochs. Three arms, one diversity
axis added at a time:

| arm | MNAR columns | MAR columns | = prior stage |
|---|---|---|---|
| **monoculture** | self-censoring only | logistic only | Stage 0–3 control |
| **diverse_mnar** | 25-subtype pool | logistic only | Stage 4b (expanded pool) |
| **full_diversity** | 25-subtype pool | 8-subtype pool | the Stage 5 target |

### The confound we found (and controlled)

While building the runner we noticed the composer's **direct** logistic-MAR and self-censoring-MNAR
paths set `intercept = logit(target_miss_rate)`, but the logistic slope inflates the marginal so
those columns realise **≈ 0.30** for a 0.25 target — whereas **MCAR Bernoulli is exact (0.25)** and
the compensated diverse **pools** also hit ≈ 0.25. So throughout Stages 0–4b, **MCAR columns missed
at 0.25 but MAR/MNAR columns missed at ≈ 0.30** — and `missing_rate` is the *first* deployable
feature. The per-column classifier could (and did) use that ~0.05 gap as a cue.

We added `compensate_rate` (default off = bit-identical legacy behaviour) so the direct paths use a
slope-compensated intercept and hit ≈ target. Stage 5 runs **both** regimes:

- **compensated** (default, all mechanisms ≈ 0.25): the proper confound-controlled result.
- **legacy-rate** (`--legacy-rate`; MCAR 0.25 / MAR,MNAR ≈ 0.30): reproduces the Stage 0–4b regime.

## Result A — compensated regime (all mechanisms ≈ 0.25; frozen probe; mean ± sd, n = 6)

| metric | monoculture | diverse_mnar | full_diversity |
|---|---|---|---|
| per-column accuracy | 0.357 ± 0.045 | 0.362 ± 0.033 | 0.327 ± 0.015 |
| MCAR recall | 0.438 ± **0.304** | 0.638 ± **0.408** | 0.513 ± **0.429** |
| MAR recall | 0.220 ± 0.143 | 0.257 ± **0.303** | 0.299 ± **0.366** |
| MNAR recall | 0.421 ± **0.324** | 0.176 ± 0.197 | 0.181 ± **0.318** |
| per-column ECE | 0.035 ± 0.014 | 0.021 ± 0.015 | 0.032 ± 0.018 |
| composition L1 | 0.476 ± 0.041 | 0.473 ± 0.038 | 0.469 ± 0.043 |
| realised miss rate (MCAR/MAR/MNAR) | 0.249/0.239/0.238 | 0.248/0.239/0.231 | 0.249/0.251/0.232 |

Per-seed (the distribution is the result — every arm is winner-take-all unstable):

```
                 MCAR per-seed                         MAR per-seed                          MNAR per-seed
monoculture      [.94, .29, .56, .13, .64, .07]        [.36, .32, .00, .06, .36, .22]        [.02, .41, .49, .90, .02, .69]
diverse_mnar     [.82, .10, .88, .99, 1.0, .03]        [.28, .41, .00, .01, .01, .84]        [.13, .58, .20, .00, .00, .14]
full_diversity   [1.0, .02, .99, .17, .81, .08]        [.02, .92, .00, .69, .14, .02]        [.00, .04, .01, .08, .07, .89]
```

Every arm: per-class recall sd ≈ 0.14–0.43, accuracy ≈ 0.33–0.36, and each seed collapses toward
whichever class its optimiser run favours (MCAR↔MAR↔MNAR anti-correlated across seeds). **No arm is
stable.** This is the same winner-take-all collapse Stage 4 saw under *fine-tuning* — now in the
*frozen probe* once the miss-rate cue is gone.

## Result B — legacy-rate regime (MCAR 0.25 / MAR,MNAR ≈ 0.30; the Stage 0–4b regime; n = 6)

| metric | monoculture | diverse_mnar | full_diversity |
|---|---|---|---|
| per-column accuracy | 0.502 ± 0.039 | 0.521 ± 0.040 | 0.341 ± 0.019 |
| MCAR recall | 0.691 ± 0.141 | 0.510 ± 0.271 | 0.385 ± 0.346 |
| MAR recall | 0.492 ± 0.151 | **0.666 ± 0.085** | 0.358 ± 0.321 |
| MNAR recall | 0.306 ± 0.138 | 0.388 ± 0.215 | 0.259 ± 0.224 |
| realised miss rate (MCAR/MAR/MNAR) | 0.249/0.303/0.302 | 0.248/0.302/0.231 | 0.249/0.251/0.232 |

Here `monoculture` (MCAR the only 0.25 class, MAR+MNAR at 0.30) and `diverse_mnar` (MAR the only
0.30 class) look *much* better and *more stable* than in Result A — `diverse_mnar` MAR is 0.666 **±
0.085**, reproducing the Stage 4b headline. But note `full_diversity`: it realises both mechanisms
through the *compensated pools*, so its columns are already ≈ 0.25 even in this "legacy" regime — it
is the one arm with **no** miss-rate cue, and it is **already unstable here** (acc 0.341, recall sd
0.22–0.35), essentially identical to its Result-A self. That is the tell.

## The decisive comparison — equalising miss rate removes the crutch

Same pool, same script, same seeds (current code); only the direct-path intercept changes
(≈0.30 → ≈0.25):

| arm | metric | legacy-rate (cue) | compensated (no cue) | Δ |
|---|---|---|---|---|
| monoculture | MCAR recall | 0.691 ± 0.141 | 0.438 ± 0.304 | **−0.25** (and destabilises) |
| monoculture | accuracy | 0.502 | 0.357 | −0.15 |
| diverse_mnar | MAR recall | **0.666 ± 0.085** | **0.257 ± 0.303** | **−0.41**, sd 0.085 → 0.303 |
| diverse_mnar | accuracy | 0.521 | 0.362 | −0.16 |
| **full_diversity** (control) | accuracy | 0.341 | 0.327 | **−0.01 (no-op)** |
| **full_diversity** (control) | MAR recall | 0.358 ± 0.321 | 0.299 ± 0.366 | ≈0 (already cue-free) |

Two things make this airtight:
1. **The collapsing cells are exactly the cue-bearing ones.** MCAR is the only 0.25 class in
   `monoculture`; MAR is the only 0.30 class in `diverse_mnar`. Those — and only those — carry a
   clean, stable, *inflated* recall in the legacy regime, and those are exactly what collapse when
   the rate is equalised (`diverse_mnar` MAR: stable 0.67 → unstable 0.26).
2. **`full_diversity` is the built-in control.** It never used the direct paths (both mechanisms go
   through the compensated pools), so `compensate_rate` is a literal no-op for it — and indeed its
   numbers barely move (Δacc 0.01, within the ≈0.04 GPU-nondeterminism floor). So compensation is
   **not** a generic "make everything worse" knob; it removes a cue from precisely the arms that had
   one and leaves the cue-free arm untouched.

## Per-subtype detectability

In the compensated regime the per-subtype recalls are **collapse-dominated** (each subtype inherits
its seed's winner-take-all outcome ⇒ mean ≈ 0.1–0.3 with sd ≈ 0.2–0.4 for essentially every MNAR and
MAR subtype). There is **no** stable per-subtype map at matched miss rate — the sharp frozen-probe
stratification reported in Stage 4 (single-mechanism) does not survive into matched-rate mixtures.
Full per-subtype tables: `aggregate.json`.

## Interpretation — this reframes Stages 0–4b

1. **Stage 4b's "stable, deployable per-column classifier" was substantially a miss-rate artefact.**
   Its diverse-MNAR arm kept MAR columns at ≈ 0.30 while MNAR/MCAR sat at ≈ 0.25; the `missing_rate`
   feature let the head carve off MAR cheaply (MAR recall 0.70), which in turn anchored the rest.
   Remove the gap and MAR recall falls to ≈ 0.26 and the whole 3-way decision destabilises.
2. **The cleanest internal evidence was already in the legacy run.** `full_diversity` realises both
   mechanisms through the *compensated* pools, so it never had the cue — and it was *already*
   unstable in the legacy regime (acc ≈ 0.36, recall sd ≈ 0.2–0.42), exactly matching every arm in
   the compensated run. The cue, not the diversity, was what separated "stable" arms from "unstable".
3. **Per-column MAR↔MNAR↔MCAR is non-identifiability-bound at matched rate, with the current
   deployable features.** The observed-value distributional features (skew/kurtosis/SMD-to-others)
   plus a frozen encoder are not enough to stably separate the three mechanisms per column once the
   miss-rate shortcut is gone; the small head collapses to a seed-dependent attractor (cf. the Stage 4
   fine-tuning instability — same pathology, now exposed in the probe).
4. **What is NOT overturned:** mechanism *type* still matters where it leaves a genuine observable
   footprint (Stage 4 single-mechanism, no mixture, each generator at its own rate, showed
   threshold/detection MNAR detectable). Stage 5's claim is narrower and specific to the *mixture*
   setting with *matched* per-column rate: there, the deployable frozen probe is unstable.

## What would be needed (future)

- **Richer MAR-axis features** beyond a single SMD-to-others statistic (multiple predictor-coupling
  measures, nonlinear), so MAR is detectable on signal rather than on rate.
- **A stability fix** for the readout: class-balanced loss, lower lr, seed-ensembling, or an explicit
  calibrated-abstention class — before any per-class number at matched rate is trustworthy.
- **Honest miss-rate handling:** since real columns genuinely differ in miss rate, miss-rate *is*
  legitimate signal — but it must be reported as such, not allowed to masquerade as mechanism
  detection. The matched-rate result is the conservative lower bound.

## Critical caveats

- **6 seeds; the distribution (sd) is the result, not the mean.** The instability is robust (every
  arm, every class, winner-take-all per seed); the *means* of unstable quantities are loosely estimated.
- **GPU training is nondeterministic** (no `use_deterministic_algorithms`); run-to-run drift on the
  cue-free arm is ≈ 0.04, an order of magnitude below the confound effects reported.
- **Matched-rate is the conservative regime.** It deliberately removes a cue that real data *does*
  contain; the honest deployment number lies between this lower bound and the (confounded) Stage 4b
  upper bound, and depends on how much real per-column miss-rate variation is mechanism-informative.
- Still single-mechanism-per-column, clean-MAR, semi-synthetic; frozen probe; current-code 10-feature
  baseline encoder; deployable features (5). Pools are curated to clean per-column mechanisms
  (25 own-value MNAR, 8 single-clean-predictor MAR) — "near-full", not all 41/40 registry generators.

## Files

- Runner: `scripts/stage5_true_mixtures.py`
- Compensated: `runs/stage0_general_baseline/stage5_true_mixtures/{arm}_s{1..6}.json` + `aggregate.json`
- Legacy-rate: `runs/stage0_general_baseline/stage5_legacy_rate/{arm}_s{1..6}.json` + `aggregate.json`
