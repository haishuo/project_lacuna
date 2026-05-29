# Stage 4b — Diverse-MNAR per-column mixtures lift (and stabilise) the hard cases

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006. Follow-up to Stage 4 (`2026-05-29-stage4-fulltune.md`).
- **Question:** Stages 0–3 concluded "per-column MNAR is hard on mixtures" and Stages 3b/4 found the
  MAR↔MNAR split is seed-unstable. Every one of those mixtures realised MNAR columns as **logistic
  self-censoring** — plausibly the *hardest* MNAR subtype (own-value dependence = core
  non-identifiability) and, until now, the only MNAR family that supported per-column targeting.
  Does the conclusion lift once mixtures contain the *easy* MNAR subtypes (threshold, detection
  limit) too?
- **Enabler:** `target_col_idx` added to the threshold/detection MNAR families (commit d0d3c2d) +
  a diverse per-column MNAR pool (`lacuna/data/mnar_column_pool.py`, commit 676d137). MNAR columns
  now draw a subtype from {self-censoring, threshold L/R/two-sided, soft-threshold, detection
  L/U/both}, each targeting its column at ~`target_miss_rate`.
- **Design (one variable):** identical config — frozen baseline encoder + per-column head +
  deployable features (the cleanest, best-calibrated lens from Stage 3a) — run with
  self-censoring-only mixtures (**control**) vs diverse-MNAR mixtures (**treatment**), train AND
  eval in the matched regime. 6 seeds each (`--seed 1..6`), 15 epochs. The **only** change between
  arms is the MNAR subtype mix; MCAR/MAR generation, labels, miss-rate target, p_observed, and the
  head are all held fixed. Script: `scripts/stage1_column_head.py --diverse-mnar`.

## Result (frozen-probe + deployable features, n = 6 seeds; mean ± sd)

| metric | self-censoring (control) | diverse-MNAR (treatment) | Δ mean |
|---|---|---|---|
| per-column accuracy | 0.486 ± 0.043 | **0.590 ± 0.028** | **+0.104** |
| **MNAR recall** | 0.391 ± **0.304** | **0.534 ± 0.107** | +0.143 |
| **MNAR precision** | 0.333 ± 0.153 | **0.600 ± 0.084** | +0.267 |
| **MAR recall** | 0.348 ± **0.281** | **0.699 ± 0.051** | +0.351 |
| MCAR recall | 0.721 ± 0.134 | 0.530 ± 0.162 | −0.190 |
| per-column ECE | 0.059 ± 0.018 | 0.081 ± 0.036 | +0.022 |
| composition L1 | 0.448 ± 0.034 | 0.436 ± 0.046 | −0.012 |

Per-seed (the distribution is the point):

```
MNAR recall  control:   [0.28, 0.14, 0.78, 0.34, 0.00, 0.81]   sd 0.30
MNAR recall  treatment: [0.38, 0.66, 0.54, 0.65, 0.41, 0.56]   sd 0.11
MAR  recall  control:   [0.18, 0.61, 0.03, 0.52, 0.73, 0.02]   sd 0.28
MAR  recall  treatment: [0.65, 0.71, 0.65, 0.66, 0.74, 0.78]   sd 0.05
```

## Findings

1. **The "per-column MNAR is hard on mixtures" conclusion LIFTS.** MNAR recall rises 0.39 → 0.53 and
   — more importantly — its **seed variance collapses ~3×** (sd 0.30 → 0.11). The self-censoring
   control swings from 0.00 to 0.81 across seeds; the diverse-MNAR treatment is stable in [0.38,
   0.66]. MNAR *precision* also rises sharply (0.33 → 0.60): fewer spurious MNAR calls.

2. **The MAR↔MNAR seed-instability — the antagonist of the entire arc — largely DISSOLVES.** Under
   self-censoring, MAR recall is wildly unstable (sd 0.28) and anti-correlated with MNAR across
   seeds (the boundary mass lands near-randomly each run; cf. Stage 3b/4). Under diverse MNAR, MAR
   recall is high *and* rock-stable (0.699 ± 0.051). With genuinely detectable MNAR present, the
   optimizer no longer dumps MAR mass into the MNAR confusion — the non-identifiable boundary that
   made every prior result seed-dependent stops dominating.

3. **Overall accuracy improves (+0.10) and composition L1 is unchanged/slightly better.** The gain
   is real, not a reshuffle: MNAR and MAR both rise, paid for partly by MCAR (0.72 → 0.53, and
   noisier) — the model stops defaulting to the easy plurality class once the other two are
   learnable.

4. **The gain is attributable to mechanism TYPE, not missing QUANTITY (confound checked).** Realised
   per-class miss rates: self-censoring MNAR columns miss at **0.315**, diverse-MNAR MNAR columns at
   **0.262** (target 0.25) — i.e. the diverse columns go missing *slightly less*, yet are detected
   *better*. So the effect is not "diverse MNAR is missing more"; threshold/detection mechanisms
   leave a sharper observable distributional footprint than self-censoring at equal-or-lower
   miss rate. (MCAR ≈ 0.25, MAR ≈ 0.31 in both arms — unchanged.)

## Interpretation — this reframes Stages 0–4

The recurring "per-column MNAR is non-identifiability-bound" / "the MAR-MNAR split won't stabilise"
results were, to a large degree, an **artefact of the generator monoculture**: training and
evaluating on mixtures whose every MNAR column was the single hardest subtype (self-censoring). That
is the same lesson the project learned at the dataset level (under-diversity of generators tanks
results), now confirmed at the column level and for *stability*, not just accuracy. With a realistic
MNAR subtype mix:

- per-column MNAR is **moderately and reliably** detectable (≈ 0.53, sd 0.11), not stuck near the
  self-censoring floor;
- per-column MAR becomes **reliably** detectable (≈ 0.70, sd 0.05);
- the deployable frozen-probe + distributional-features lens is enough — **no fine-tuning, no
  oracle.** (And per Stage 4, fine-tuning would have re-introduced the instability this regime
  removes.)

This is the strongest, cleanest result of the column-level arc: the per-column estimand is viable
and *stable* once the mixture's mechanism diversity matches reality.

## Critical caveats

- **Train+eval in the matched regime.** This measures the achievable ceiling when the model is
  trained on the same MNAR diversity it is tested on — the right question for "is per-column MNAR
  inherently hard, or was the monoculture the problem." It is **not** a claim about a model trained
  on self-censoring and deployed on diverse data (train/test mismatch is a separate question).
- **Still single-mechanism-per-column mixtures, clean-MAR regime, semi-synthetic.** Real data has no
  per-column ground truth (ADR-0006). The diverse MNAR pool is 8 subtypes — broader than 1, but not
  the full ~41-generator MNAR registry (only the per-column-targetable families).
- **MCAR cost and a small ECE rise** (0.059 → 0.081) are the price; both are modest and the net
  (accuracy, MNAR/MAR recall, MNAR precision, composition) is favourable.
- n = 6 seeds. Report the distributions, not the means (the control's sd ≈ 0.3 is itself the prior
  result). Frozen-probe; current-code 10-feature baseline encoder; deployable features (5).

## Files

- Per-run: `runs/stage0_general_baseline/diverse_mnar_exp/{selfcensor,diversemnar}_s{1..6}.json`
- Aggregate: `runs/stage0_general_baseline/diverse_mnar_exp/aggregate.json`
