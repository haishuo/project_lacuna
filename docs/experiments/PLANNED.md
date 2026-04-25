# Planned experiments

Current queue, in priority order. Each entry is a single experiment;
when it runs, its contents move into a new dated file
(`YYYY-MM-DD-short-slug.md`) with results filled in and this entry
gets deleted from here. At any moment, this file reflects the current
queue.

---

## State of play (as of 2026-04-25)

**The ablation arc is closed.** The two committee-grade n=30
experiments — `n30-mle-vs-disable` (2026-04-23) and
`full-5spec-canonical-n30` (2026-04-25) — together pin the answer:

- The cached MLE Little's slot **hurts** mechanism classification
  (Δ = +0.029 in favour of removing it, CI [+0.002, +0.054]).
- The non-MCAR feature groups (`missing_rate_stats` +
  `cross_column_corr`) **contribute** ~6 pp accuracy when present
  (Δ = +0.059 vs `all_disabled`, CI [+0.021, +0.095]).
- The bakeoff (2026-04-21) screened six MCAR-test families
  (parametric MLE, MoM, revived heuristic, propensity-AUC, HSIC,
  MissMech); none clears the advancement bar at n=10.

ADR 0004 (drafted 2026-04-25) codifies the architectural
implication: remove the MLE slot, keep the rest of the pipeline.
Code changes implementing ADR 0004 are scoped to a separate session
— this PLANNED.md no longer queues training runs, only the
deferred / exploratory items below remain notional.

The dissertation methods-section ablation table is the per-spec
summary in `2026-04-25-canonical-n30.md`.

---

## (No queued experiments)

The pre-defense ablation arc is complete. The remaining items below
are exploratory and not on any timeline.

---

## Deferred / exploratory (not queued)

- **`root-cause-no-mcar-helps`** — dedicated analysis session. The
  bakeoff showed all six MCAR families non-contributory, including
  the revived median-split SMD heuristic that was contributing at
  the same slot per ADR 0001. The canonical confirmed the rest of
  the pipeline DOES help, so the puzzle is now sharper: why is the
  MCAR slot specifically non-contributory when the simpler missingness-
  pattern groups carry +6 pp on the same architecture? Candidate
  paths:
    - Re-run the 2026-04-17 `feature-group-ablation` at n=30 to test
      whether the +6 % MCAR-slot contribution at n=5 was small-sample
      noise.
    - Interaction-effect sweep: re-add the pointbiserial and
      distributional groups (removed by ADR 0001) alongside each MCAR
      variant, testing whether the MCAR slot ever contributed only
      via interaction with features we've since removed.
    - Architectural probe: inspect MoE gating weights on a converged
      `disable_littles` model — does the router put zero weight on
      whatever-was-in-the-MCAR-slot regardless of contents?
  Not blocking the dissertation. Good material for a methodological
  discussion chapter or post-defense paper.

- **`per-group-decomposition`** — two new specs
  (`disable_littles_and_missing_rate`, `disable_littles_and_cross_column`)
  at n=30 to cleanly attribute the canonical's +6-pp joint
  contribution to one or both of the non-MCAR groups individually.
  Optional. The headline finding (pipeline contributes, MCAR slot
  doesn't) is established without this; it would only refine the
  narrative.

- **`sentinel-subset-analysis`** — was conditional on
  confirming "MLE hurts" cleanly. The canonical does that, so this is
  no longer load-bearing for the dissertation. Could still inform the
  root-cause story (is the harm driven by the 16.4 % MLE-sentinel
  rate, or by the feature even on real values?). Defer indefinitely.

- **`neural-mcar-detector`** — Muzellec et al. 2020 Sinkhorn
  divergence as an MCAR feature. Was the escalation path if simpler
  alternatives all failed; given that the architectural slot itself
  is non-contributory at n=10 across six families, a more complex
  neural detector is unlikely to succeed where simpler ones failed
  equally. Formally rule out for now.

- **`cross-registry-generalization`** — train on `lacuna_tabular_110`
  minus one generator family, eval on the held-out family. Methods-
  paper material; not load-bearing for the dissertation ablation
  story.
