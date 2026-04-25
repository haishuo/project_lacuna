# Evidence standard — retrospective and going-forward

This document records a methodological reflection, prompted on
2026-04-20 after the partial-data review of the `n10-followup`
experiment. It is not an ADR (those are decisions) and not a completed
experiment (no sweep was run for this). It's a note-to-self about
statistical rigor, written down so future-me and any committee reader
can see that the limitation was acknowledged rather than hidden.

## The concern

The first several Lacuna ablation decisions were made on n=5 paired
data. That sample size was chosen because each run costs ~22 min and
the full 7-spec sweep at n=5 is already 13 hours. At n=5, paired
Wilcoxon's minimum two-sided achievable p-value is 0.0625 — we could
never reach p<0.05 by rank test alone. Bootstrap CIs at n=5 on effect
sizes around |Δ|~0.05 with per-seed SD ~0.07 produce 95% CIs that are
extremely sensitive to individual seed draws: a single unfavorable run
swings the CI endpoint by ~0.03-0.05, enough to cross zero.

In practice, this means several "the effect is real / not real"
judgments were made at the zero-crossing boundary:

- **ADR 0001 (2026-04-18) — `disable_littles` with the heuristic:**
  Δ=−0.064, CI [−0.169, **−0.008**]. Just excluded zero. Wilcoxon
  p=0.058. Treated as "contributory."
- **ADR 0001 — `disable_pointbiserial` and `disable_distributional`:**
  CIs [−0.031, +0.077] and [−0.078, +0.046]. Span zero with ~equal
  positive and negative reach. Treated as "non-contributory." The
  honest statement was "no measurable contribution detected at n=5,"
  not "no contribution exists."
- **2026-04-19 sweep — `disable_littles` with real Little's:**
  Δ=+0.061, CI [−0.030, +0.120]. Just included zero. Treated as
  "suggestive of harmful."

The ADR 0001 heuristic result and the 2026-04-19 real-Little's result
are mirror images — same slot, comparable effect sizes in opposite
directions, both CIs tangent to zero. Neither can be trusted as the
final word.

The partial `n10-followup` data (n=8 at time of reflection) confirmed
the instability: the +0.061 effect shrank to +0.046 with a wider CI
that clearly includes zero. Three more seeds were enough to materially
weaken the conclusion. This is what a true borderline effect looks
like under resampling.

## What this means for already-made decisions

- **ADR 0001's removal of `pointbiserial` and `distributional`** was
  defensible on engineering grounds (simpler code, fewer moving parts)
  regardless of statistical decisiveness. The code is in a cleaner
  state. But the claim that those features are *proven non-contributory*
  was stronger than the n=5 evidence supports. The correct claim is
  "no contribution measurable at n=5; removed for simplicity; could be
  revisited if a future experiment suggests otherwise."
- **ADR 0002's upgrade from heuristic to cached real Little's** was
  motivated by "the slot is contributory per ADR 0001, so upgrade it to
  a defensible statistic." The motivation chain is only as strong as
  the ADR 0001 n=5 evidence. The upgrade itself (caching real Little's
  per dataset-generator pair) is a clean piece of engineering that
  remains valuable even if the downstream ablation says the feature
  isn't useful — it at least means we're comparing the right thing.
- **ADR 0003 (dual-method cache)** is similarly engineering-motivated
  (one cache, two methods, no extra cost) and doesn't depend on
  statistical claims from ADR 0001 being exactly right.

None of these need to be re-done. The code is fine. What needs
updating is the *language* used in the dissertation and committee
presentations: "provisionally concluded at n=5" rather than
"demonstrated."

## Going forward — n=30 as the rigor floor

For any remaining question whose answer would materially affect
dissertation claims, the standard is:

1. **n=30 paired seeds minimum** unless a specific argument justifies
   less. At n=30, bootstrap CI half-widths tighten by a factor of
   ~2.45 over n=5, and Wilcoxon's combinatorial headroom reaches
   p<<0.001. The sample size is where CLT-based approximations become
   trustworthy and borderline effects resolve.
2. **Pre-register the decision rule** before running the sweep, in the
   planned-experiment file. "CI excludes zero → verdict A; CI
   includes zero → verdict B" written down in advance, not
   post-hoc adjusted.
3. **Report both point estimates and CIs** in the dissertation text —
   CIs, not p-values, are the primary decision instrument given the
   small sample sizes typical of ML ablation.
4. **Flag provisional findings as provisional.** Any n≤10 conclusion
   is labelled as such. Any n=30 conclusion is labelled as definitive.

The first n=30 sweep to apply this standard is
`n30-1v1-littles-slot` in `PLANNED.md`, contingent on the n=10
followup being inconclusive. Future questions — if the committee
discussion identifies them — should default to n=30 paired runs unless
otherwise argued.

## What this isn't

This document doesn't claim the previous decisions were *wrong* — only
that they were made on insufficient evidence to be *certain*. The
decisions may well be right. We'll know for sure as subsequent n=30
sweeps either confirm or overturn them. In the meantime, the honest
statement is "provisional at n=5; update when stronger evidence
arrives."

## Specific revival risk assessment (2026-04-20 update)

When asked whether the deletions in ADR 0001 and ADR 0002 should be
reconsidered in light of this evidence standard, the answer
differentiates by feature based on the direction of the n=5 point
estimate:

| Feature | n=5 Δ | Direction | Revival priority |
|---|---:|:---|:---|
| `pointbiserial` | +0.018 | Variant was *better* than baseline — removal helped | Low (would re-delete) |
| `distributional` | −0.007 | Essentially zero in either direction | Low (no effect detected) |
| `median-split SMD` (the Little's heuristic) | −0.064 | Variant was *worse* — heuristic helped | **High** |

The median-split SMD heuristic is the one where the n=5 evidence
actually pointed toward "helpful," and it was then replaced (not just
deleted) by a feature (real cached Little's) that subsequently failed
to reproduce that benefit. This is the scenario where revival matters
most: we may have regressed a useful feature while attempting to
upgrade it. The heuristic is therefore added to the
`mcar-alternatives-bakeoff` experiment (PLANNED.md §5) as a
first-class candidate alongside the nonparametric / ML alternatives.

Pointbiserial and distributional are not being revived pending that
bakeoff's outcome, because:

- Their n=5 point estimates did not favor keeping them (pointbiserial
  was actively +0.018 in the *removal helps* direction).
- Their absence is reversible cheaply later if a future finding
  motivates revisiting.
- Including them now would cost 2 additional specs × whatever seed
  count × wall-clock, with low expected scientific payoff.

## Cross-references

- Experiments: `2026-04-17-feature-group-ablation.md`,
  `2026-04-19-mle-vs-mom.md`, `2026-04-19-n10-followup.md`
- ADRs: `docs/decisions/0001`, `0002`, `0003` — all predicated on
  n=5 evidence to varying degrees
- Follow-up planned: `PLANNED.md` §1 (`n30-1v1-littles-slot`)
