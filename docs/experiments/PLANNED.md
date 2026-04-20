# Planned experiments

Current queue, in priority order. Each entry is a single experiment; when
it runs, its contents move into a new dated file
(`YYYY-MM-DD-short-slug.md`) with results filled in and this entry gets
deleted from here. At any moment, this file reflects the current queue.

---

## 1. `sentinel-subset-analysis` — Is the `disable_littles` win driven by MLE's 16.4% sentinel rate?

- **Status:** queued (run after `n10-followup` if that confirms the effect)
- **Motivation:** The `disable_littles` advantage vs MLE could be an
  artifact of MLE's high sentinel rate — 16.4% of cache entries are
  `(stat=0, p=1)`, concentrated on high-d datasets where EM struggles. A
  model might learn to use "got a sentinel" as a dataset fingerprint. If
  `disable_littles` still wins when we restrict the comparison to pairs
  where Little's returned a real (non-sentinel) value, the advantage is
  about the feature's ML properties, not the sentinel rate. If it
  doesn't, the story is "MLE's sentinel-driven memorization is the
  problem" — a different finding entirely.
- **Hypothesis:** The `disable_littles` advantage survives on the
  real-value subset but is smaller (say, 3–4% instead of 6%).
- **Required harness change:** `training/report.py` currently summarizes
  metrics batch-wise; to do subset analysis we need per-sample rows
  tagged with their (dataset_name, generator_id). ~30 LOC change to
  `validate_detailed` to log this as a sidecar CSV alongside predictions.
- **Re-sweep specs:** `baseline`, `baseline_mom`, `disable_littles` at
  n=5 (or reuse the `n10_disable_littles` run if the harness change is
  in place by then).
- **Estimated runtime:** 5.5 h for the sweep + ~1 day for harness
  changes, tests, and Phase 2 subset-analysis code.
- **Decision rule:**
  - Effect survives on real-value subset → feature is genuinely bad for
    ML, not a sentinel artefact. Strong case for removal.
  - Effect evaporates on real-value subset → sentinel contamination is
    the root cause. Fix = reduce sentinel rate (e.g., fall back to MoM
    when MLE sentinels; or keep MoM as default since its sentinel rate
    is 0.9%).

---

## 2. `full-6spec-canonical` — The definitive dissertation ablation

- **Status:** queued (run after either of the above, pending direction)
- **Motivation:** After the outcome of the first two experiments, we'll
  know whether the Little's-slot stays or goes. Either way, the
  committee will want ONE canonical ablation table that measures every
  surviving feature group's contribution at adequate statistical power.
  This is it.
- **Hypothesis (n/a — descriptive experiment).**
- **Specs:** `baseline`, `baseline_mom`, `disable_missing_rate`,
  `disable_cross_column`, `disable_littles`, `all_disabled` — the full
  `DEFAULT_SPECS` list.
- **Seeds:** 10.
- **Runs:** 6 × 10 = 60.
- **Estimated runtime:** ~22 h — overnight plus morning.
- **Command:**
  ```bash
  python -u scripts/run_ablation.py \
      --config configs/training/ablation.yaml \
      --seeds 1 2 3 4 5 6 7 8 9 10 \
      --littles-cache /mnt/artifacts/project_lacuna/cache/littles_mcar_v2.json \
      --csv /mnt/artifacts/project_lacuna/ablation/canonical_n10.csv \
      2>&1 | tee /mnt/artifacts/project_lacuna/ablation/canonical_n10.log
  ```
  (If `disable_littles` wins the `n10-followup`, we may want to drop the
  `baseline_mom` spec from this too and save 10 runs. Decide at the
  time.)
- **Decision rule:** whatever the CIs say becomes the ablation table in
  the dissertation methods section. No branching actions expected — this
  is the "lock in the story" experiment.

---

## 3. `mcar-alternatives-bakeoff` — Test multiple Little's-alternatives at once

- **Status:** queued (run after `n10-followup` and `sentinel-subset-analysis`
  if those confirm the Little's feature is harmful; becomes moot if either
  exonerates the feature)
- **Motivation:** The current MCAR-slot features (Little's MLE, MoM) both
  assume multivariate normality (MLE explicitly; MoM via pairwise
  Gaussian moments). Real tabular data in `lacuna_tabular_110` is
  heavy-tailed, skewed, and integer-encoded-categorical-laden — MVN is
  badly violated. The 2026-04-19 sweep found that any cached Little's
  feature is worse than having none. This experiment tests *multiple*
  alternative MCAR statistics in one sweep, to separate three possible
  failure modes:
  - **(H1)** MVN assumption is the root cause (distribution-free methods
    should succeed).
  - **(H2)** Bivariate / linear measures are insufficient (kernel /
    tree-based methods should succeed where rank-based methods don't).
  - **(H3)** The architectural slot itself is the problem (all
    alternatives fail equally).
- **Method landscape (from 2026-04-19 lit-scan):** we cover ~11 families
  of MCAR tests across statistics and ML. Top candidates for a
  cached-scalar slot on mixed-type tabular data, ranked by expected
  suitability:

  | Rank | Method | Why for Lacuna | Family |
  |---|---|---|---|
  | 1 | **RF / GBM propensity-AUC** | Native mixed-type; nonlinear; deterministic with seed; sklearn-native; cheap | ML (folklore) |
  | 2 | **Jamshidian-Jalal `MissMech`** | Statistical community's modern recommendation when MVN fails; R package exists | Permutation |
  | 3 | **HSIC(R, X)** | Kernel independence test; theoretically clean; Python `dcor`/`hyppo` | Kernel |
  | 4 | Kendall's τ | Pure ranks, zero assumptions, but *linear* rank measure (weak for nonlinear) | Nonparametric |
  | — | Muzellec OT Sinkhorn divergence | Deferred (below); heavier engineering | ML |

  Rationale for picking {propensity-AUC, MissMech, HSIC} over {Kendall
  alone}: propensity-AUC directly addresses mixed types and
  nonlinearity, which a bivariate rank measure doesn't. MissMech is the
  statistical community's consensus alternative to Little's, so its
  performance under Lacuna is a dissertation-relevant comparison in its
  own right. HSIC fills the "kernel-based" quadrant. Kendall becomes
  redundant with propensity-AUC for our use case.

- **Hypothesis:** At least one of {propensity-AUC, MissMech, HSIC} at the
  Little's-slot position matches or exceeds the +6% contribution that
  the original heuristic provided in the 2026-04-17 sweep. If so,
  **H1** (MVN was the problem) is confirmed and we have a working
  replacement. If none do, **H3** (the slot itself is architecturally
  wrong) is the more plausible explanation and we delete the slot.

- **Implementation (prerequisite, ~2 weeks):**
  - Add three test functions to `pystatistics.mvnmle` (or a new
    `pystatistics.nonparametric_mcar` submodule):
    - `propensity_mcar_test(X, R, *, model='rf'|'gbm', cv=5, seed)` →
      returns mean and max per-column OOF AUC + permutation-null p.
    - `hsic_mcar_test(X, R, *, kernel='gaussian', bandwidth='median')`
      → HSIC statistic and gamma-approximation p-value.
    - `missmech_mcar_test(X, R, *, n_boot=999, seed)` — either wrap
      the R package via `rpy2` or reimplement the algorithm (Hawkins
      test on covariance homogeneity across patterns with k-NN
      imputation).
  - Extend `lacuna.data.littles_cache` to schema v3 with entries for
    all three new methods (stat + p_value for each). Cache build cost:
    propensity-AUC adds ~0.5s/pair (RF per column), HSIC adds ~0.1s,
    MissMech adds ~2-5s (bootstrap). Total ~3-6 hours for full
    lacuna_tabular_110 rebuild — acceptable one-time cost.
  - Extend `VALID_METHODS` in `littles_cache.py` and the
    `SemiSyntheticDataLoader` selector to include `propensity`, `hsic`,
    `missmech`.
  - Add `baseline_propensity`, `baseline_hsic`, `baseline_missmech` to
    `DEFAULT_SPECS`.

- **Specs:** `baseline` (MLE, for reference), `baseline_propensity`,
  `baseline_hsic`, `baseline_missmech`, `disable_littles` (control),
  5 seeds each.
- **Runs:** 5 × 5 = 25.
- **Estimated runtime:** ~9 h for the sweep + ~2 weeks for
  implementation (three new pystatistics functions + schema bump +
  Lacuna plumbing + tests + regenerated cache).

- **Decision rule (primary):**
  - Any of {propensity, HSIC, missmech} beats `disable_littles` at
    95% CI → **H1 confirmed**. Promote the winner to default (new
    ADR supersedes ADR 0002/0003).
  - None beat `disable_littles` → **H3 confirmed**. Delete the
    Little's-slot entirely. The finding itself becomes a dissertation
    methods contribution: "hand-rolled MCAR test statistics, from
    parametric to nonparametric to kernel to tree-based, do not
    provide useful features for neural mechanism classification in
    this architecture."
- **Decision rule (secondary, method-vs-method):** paired comparisons
  between the three alternatives inform which is best *if one wins*.
  This also answers the H1-vs-H2 question: if propensity and HSIC beat
  Kendall-style methods by a margin, nonlinearity matters beyond MVN.

- **Escalation path if all fail:** defer to a `neural-mcar-detector`
  experiment (Muzellec OT-based, below). But frankly, if RF / HSIC /
  MissMech all fail, it's very unlikely that a much more complex neural
  method succeeds — so the practical decision in that branch is just to
  delete the slot.

---

## Deferred / exploratory (not queued)

- **`neural-mcar-detector`** — adapt Muzellec et al. (2020)'s Sinkhorn
  divergence between `X | R=j=0` and `X | R=j=1` distributions as an
  MCAR feature. Only pursue if `mcar-alternatives-bakeoff` rules out
  every simpler explanation. Estimated engineering: 1–2 weeks to wrap
  the published code (`BorisMuzellec/MissingDataOT`) into a
  deterministic cached feature.
- **`cross-registry-generalization`** — train on `lacuna_tabular_110`
  minus one generator family, eval on the held-out family. Tests whether
  the model learned generator fingerprints vs. the mechanism class.
  Interesting for a methods paper but not load-bearing for the dissertation
  ablation story. Revisit once the core ablation is locked.
- **`cpu-vs-gpu-cache-build`** — characterise the cache-build speedup
  from pystatistics 2.1.0's batched EM + GPU path vs. pre-2.1.0
  serialised CPU. Nice anecdote for the engineering section of the
  dissertation; not a research question per se.
