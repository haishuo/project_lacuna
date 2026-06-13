# D2 — Footprint-Geometry / Regime Map & Transfer Postdiction (findings)

*Analysis only — **no model, no training, no acquisition** (PI 2026-06-07 chose D2). Deterministic
descriptive statistics on the EXISTING corpus. Implements D2 of
`CONSOLIDATION-MEMO-SCF-wealth-conditional-scaling.md` (§10). Code:
`lacuna/survey/regime_descriptors.py` (+8 tests), `scripts/build_regime_map.py`; artifact
`runs/regime_map.json`. 246 survey tests green.*

---

## 0. What D2 was for

The SCF result split the held-out domains into "diversity helped" (psychology +0.064, health +0.037) and
"diversity did not" (NHANES +0.045 noisy, wealth −0.053 flat). The revised hypothesis (H1) is that
diversity helps **only when the added domains teach the relevant footprint geometry** for the held-out
domain. D2 asks: **can a measurable regime/footprint-geometry metric, computed on data we already have,
*postdict* that split?** If yes, we have a tool to target acquisition by regime (not domain count) and a
mechanism for the dissertation reframe.

## 1. The metric (scale-invariant, because φ standardizes)

The φ-spine **standardizes each column within-observed** (z-score) before pooling order statistics, so
raw dollar scale / dynamic range is largely normalized away. What *survives* standardization — and so
defines the footprint geometry the model actually sees — is the **scale-invariant shape**. Each
non-constant column is placed at a 4-D coordinate:

- **L-skewness τ₃** and **L-kurtosis τ₄** (robust L-moments; bounded in/near [−1,1] even under heavy
  tails — unlike ordinary kurtosis, which a single wealth outlier sends to ~10²–10³),
- **log₁₀(cardinality)** (the continuous ↔ ordinal/Likert axis),
- **has_negative** (support sign).

Coordinates are z-standardized across **all** corpus columns; **coverage_distance(D | pool)** = mean over
D's columns of the nearest pool-column distance (how well a training pool covers D's geometry);
**coverage_gain(D)** = coverage_distance(narrow) − coverage_distance(diverse); **isolation(D)** =
coverage_distance to the rest of the corpus. Validated against closed-form L-moments (uniform τ₄=0,
exponential τ₃=1/3, normal τ₄≈0.123).

## 2. The domain × regime map

| domain | n_col | L-skew | L-kurt | log₁₀card | med card | moment-skew | excess-kurt |
|---|---|---|---|---|---|---|---|
| labor | 26 | 0.109 | 0.110 | 1.55 | 36 | 0.30 | 0.7 |
| psychology (bfi) | 28 | −0.068 | 0.070 | 0.78 | 6 | −0.27 | −0.2 |
| health (yrbss) | 5 | 0.028 | 0.019 | 0.90 | 8 | 0.14 | −1.0 |
| nhanes (wt+pov) | 13 | 0.047 | 0.057 | 1.65 | 45 | 0.19 | −0.5 |
| **wealth (SCF)** | 6 | **0.898** | **0.789** | **3.49** | **3112** | **10.88** | **143.5** |

**Wealth is a dramatic geometric outlier on every axis** — an order of magnitude more right-skewed and
heavy-tailed than anything else, and ~70× the cardinality. Labor/psychology/health/NHANES form one tight
cluster (near-symmetric, modest L-kurtosis, low–moderate cardinality). The L-kurtosis vs moment-kurtosis
gap for wealth (0.79 vs 143.5) is exactly why the metric uses **L-moments** — the moment version is
dominated by a handful of high-net-worth households; the L-version stays interpretable.

## 3. Postdiction — regime coverage tracks the observed transfer split

| held-out D | obs Δ | sig | cov_narrow | cov_diverse | **cov_GAIN** | **isolation** |
|---|---|---|---|---|---|---|
| psychology | +0.064 | HELPS | 0.535 | 0.334 | 0.201 | 0.334 |
| health | +0.037 | HELPS | 0.371 | 0.199 | 0.172 | 0.199 |
| nhanes | +0.045 | noisy | 0.558 | 0.338 | 0.220 | 0.334 |
| **wealth** | **−0.053** | **flat** | **3.405** | **3.376** | **0.028** | **3.376** |

- **rank corr(coverage_gain, Δ) = +0.80** (n=4, indicative).
- **rank corr(isolation, Δ) = −0.20** (weak; the expected sign — more isolated ⇒ less transfer — but
  only wealth is meaningfully isolated, so the rank statistic is dominated by ties among the other three).

**The robust, large signal is the wealth separation.** Wealth has, by an order of magnitude:
- the **lowest coverage_gain** (0.028 vs 0.17–0.22) — adding the diverse pool brings training *almost no
  closer* to wealth's geometry, because **nothing in the corpus is near wealth's heavy-tailed regime**;
- the **highest isolation** (3.376 vs 0.199–0.334) — wealth's columns are ~10× farther from any other
  domain's columns than the others are from each other.

And wealth is the **only** domain where diversity failed. So the SCF flatness is **postdicted** by the
metric: the φ-spine cannot transfer into a footprint geometry the training pool does not contain, and the
pool contains nothing geometrically like wealth.

## 4. Honest reading & limits

- **This is postdiction, not prediction.** The metric was computed knowing the Δs. Its value is forward:
  it gives a **mechanism** (transfer needs footprint-geometry *coverage*) and a **measurable proxy**
  (coverage_gain / isolation) that can be computed for a *candidate* domain **before** acquiring it.
- **The strong claim is one-sided:** regime **isolation predicts transfer FAILURE** (wealth, clean and
  large). The metric is **much weaker at the fine ordering** among the three "near" domains — NHANES has
  the highest coverage_gain (0.220) yet only middling, noisy Δ, and psychology has the highest Δ yet
  middling gain. So it does **not** yet predict *which* near domain benefits most; that ordering is within
  the Δ noise anyway (NHANES is "noisy, not sig").
- **n = 4 held-out domains.** A rank corr of 0.80 here is suggestive, not significant. The result is best
  stated as: *the one domain the pool fails to cover geometrically (wealth) is the one diversity fails to
  help* — a single, decisive, mechanistically-coherent case, not a fitted law.
- **Does not separate the two §6 sub-readings cleanly:** wealth is simultaneously the most heavy-tailed
  *and* the highest-cardinality *and* the only un-covered domain, so "continuous-vs-ordinal" and
  "regime/scale-mismatch" both point the same way here. Disentangling them needs a **regime-matched
  continuous** held-out domain (a future diagnostic, D3), not another descriptor.

## 5. What D2 delivers (and what it doesn't)

**Delivers:**
- A concrete, tested, scale-invariant **footprint-geometry metric** and a **domain × regime map** of the
  current corpus.
- A **coverage / isolation** proxy that **postdicts** the SCF transfer failure mechanistically — turning
  "wealth didn't transfer" from an anomaly into "wealth is geometrically uncovered."
- An operational **acquisition screen**: for any candidate survey, compute its target columns' regime
  vectors and its coverage_gain against the current pool **before** downloading. Low gain / high isolation
  ⇒ acquiring it alone will likely *not* improve transfer (you would also need its geometric neighbours).

**Does not deliver:** a validated *predictive* law (n=4, postdiction), nor a separation of family-vs-regime
(needs D3), nor any statement that the wealth flatness is *not* partly the optimization-instability seen in
the diverse seed collapses (still unresolved; D3).

## 6. Implications for the next decisions (no experiments proposed)

- **Acquisition (D5) screen, now computable:** rank candidate continuous surveys by **coverage_gain to the
  current pool**. A lone heavy-tailed income/wealth domain scores like SCF (low gain) ⇒ acquiring *one* is
  predicted *not* to firm transfer; you need a **cluster** of regime-similar continuous domains so the pool
  can interpolate the geometry. This is the precise form of "target by family/regime, not count."
- **D3 (first experiment when runs resume) is now sharper:** hold out a continuous domain that **is**
  covered (regime-matched) and test whether continuous→continuous transfer appears — directly separating
  "regime mismatch" (fixable by acquisition) from "instability / representational limit" (not). The regime
  map tells us which existing splits are regime-matched vs mismatched to design that test.
- **Dissertation reframe (D4):** the metric supplies the mechanism the §0 reframe needed — *"transfer is
  structured by target-family/regime; diversity helps only when the pool covers the held-out footprint
  geometry,"* with **coverage/isolation** as the operational variable and **SCF wealth as the clean
  geometrically-uncovered counterexample.**

---

*No experiments, runs (model/training), downloads, or architecture changes are performed here. D2 produces
a regime map + a coverage/isolation proxy that postdicts the SCF transfer split and becomes a forward
acquisition screen. Next decisions (D3/D4/D5) remain PI calls; no new experiment is proposed.*
