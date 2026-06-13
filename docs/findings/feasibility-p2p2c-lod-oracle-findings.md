# P2.2c — LOD / Top-Coding Oracle Gate — Findings

*Branch `p2/delta-prior-rearchitecture`. Spec: `PROPOSAL-P2.2c-LOD-detectability-spectrum-audit.md`.
Oracle pre-check only — **no model trained.** Stop-for-review per the approved plan (oracle result
before any model training).*

## Result: gate PASSED — LOD/top-coding is strongly detectable under the fitted-Gaussian X-model

Bayes-optimal n-sample error (n=1024) of LOD(δ) vs a matched-rate MAR null — **point** and the
**β₁′-profiled** (strongest re-fitting MAR) null — at matched rate 0.30, τ at the 0.70 quantile.

| X-model (target, predictor) | δ=0 | δ=1 | δ=2 | δ=3 (profiled) |
|---|---|---|---|---|
| synthetic ρ=0.0 | 0.49 | 0.27 | 0.005 | **0.000** |
| synthetic ρ=0.3 | 0.48 | 0.02 | 0.000 | **0.000** |
| synthetic ρ=0.6 | 0.49 | 0.009 | 0.000 | **0.000** |
| cps1988 (wage, education) | 0.50 | 0.019 | 0.000 | **0.000** |
| psid7682 (wage, year) | 0.50 | 0.018 | 0.000 | **0.000** |
| computers (price, ram) | 0.51 | 0.009 | 0.000 | **0.000** |
| hmda (lvrat, mhist) | 0.49 | 0.025 | 0.000 | **0.000** |
| cps1985 (wage, education) | 0.49 | 0.015 | 0.000 | **0.000** |
| workinghours (hours, child5) | 0.50 | 0.030 | 0.000 | **0.000** |

(profiled ≈ point throughout — MAR cannot mimic the target-value truncation, as expected.)

- **δ=0 → Bayes error ≈ 0.5** everywhere (chance — correct; H0=H1).
- **δ≥1 → Bayes error collapses** on **all 6/6 real-fitted X-models**; δ≥2 is perfectly separable.
- This is the **detectability spectrum at the oracle level**: same matched-rate protocol, same
  datasets — own-value smooth self-censoring was hard/flat for the model; LOD's hard truncation edge
  is blatantly separable in principle. NORTH-STAR §2's prediction holds at the oracle.

**Gate decision (per spec §3):** signal exists ⇒ **proceed to the model ladder** (next cycle). A
model floor there would now be *attributable* — to representation/transfer, not absence of signal.

## The honest caveat (do not over-read the pass)

**The oracle is under the fitted-Gaussian X-model — and a Gaussian-oracle pass is necessary, not
sufficient.** The decisive reason for caution is the own-value precedent:

- Own-value self-censoring *also* had in-principle detectability at moderate/strong δ (rung 1: the
  model learned it on synthetic Gaussian X, adj-acc 0.90). An analogous Gaussian oracle on real
  columns would very likely have **also passed** for own-value.
- Yet the own-value **model floored on real (non-Gaussian) X** (rung 3/3b, proxy, cardinality,
  strong-δ — all flat).

So **this Gaussian-oracle pass does not, by itself, distinguish LOD from own-value's situation.** It
confirms the mechanism carries information *under the idealized model* and **clears LOD to the model
ladder** (and pre-attributes a future floor to representation/real-geometry rather than "no signal")
— but it does **not** predict that the encoder will recover LOD on real X. The decisive test remains
the model on real (non-Gaussian) X with semi-synthetic LOD holes, held out.

### Why there is nonetheless a *structural* reason for optimism (a hypothesis, not a result)

LOD's footprint differs from own-value's in a way that should transfer better, independent of the
Gaussian oracle: it is a **hard truncation edge in the OBSERVED target marginal** — directly visible
in the raw observed values (top-coding a real wage column leaves a visible upper-truncation),
**robust to non-Gaussianity**. Own-value smooth self-censoring leaves a subtle conditional-shift that
is confoundable with the unknown baseline. Whether this sharpness actually transfers to the encoder
on real X is exactly what the model ladder must test — it cannot be claimed from the oracle.

## What this licenses (and what it does not)

- **Licenses:** running the LOD model ladder (semi-synthetic LOD on **held-out** real survey X,
  target-conditioned, same δ-bins / RPS / calibration / leakage gate / manifest), with the oracle
  having established that a floor there is a *transfer/representation* result, not a missing signal.
- **Does NOT license:** any claim that LOD "works" on real survey X, or that the footprint channel is
  alive, until the **model** (not the oracle) beats uniform out-of-family on held-out real survey X.

## Methodological note (the P2.2b lesson applied)

Running the oracle FIRST is the correction from P2.2b. It (a) could have killed LOD cheaply (flat
oracle ⇒ stop); it did not, and (b) it pre-attributes the upcoming model result. Had we run this
oracle for own-value at the start, we would have known its model floor was a *transfer* failure, not
an absence of signal — which is precisely the distinction the whole P2.2b ladder had to recover the
hard way.

## Status & recommended next

Gate PASSED (signal exists under the fitted-Gaussian model on 6/6 real datasets). **Stop for review
before any model training** (per plan). Recommended next cycle: the **LOD model ladder** — train a
target-conditioned δ-prior on semi-synthetic LOD holes on real survey X, validate on **held-out**
real survey datasets, and run the **direct A/B vs own-value self-censoring on the same datasets**.
That model A/B — not the oracle — is the real test of whether the detectability spectrum transfers to
the learned channel. Same RPS / δ-bins / calibration / leakage gate / manifest; no architecture
change. Determinism, fail-loud, manifest-validated throughout. Full suite green (1314 passed, 1
skipped); `runs/p2p2c-lod-oracle.json` saved.
