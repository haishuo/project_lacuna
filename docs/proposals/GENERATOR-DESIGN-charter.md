# Generator Design Charter — Lacuna (Recoverability Reformulation)

*The generators are the load-bearing assumption of the reformulated design: the entire deployable
claim ("this per-column functional bound holds at 95% coverage") transports to real data only if
real missingness lives inside the generator family. This charter governs all generator work.
Grounded in `docs/LITERATURE-REVIEW.md`. Build spec + advisor artifact. No code yet — principles
and validation protocol first.*

---

## 1. Why generators became load-bearing (the role change)

In v1.0 the generators were a *training aid* for a classifier — a slightly-off generator gave a
slightly-off decision boundary (graceful degradation). In the reformulation the generators **are
P_prior**, and certification on plasmode transports to real data only if the plasmode→real
distribution shift is small. Conformal coverage breaks exactly under that shift (Gibbs 2023; see
review §3). So generator realism is not a quality-of-life nicety — it is the binding validity
condition of the coverage guarantee. A wrong spine in v1.0 cost accuracy; a wrong generator family
here costs **soundness**, silently, exactly where the guarantee is claimed.

This is empirically confirmed: imputation benchmarks (review §5) show the *missingness scenario
determines the conclusion* (MCAR compresses method differences; structured gaps reveal them). A
certification built on the wrong scenario is silently wrong.

## 2. The harder bar (why our generators ≠ the field's)

The field builds generators (`ampute`, `mdatagen`, even Radosavljević's realistic one) to **rank
imputation methods** — a weak realism bar, robust to mis-specification because only *relative*
orderings must survive. Lacuna needs generators realistic enough to **certify a transportable
quantitative bound** — a strictly stronger bar, because we make a deployable numerical claim, not a
comparison. No prior work certifies coverage of a deployable bound. This is the gap and the
contribution.

## 3. The four-layer realism decomposition (what "good" means, and what is checkable)

Generator realism is not one uncheckable thing. It factors, and the genuinely uncheckable part is
narrow:

| layer | what it is | checkable against real data? | how |
|---|---|---|---|
| **L1 marginal footprint** | rates, block structure, co-missingness | **YES** | footprint-statistic match + discriminator-near-chance vs ESS/NHANES/GSS masks |
| **L2 conditional/proxy structure** | dependence of missingness on *other observed* columns (the MAR component) | **YES** | we measured it at 0.90 (Stage-2); match the conditional refusal model |
| **L3 coverage/span** | does the family contain the real mechanism at all | **YES** | support/density check: does real missingness fall inside the generated manifold |
| **L4 residual δ** | dependence on the *missing value itself, after conditioning on all observed* | **NO — by theorem** | not claimed; swept explicitly; anchored by the nonresponse literature (§6) |

**The encouraging consequence:** most of what determines a recoverability bound (rate × proxy-
explained spread) lives in L1–L3 and is empirically validatable against the real corpora we now
hold. Only the δ-multiplier (L4) is irreducibly assumption — and it is the one factor we *sweep*
rather than fit. This is why the bound factors as rate × residual-spread × δ-range.

## 4. The design: adopt the floor, extend the structure, randomize the unknowable

- **Adopt as the floor.** Make `pyampute`'s weighted-sum-score mechanisms a *subsumed baseline* in
  the vocabulary, so we are provably ≥ the field standard and reviewers see continuity. Adopt the
  "plasmode" framing (Weberpals) in all documentation.
- **Extend L1–L3 from real data (Radosavljević method, our acceptance test).** Learn co-missingness
  blocks, inter-variable correlation, and informative-missingness structure from ESS/NHANES/GSS.
  Unlike Radosavljević (who validated by imputation-ranking), hold it to the **realism gate as an
  acceptance test**: a discriminator must be near chance distinguishing our masks from held-out
  real masks, AND footprint statistics must match held-out real distributions within tolerance.
- **Randomize L4 (the contribution, no prior art).** Realism-anchored *center* (δ informed by the
  bracket/linkage/nonresponse literature, e.g. Jabkowski for ESS income), domain-randomized
  *spread* (broad enough that the real δ is inside the support — the sim-to-real coverage posture),
  swept explicitly so it appears once, in the bound, never hidden in the imputer (cf. reformulation
  §7.1 ignorable-only training split).

**Philosophy call (the one genuinely open decision):** realism (tighter bounds, must defend
fidelity) vs domain randomization (wider, conservative, defensible bounds, only must argue "real is
inside the support"). **Decision of record: hybrid — realism-anchored center, domain-randomized
spread.** For a dissertation whose product is *honest uncertainty*, never having to claim a
generator is "real" (only that it brackets reality) is the stronger epistemic position; the realism
gate keeps the center from being gratuitously loose so bounds stay useful.

## 5. The circularity guard

The danger mirrors the cat critique: hand-design generators from our assumptions, certify against
them, and we have certified against ourselves. The break: **L1–L3 are validated against real
refusal data we did not author** (ESS/NHANES/GSS) via the realism gate — a falsifiable empirical
test, not an appeal to intuition. L4 cannot be broken (theorem) — so it is never *claimed*, only
swept and labeled. The asymmetry (validate what you can, randomize what you can't, claim only the
former) is the whole honesty argument and must be stated explicitly wherever generators are used.

## 6. Validation protocol (generator quality is MEASURED, a first-class deliverable)

A generator family is admissible for certification only if it passes, per layer:

1. **L1 footprint gate = a Classifier Two-Sample Test (C2ST).** Not a bespoke check: the
   "discriminator near chance" gate IS the C2ST (Lopez-Paz & Oquab 2016) — train a classifier to
   distinguish generated vs held-out-real masks; near-chance accuracy ⇒ distributions match, and
   the classifier *localizes which features are unrealistic* (actionable for fixing generators).
   Make it finite-sample rigorous via the **conformal C2ST** (Hu & Lei; Bansal et al. 2025), which
   converts any classifier's scores — even a weak one — into exact finite-sample p-values with
   Type-I control (bonus: ties the realism gate to the SAME conformal machinery as the coverage
   layer §3). Alongside C2ST, a footprint-statistic panel mapped to the recognized fidelity
   taxonomy — attribute / bivariate / population / application fidelity (Dankar 2022) — plus an
   **authenticity** check (Alaa et al. 2021) guarding against the generator merely memorizing real
   masks. See `docs/LITERATURE-REVIEW.md` §4 for citations.
2. **L2 conditional gate.** The conditional missingness model (missingness | other observed cols)
   matches the real conditional refusal model (the 0.90-predictable structure) on held-out
   instruments.
3. **L3 span gate.** Real missingness patterns fall inside the support/typical-set of the generated
   manifold (out-of-support real patterns are reported, never silently extrapolated over).
4. **L4 honesty check.** The δ-grid brackets the literature-anchored range for each reference class;
   the span is reported, and any real column whose plausible δ exceeds the grid is flagged R3
   (unanchored), not silently bounded.
5. **End-to-end coverage check.** On held-out plasmode spanning {datasets × mechanism vocab × δ-grid
   × rates}, the assembled bound brackets the realized functional gap at the stated level — the
   conformal-under-shift certification (review §3).

All gates run against held-out real corpora / held-out plasmode (block-aware, leave-instrument-out
where applicable). Failures are reported, never tuned away.

## 7. Re-objectiving the existing generators (the work)

The current `lacuna/generators/` (29 modules, ~7,800 LOC) was built for *diversity* (cover
mechanism types as classifier training signal), not *realism* or *coverage*. The work is therefore
re-objectiving, not greenfield:

1. **Audit + map** the existing MCAR/MAR/MNAR families against `pyampute`'s vocabulary and the L1–L4
   decomposition (what do we already produce; what's the realism status; what's missing).
2. **Wire the realism gates** (§6) as a test suite against the corpora — this is the new acceptance
   harness and the first concrete build.
3. **Fit L1–L3 from real data** where current generators are unrealistic (Radosavljević-style).
4. **Make δ an explicit swept axis** across all MNAR families; remove any implicit/fixed δ.
5. **Subsume pyampute** as a registered baseline family for continuity + benchmarking.

## 8. What this buys, and the honest limit

Buys: a generator family whose realism is *measured* on three axes against real data, with the
uncheckable axis swept and anchored — making the downstream coverage guarantee transportable on
stated, falsifiable conditions. Limit (stated up front): L4 is unverifiable by theorem; we never
claim the generated δ-dependence is real, only that the swept range brackets the
literature-anchored plausible range. Where it doesn't, we abstain (R3). The coverage guarantee is
*conditional on L1–L3 realism + L4 bracketing*, and both conditions are reported, not assumed.

## 9. Immediate next steps (build order)

1. **Generator audit doc** — map existing 29 modules to pyampute vocab + L1–L4 status. (analysis)
2. **Realism-gate harness** — `generator_validation.py` extended into the §6 acceptance suite
   against ESS/NHANES/GSS. (first code; reuses existing `analysis/generator_validation.py`)
3. **L1 footprint fitting** from real corpora for the families that fail the gate.
4. Then: the functional-gap experiment can run on certified generators (the demo), and the δ-anchor
   ledger v1 (Jabkowski ESS income + NHANES self-report-vs-measured-weight) feeds R2.

*The generators are where the reformulation's soundness lives. We spend disproportionate time here
on purpose — and we make their quality a measured, reported quantity rather than an assumption,
because the entire coverage claim rests on it.*
