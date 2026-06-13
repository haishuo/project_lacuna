# Consolidation Memo — SCF Wealth & the Conditional Scaling Hypothesis

*Consolidation document — **analysis only. No experiments, no runs, no acquisition, no model/architecture
work** (binding; PI 2026-06-07). Records the SCF wealth result as a *scientific finding* and revises the
scaling hypothesis. Governed by `NORTH-STAR.md` and `MASTER-lacuna-survey-architecture.md`; grounded in
`feasibility-stage1-crossdomain-findings.md` (§7 tightened evidence, §8 SCF) and
`PROPOSAL-SCF-wealth-acquisition-plan.md`. Recommended to be folded into the MASTER doc on PI sign-off.*

---

## 0. Headline (PI interpretation, accepted)

SCF does **not** mean "acquisition failed." It **falsifies the overly broad scaling hypothesis** and
replaces it with a conditional one.

- **Old (falsified, broad form):** *more domain diversity improves transfer.*
- **Revised (supported):** *domain diversity improves transfer **conditionally** — depending on the
  held-out domain's **target family / distribution regime / idiom**. Diversity helps only when the added
  domains teach the **relevant footprint geometry**.*

Standing facts after SCF: **(1)** the φ-spine works; **(2)** SCF wealth is **detectable**; **(3)** more
data is **not** automatically useful unless it is the **right** data; **(4)** corpus expansion should now
be **targeted by target family**, not generic domain count; **(5)** the dissertation claim should shift
from "more domains improve transfer" to "**transfer is structured by target family and idiom; domain
diversity helps only when the new domains teach the relevant footprint geometry.**"

---

## 1. The SCF ingestion result

Acquired the **Survey of Consumer Finances 2022 Summary Extract** (`rscfp2022.dta`; U.S. Federal Reserve;
public domain; sha256 `3bb4d890…`) as a new continuous-target **`wealth`** domain — the acquisition
rubric's top-of-both-axes pick (`PROPOSAL-SCF-wealth-acquisition-plan.md`). Ingested **strictly per the
acquisition framework** (`scripts/build_scf_role_b.py`):

- **Implicate 1 only** (`y1 % 10 == 1`): the file's 22,975 rows are 4,595 families × 5 imputation
  implicates. The 5 implicates are imputed copies of the **same families** — *not* 5× data (the
  slice-inflation trap, framework §1). One row per family.
- **Role-B only** (binding exception): the public file is **multiply imputed**, so item-nonresponse is
  already filled — there is **no natural-missingness role-A archive** to preserve. Recorded
  `role_a_archive = none`. We never train on natural missingness; here there is none.
- **Targets** (net worth + income + major asset aggregates): `networth`, `income`, `asset`, `fin`,
  `nfin`. No sentinel recode (imputed dollars carry no refuse/DK codes); plausible range **admits
  negative net worth** (6.9% negative). **All 5/5 targets pass** the §3 gates (spike 0.000, range-ok,
  cardinality 1,159–4,250). **4,595 rows, `preferred` tier** — ~8× any prior role-B base.
- Design/weight/id columns (`wgt`, `y1`, `yy1`) excluded. Registered in the new tested
  `survey_catalog.ROLE_B_BASES` ledger (`wealth` domain, block `scf`). 238 survey tests green.

**Read:** ingestion is clean and large. The base is the *biggest, cleanest continuous-target role-B base
in the corpus* — exactly the "decisive, low-noise held-out domain" the plan promised.

## 2. The categorical-dilution confound — why the continuous-only base is the valid wealth test

A first pass on the **as-built** base (5 continuous targets **+ 8 categorical/ordinal predictors**) gave a
held-out wealth AUC of **0.441 ± 0.005 — below chance.** This is **not** anti-transfer; it is a
**measurement artifact** of the frozen pipeline:

- `select_target_predictor` samples the censored target **uniformly over all non-constant columns** —
  there is **no cardinality filter**. So a categorical predictor (`married`, `lf`, `edcl`, …) can be drawn
  as a "top-coding target," which is **degenerate** for a continuous idiom (top-coding a binary at the
  0.70 quantile is meaningless).
- With **8/13 categorical columns** (`continuous_col_fraction = 0.46`), ~54% of held-out test draws are
  degenerate, dragging the pooled AUC below 0.5.

**Fix:** restrict the base to **continuous columns only** — the 5 wealth aggregates + continuous `age`
(`cont_frac = 1.0`). Then **every** uniformly-sampled target is a valid continuous idiom target. The
framework's continuous-target requirement *binds at target selection*; the continuous-only base is the one
that honors it. We emit both bases — `rb_scf2022_wealth_cont` (**canonical**, used for the test) and
`rb_scf2022_wealth` (retained to *document* the confound).

**Validity statement:** the **continuous-only** base is the valid wealth-domain test. The below-chance
as-built number is a confound, recorded, not a result. *(NHANES bases carry ~22% categorical columns — a
milder version of the same effect — left unchanged for reproducibility; SCF's 62% made it acute. This is a
known, bounded caveat on all role-B held-out numbers, largest for SCF-as-built, removed in SCF-cont.)*

## 3. The final SCF leave-one-domain-out result

Block-aware leave-one-domain-out, continuous-only base, NARROW (labor) vs DIVERSE (+psych+health+NHANES),
test on held-out wealth (`scripts/run_curve_tightened.py`):

| seeds | narrow (labor) | diverse (+others) | Δ | verdict |
|---|---|---|---|---|
| 5 (canonical) | 0.677 ± 0.012 | 0.571 ± 0.156 | −0.107 | ~flat (huge variance) |
| 10 (tightened) | 0.655 (SE 0.026) | 0.602 (SE 0.041) | **−0.053** (2·SE 0.098) | **~flat / inconclusive** |

Per-seed (10): narrow `[.684 .675 .656 .690 .681 .679 .426 .668 .728 .659]` (1 collapse);
diverse `[.341 .680 .425 .718 .689 .682 .504 .641 .693 .647]` (3 collapses).

The other three LOO rows and the cumulative curve **reproduce §7 exactly** (determinism intact):
bfi +0.064 (sig), yrbss +0.037 (sig), NHANES +0.045 (noisy), cumulative slope +0.0068 (n.s.).

**Two facts:** (a) **narrow ≈ 0.66** ⇒ wealth top-coding is **detectable**, on par with the bfi/yrbss
narrow baselines (0.665 / 0.642); (b) **Δ = −0.05 to −0.11, within 2·SE ⇒ diversity does not help**, and
the diverse pool is **less stable** (3 collapses vs 1). The sign is negative but the magnitude is
**inconclusive**; the honest claim is **flat, not "diversity hurts."**

## 4. Revised interpretation of the scaling hypothesis

| | claim | status after SCF |
|---|---|---|
| **H0 (broad)** | more domain diversity ⇒ better transfer (any held-out domain) | **falsified** — wealth is a clean counterexample |
| **H1 (conditional)** | diversity ⇒ better transfer **iff** the added domains teach the relevant footprint geometry for the held-out target family/regime | **supported** — bfi/yrbss (helped) vs NHANES/wealth (not) |

The cumulative in-family slope (labor test) was already weak (+0.0068, n.s.); the **leave-one-domain-out**
was the load-bearing evidence for H0, and it now **splits by target family** (§6). H1 is the minimal
revision consistent with all four held-out domains.

## 5. Three distinctions that must not be conflated

The SCF result is only coherent if these are kept separate:

1. **Target-domain detectability** — *can the idiom be detected in domain D's own targets, given a model
   trained appropriately?* Property of (D's targets, idiom, φ-spine). **Wealth: YES** (narrow 0.66). This
   is the comparison-class-relative detectability of §3 of the MASTER (vs profiled MAR-on-predictors).
2. **Transfer benefit from added domains** — *does adding OTHER domains to training improve performance on
   held-out D?* Property of (training pool, D). **Wealth: NO** (Δ flat). This is what the leave-one-domain-
   out measures.
3. **Domain-family mismatch** — *is D's target family / distribution regime far enough from the training
   domains that the diverse pool does not teach D's footprint geometry?* Property of (D, pool) **geometry**.
   **Wealth: plausibly YES** — the candidate explanation for why (1)=yes but (2)=no.

**The crux:** detectability (1) and transfer (2) are **independent axes**. A domain can be perfectly
detectable in-domain yet gain nothing from a diverse pool if (3) the pool is the wrong geometry. SCF is the
first clean demonstration of (1)=yes ∧ (2)=no, and (3) is the hypothesized bridge. Before SCF we implicitly
assumed (1) ⇒ (2); SCF severs that.

## 6. The emerging continuous-vs-ordinal asymmetry

Reading the four held-out domains by target family:

| held-out domain | target family / regime | diversity Δ | significance |
|---|---|---|---|
| psychology (bfi) | **ordinal / Likert** (bounded, low-card) | **+0.064** | sig (>2 SE) |
| health (yrbss) | **ordinal / binary** risk behaviors | **+0.037** | sig (>2 SE) |
| NHANES weight+poverty | **continuous**, moderate scale | +0.045 | not sig (noisy) |
| **wealth (SCF)** | **continuous, heavy-tailed**, huge dynamic range, negatives | **−0.053** | flat/inconclusive |

The two **significant** gains are **ordinal/Likert**; the two **continuous** held-out domains are
**flat/noisy**. This is the asymmetry the PI flagged. **Two readings, not yet disentangled:**

- **(a) Family asymmetry (ordinal vs continuous):** the diverse pool contains ordinal domains (bfi/yrbss)
  that teach ordinal footprint geometry; ordinal held-out domains benefit, continuous ones do not.
- **(b) Regime/scale mismatch within continuous:** the pool's continuous domains (labor wages ~$10⁴–10⁵;
  NHANES weight/poverty bounded) are *unlike* SCF wealth (dollars to $2.2×10⁹, heavy right tail, negatives).
  Even **continuous→continuous** transfer fails when the **regime** (tail index, dynamic range) differs.

**(b) is the sharper and more falsifiable hypothesis,** and is **partly evidenced already:** the pool
*does* contain continuous domains, yet wealth transfer is flat ⇒ generic "continuous" is not enough; the
**regime** must match. Distinguishing (a) from (b) requires a regime-matched continuous held-out domain
(§8). Either way the conclusion is the same at the level of H1: **transfer is target-family/regime-specific.**

**Honest caveats (attributable-not-confounded discipline):**
- The diverse-pool **instability** (3 seed collapses on SCF's heavy-tailed scale) may be an **optimization
  artifact** of a heterogeneous pool on extreme X, not a fundamental transfer fact. No tuning was done
  (frozen). The central tendency is clearly not a positive gain, but the *mechanism* (mismatch vs
  instability) is unresolved.
- **One** clean continuous held-out domain (SCF) + **one** noisy one (NHANES) is **suggestive, not
  decisive** for the asymmetry. n=2 on the continuous side.
- Detectability is **comparison-class-relative** (MASTER §3): "wealth detectable" = vs profiled
  MAR-on-predictors at matched rate. "Transfer flat" names its comparison: the **current** diverse pool.

## 7. What this means for the acquisition thesis

- The **generic** acquisition thesis ("acquire more domains → curve firms up") is **not supported** by the
  decisive test. The §7 "acquisition justified" conclusion is **narrowed**: it was driven by ordinal/Likert
  held-out gains (bfi/yrbss); it does **not** extend to the continuous, idiom-relevant domains we most care
  about (top-coding/self-censoring live on continuous targets).
- **Corpus expansion must be targeted by target family / regime, not domain count.** Adding another generic
  domain is **not** expected to improve transfer to a held-out continuous heavy-tailed domain unless it
  **teaches the relevant footprint geometry** (regime-matched continuous targets).
- This **reframes the grant ask**: not "fund a bigger corpus" but "fund the corpus that fills the **specific
  footprint-geometry gaps** for the target families the tool must serve." The acquisition rubric's two axes
  (idiom value vs diversity value) need a **third lens: regime/footprint-geometry match** to the held-out
  family.
- It does **not** retract the φ-spine validation or wealth detectability. The bottleneck is still data, but
  the **selection principle** for data has changed from *count* to *family/regime coverage*.

## 8. Evidence required before acquiring another continuous-domain survey

Acquire only against a **pre-registered, falsifiable transfer prediction**, not "+1 continuous domain":

1. **A regime-distance prediction.** Before acquiring D, quantify D's target **regime** (tail index /
   dynamic range / support, e.g. negatives) and show it is **close to** an existing or co-acquired training
   continuous domain. Pre-register: *"because D's regime matches {training domain}, diversity should improve
   held-out transfer to D by ≥ the bfi effect."* (This metric is an **analysis** on existing data — no
   acquisition needed to define it.)
2. **A continuous→continuous transfer existence proof.** Demonstrate (when runs resume, no acquisition) that
   **regime-matched** continuous domains transfer to a held-out continuous domain *at all* — i.e. hold out
   NHANES poverty while training includes labor+NHANES-weight (regime-similar continuous), and check Δ>0. If
   even regime-matched continuous→continuous transfer is flat, acquisition of continuous domains is **not**
   justified regardless of count, and the limit is representational/identifiability, not coverage.
3. **Resolve the instability confound.** Show the diverse-pool seed collapses are not the whole story (more
   seeds, or understanding the collapse mechanism) so a future Δ is trustworthy.
4. Only if 1–3 favor acquisition: target an **income/wealth/expenditure** survey whose **public top-coding**
   regime matches a training continuous domain, with the pre-registered prediction attached.

## 9. Evidence required before reframing the dissertation around domain-family-specific transfer

The reframe in §0 is promising but **not yet earned** as a thesis claim. Needed:

1. **Replication of the asymmetry with adequate seeds** — tighten the two noisy continuous results so the
   split is *"≥2 ordinal helped (sig) ∧ ≥2 continuous not-helped (sig-flat)"*, not "2 helped, 2 noisy."
2. **Disentangle (a) family vs (b) regime** (§6) with **one regime-matched continuous** held-out domain:
   if regime-matched continuous→continuous transfers but mismatched does not, the claim is **regime
   geometry**, the stronger and more defensible form.
3. **A mechanism + measurable proxy.** Articulate *why* (footprint geometry differs by family/regime) and
   define a **footprint-manifold-distance** proxy that **predicts** which held-out domains benefit — turning
   the asymmetry from an observation into a **predictive law** (M2 coverage made operational).
4. **Oracle-gating where possible.** Where the question is "can't transfer" vs "model won't transfer,"
   anchor with the profiled-MAR oracle (the comparison-class engine), per the project's "a 'no' is a result
   only if oracle-gated" discipline.
5. **Rule out the optimization-artifact reading** (instability) as the explanation for the continuous flatness.

Until 1–5, the dissertation should state the asymmetry as a **strong working hypothesis with one clean
counterexample (SCF)**, not a proven law.

## 10. Recommended next decision points (no experiments authorized yet)

Sequenced; each is a PI decision, not an auto-run:

1. **D1 — Fold this memo into the MASTER doc** (§4/§5 of MASTER) and mark H0→H1 as the current scaling
   position. *(Writing only.)*
2. **D2 — Define the footprint-geometry / regime-distance metric** as an analysis on the **existing** corpus
   (tail index, dynamic range, support, low-card fraction per domain/target). *No acquisition, no training.*
   Output: a domain×regime map that *predicts* transfer — the prerequisite for both §8.1 and §9.3.
3. **D3 — When runs resume, the first experiment is diagnostic, not acquisition:** the
   **continuous→continuous transfer existence proof** (§8.2) + **stability re-run** (§9.1/§9.5) on existing
   data. This resolves regime-mismatch vs optimization-instability and is the gate for everything downstream.
4. **D4 — Decide the dissertation framing** (keep "more domains" narrowed to ordinal, or commit to the
   family/regime-specific reframe) **after** D3, per §9.
5. **D5 — Acquisition is deferred** until D2+D3 produce a pre-registered, regime-matched transfer prediction
   (§8). No download is recommended now.

**Binding for now:** architecture frozen; no runs; no acquisition. The φ-spine and SCF detectability stand;
the open scientific question is **the structure of transfer (family/regime), not the architecture and not
the domain count.**

---

*No experiments, runs, downloads, or model/architecture changes are proposed by this document. It
consolidates the SCF result into a revised, conditional scaling hypothesis and defines the evidence and
decision points that must precede any further acquisition or reframing.*
