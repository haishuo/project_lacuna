# HANDOFF — RESET TO v1.0 (2026-06-02)  [TEMPORARY / decision record]

**Decision (by the PI):** the column-level → composition → metadata-prior → subtype → mixed-MoE
research arc is **no longer trusted** and will **not be built upon**. Reset all the way to **v1.0**,
confirm v1.0 still works, and **start fresh** from there. Nothing is deleted — the arc is preserved on
branches; we simply stop patching it.

## Repo state (what to revert to / what is preserved)
- **v1.0 = git tag `v1.0-canonical` = commit `e3f85472`** (Experiment-10 dissertation model, RUN-054).
  `main` is pristine at v1.0. **Start here.**
- The abandoned arc is on **branch `experiment/subtype-layer`** and its ancestors
  (`experiment/metadata-prior` ← `experiment/composition-stageb` ← `experiment/composition-estimand`
  ← `experiment/column-level-*`), all pushed to origin. Preserved for reference; **do not extend.**
- The working Python is conda env `lacuna` (`/home/haishuo/miniconda3/envs/lacuna/bin/python`).

## The single most important lesson (why we're resetting)
**The actual MoE network was never retrained during the entire arc.** Every stage (column head,
composition head, metadata prior, subtype layer, and the final stage-R/T scripts) either (a) froze the
v1.0 encoder and bolted on a *new* head, or (b) ran a RandomForest / MLP on hand-stats. `v1.0`'s ~92%
is the ONLY real full-MoE result. The "experiments" repeatedly tested proxies and mislabeled them as
"the network." This eroded trust in the whole arc. (Confirmed: no `scripts/stage*.py` instantiates
`MixtureOfExperts`; the real pipeline is `scripts/train.py` → `create_lacuna_model` → `Trainer`.)

## Other lessons worth carrying (not code to reuse — *cautions*)
1. **Always retrain/run the real `LacunaModel` (MoE)** for any claim about "Lacuna". A frozen-encoder
   probe or an RF-on-features is an ABLATION, never "the network." A real full-MoE run on this box is
   ~15–35 min (early-stopping); seconds/minutes ⇒ you ran a proxy.
2. **The miss-rate confound is pervasive and decisive.** v1.0 / natural-rate MAR-vs-MNAR separation
   (MNAR recall ~0.96) was *substantially the miss-rate cue*. At matched rate it collapses to ~chance
   (MAR-vs-MNAR AUC ~0.55–0.61), confirmed on the real MoE. Any MAR/MNAR claim must control rate.
3. **MAR-vs-MNAR is at/near the Molenberghs floor on realistic matched-rate data** even for the full
   MoE: it reads MCAR/structure but cannot separate MNAR from MAR (they converge — MNAR-on-correlated
   columns leaves a MAR-like coupled footprint). MNAR becomes the systematic *loser* at matched rate.
4. **Calibration = temperature scaling is argmax-invariant** (fixes ECE, never accuracy/confusion).
5. **Check the dataset pool size before trusting any eval.** `configs/training/survey.yaml` lists only
   **2** val datasets; `semisynthetic_full.yaml` gives 21 train / 7 val. Catalog has 43 datasets total.
6. **Semi-synthetic is the only ground truth** (real missingness is unrecoverable — Molenberghs + the PI's
   "missing in real life is missing forever"). Accuracy is measurable only on punched holes; real data =
   face-validity / consensus-match only.

## Conceptually-salvageable ideas (on the abandoned branches — reference only, do NOT graft in)
- Observable mask-footprint feature set + discriminator-as-realism-ruler (Stage A).
- Composition / by-cell denominator framing (ADR-0007).
- Metadata→prior authoring (local LLM → frozen auditable table) (ADR-0008 / P-arc).
- `label_collapse` + the proven fact that mixed-gen → collapsed dataset label feeds the existing MoE
  with ZERO model/Trainer/loss changes (only a wrapper loader) — if mixtures are ever revisited.

## MANDATE for the fresh session (the chip)
1. **Revert to v1.0** (`git checkout v1.0-canonical` / work off `main`); confirm a clean tree.
2. **Re-run v1.0 "Lacuna-Survey" to confirm it still trains to ~v1.0 accuracy.** NOTE: the PI recalls
   Lacuna-Survey had a *large* train/val dataset pool and is unsure whether that == `semisynthetic_full`
   (which may predate the survey specialization and include instrument/non-survey data). **First task:
   determine v1.0's ACTUAL Lacuna-Survey training config** (inspect the configs *at the v1.0 commit* and
   the RUN-054 / survey checkpoint metadata) — do not assume `survey.yaml` (its 2-dataset val list is
   suspect / possibly post-1.0). Re-run the *real* config; confirm it reproduces.
3. **Then: a painfully detailed report on HOW THE LACUNA GENERATORS FUNDAMENTALLY WORK** — not
   per-generator, but the fundamental operation: **what goes in, what comes out, how it decides which
   cells to set missing** (the `Generator.apply_to(X, rng) → mask` contract; the predictor-view
   z-scoring; how MCAR/MAR/MNAR families differ in their decision rule; the registry).
4. **Critically evaluate realism:** *is the "improved" mixed hole-punching actually more realistic?*
   Reason from first principles about **how real missingness actually arises** (survey nonresponse,
   skip logic, LOD censoring, attrition, etc.) and judge whether the generator family captures it — or
   whether the realism claim (Stage A/B) is overstated. This is the PI's core open doubt.

Files: this doc + the memory node `metadata-prior-direction` (dense arc log) on the experiment branch.
