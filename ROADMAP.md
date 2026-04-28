# Lacuna Roadmap

A forward-looking plan for getting Lacuna from its current pre-alpha
research state to a publishable, pip-installable library and beyond. This
document is for orientation only — it is *not* a commitment to specific
calendar dates and does *not* prescribe immediate work. Its purpose is to
make architectural priors visible so that ongoing research work doesn't
paint the project into a corner that's expensive to escape later.

For a snapshot of the current technical state see `docs/ACADEMIC_REFERENCE.md`.
For the iteration history see `docs/decisions/0005-lacuna-survey-iteration-arc.md`.

---

## 1. Current State and Posture

**Pre-alpha. Research mode.** The codebase is shaped for active iteration
on the trained model, the synthetic generator distribution, and the real-
anchor calibration corpus. It is *not* shaped for external users.

Concretely:

- The public API is "read the source." There is no stable entry point;
  `demo/pipeline.py:run_model` is the closest thing to one and its return
  shape has changed several times during recent work.
- Documentation is academic (`docs/decisions/`, `docs/experiments/`,
  ADR 0005's iteration trail) — load-bearing for the dissertation and for
  internal honesty, but not user-facing.
- Model artifacts (`demo/model.pt`, `lacuna_survey/deployment/calibration.json`)
  are not versioned independently of the code; mismatched code/weights
  silently produce wrong answers.
- Configuration is YAML files at hardcoded paths.
- The demo is a Streamlit app intended for live demonstration, not a CLI
  or library.

This is correct for the current phase. The project is in service of (a) a
dissertation and (b) a longer-term commercial/industrial ambition. Shipping
a half-finished product would compromise both. The decision is "when," not
"whether."

## 2. Architectural Priors

Decisions that shape the codebase *now* even though productization happens
later. Each of these is either already in effect or should be enforced as a
default going forward.

### 2.1 Variant Package as Primary Unit

Lacuna-Survey is a *variant package*, not a feature flag. Future variants
(Lacuna-Records, Lacuna-Longitudinal, Lacuna-Sensor) follow the same
pattern: their own anchors, their own generator registry, their own
calibration corpus, their own training config, optionally their own
demo entry point. Base `lacuna/` provides the encoder, MoE, training loop,
generator framework — nothing variant-specific.

The pip-installable shape this implies:
- `lacuna-core` — base framework.
- `lacuna-survey` — survey variant. Depends on `lacuna-core`.
- `lacuna-records`, `lacuna-longitudinal`, etc. — future variants.

**Practical rule:** when adding a survey-specific helper, put it in
`lacuna_survey/`. Resist the temptation to put it in `lacuna/` "in case it's
useful." Cross-pollination should happen via promotion, not by default.

### 2.2 Output Schema Stability

Currently the output of inference is an untyped dict. Long-term it must be
a typed result object with a `schema_version` field and semver guarantees.

**Practical rule:** add `schema_version: 0` to the inference output now,
even though the schema is still mutating. Locks in the discipline of
treating the output as a versioned contract rather than a free-form dict.

### 2.3 Model Artifact Distribution

Model weights, calibration parameters, and OOD parameters (when used) are
versioned independently of code. The intended distribution path is
**Hugging Face Hub** with the `from_pretrained()` pattern — this is the
modern convention and generalises trivially across multiple variants.

A typical 1.0 call:

```python
classifier = lacuna_survey.SurveyClassifier.from_pretrained(
    "yourname/lacuna-survey", revision="v11"
)
```

Alternatives considered:
- *Bundle weights in the wheel.* Workable for one variant, breaks down at
  multiple. Also bloats the install for users who don't need GPU support.
- *Lazy download from a custom CDN.* Reinvents Hugging Face Hub badly.

### 2.4 Configuration as Object, Not Path

Currently `_CALIBRATION_PATH = Path(__file__).parent.parent / "lacuna_survey" / ...`
is a module-level constant. For a library this becomes a constructor
argument or a config object the user can override. Hardcoded paths in
business logic make tests harder, deployment harder, and forking harder.

**Practical rule:** when refactoring for 1.0, all paths flow through
explicit constructor arguments with sensible defaults. Module-level path
constants are a smell.

### 2.5 License: Apache 2.0

For commercial-friendly distribution with patent grants, **Apache 2.0** is
the standard choice. PyTorch (BSD-3), Hugging Face Transformers (Apache 2.0),
TensorFlow (Apache 2.0), scikit-learn (BSD-3) — Lacuna will sit alongside
these in dependency trees and should match the license expectations of that
ecosystem.

MIT is also acceptable. GPL or AGPL would force commercial users into open-
source obligations and is incompatible with the stated commercial ambition.

**Action item, low cost:** drop a `LICENSE` file with Apache 2.0 text and
a `NOTICE` file at repo root. Doesn't change the code; clarifies intent for
any future contributor or commercial partner.

### 2.6 Branding and Trademark

"Lacuna" is a common English word. A search of PyPI, GitHub, and
trademark databases will likely turn up unrelated projects with the name.
For commercial branding, options:

- Keep "Lacuna" as the project name; pick a different package name (e.g.
  `pylacuna`, `project-lacuna`, `lacuna-ml`).
- Pick a portmanteau or compound name that is more uniquely searchable.
- Defer the branding decision to a separate exercise with legal advice.

Worth doing once before public PyPI publication; not worth doing now.

## 3. The 1.0 Deliverables Checklist

What "shippable as a v1.0 library" actually means. None of these need to
exist before research is complete; they're the focused-sprint output.

### 3.1 API Surface

- [ ] One-line entry point: `lacuna_survey.predict(df) -> Posterior`.
- [ ] Power-user entry point: `SurveyClassifier(...)` class with
  `from_pretrained`, configurable calibration, batch inference.
- [ ] Typed result objects (dataclass or pydantic) with `schema_version`.
- [ ] All paths configurable via constructor args; no hardcoded paths.
- [ ] Public `__all__` declarations on every module so star-imports are
  predictable.
- [ ] No surprise dependencies on training-time modules at inference time.

### 3.2 Model Artifact Story

- [ ] Weights uploaded to Hugging Face Hub under a stable identifier.
- [ ] `from_pretrained("...", revision="v11")` works end-to-end on a
  fresh machine with only `pip install lacuna-survey`.
- [ ] Calibration parameters versioned with the weights.
- [ ] Local-cache fallback for offline use.
- [ ] Checksums verified on download.

### 3.3 Reproducibility

- [ ] Bit-for-bit reproducible inference: identical output given identical
  input + model version + seed across CPU. (GPU determinism is harder;
  acceptable to relax for GPU paths and document.)
- [ ] CI runs inference smoke tests against published checkpoints.
- [ ] `lacuna-survey --version` reports both code version and weights
  version.

### 3.4 User Documentation (separate from academic record)

- [ ] `README.md` — 60-second pitch + install + minimal example.
- [ ] `docs/getting_started.md` — 15-minute tutorial.
- [ ] `docs/api.md` — autogenerated API reference.
- [ ] `docs/theory.md` — short, accessible primer on missingness mechanisms
  and what Lacuna is computing. Aimed at applied users, not methodologists.
- [ ] `docs/migration.md` — version-to-version migration guide (starts
  empty; populated from v1.0 onward).

The current `docs/decisions/`, `docs/experiments/`, and `docs/ACADEMIC_REFERENCE.md`
move to `docs/research/` or a separate `lacuna-research-archive` repository.

### 3.5 Quality

- [ ] Test suite passes on CPU and GPU with deterministic seeding.
- [ ] CI matrix covers Python 3.10, 3.11, 3.12 (or whatever's current).
- [ ] Type checking clean under `mypy --strict` for the public API.
- [ ] Inference latency budget documented (currently ~50-200 ms per dataset
  on CPU; should be benchmarked formally before shipping).
- [ ] Memory footprint documented (currently ~11 MB checkpoint + working
  memory; should be characterised across input sizes).

### 3.6 Distribution

- [ ] PyPI namespace claimed and held by a placeholder before any active
  publication.
- [ ] First public release tagged `0.1.0` (research preview) with explicit
  "API is unstable" notice.
- [ ] `1.0.0` only after one release cycle of `0.x` with no breaking
  changes — minimum 1 month of stability before declaring stable.

## 4. Phased Release Plan

A rough sketch. Calendar dates deliberately omitted; phases gate on
dissertation progress and capability completeness.

### Phase 0 — Research mode (current)

Active iteration on model, generators, calibration, anchor corpus.
Goal: dissertation-defensible scientific contribution. No external users.

### Phase 1 — Architectural lockdown

Trigger: variant model accuracy and probabilistic calibration is
considered "feature complete" relative to dissertation defense, with no
expected further architectural retraining beyond minor refits.

Work:
- Implement the architectural priors above (output schema, paths-as-args,
  variant package boundary).
- Add `LICENSE` and `NOTICE`.
- Lock the public API surface (decide what's `lacuna_survey.predict`).
- Reserve PyPI and Hugging Face namespaces.
- Tag a `0.1.0-rc` for internal review only.

Estimated effort: 1-2 weeks focused work, post-research.

### Phase 2 — Public preview (`0.1.0` → `0.x`)

Trigger: Phase 1 complete and a small group of beta users (academic
collaborators, methodology peers) is willing to evaluate.

Work:
- Publish to PyPI as `0.1.0` ("research preview, API unstable").
- Publish weights to Hugging Face Hub.
- Collect feedback. Expect API changes between minor versions.
- No marketing; word-of-mouth and conference adjacency only.

Estimated duration: 1-3 months in `0.x`.

### Phase 3 — Stable library (`1.0.0`)

Trigger: one minor-version cycle (e.g. `0.5.0`) with no breaking changes
from the prior minor version, indicating the API has stabilised.

Work:
- Tag `1.0.0` with semver guarantees.
- Polish user documentation; produce a Getting Started tutorial.
- Migrate the academic dossier (`docs/research/` or separate repo).
- Announce more broadly (e.g. methodology mailing lists, methodologically-
  adjacent conference talks).

### Phase 4 — Beyond 1.0

Open-ended. Possible directions:
- Additional variants (Lacuna-Records, Lacuna-Longitudinal, Lacuna-Sensor).
- Hosted SaaS for users who don't want to install (probably last priority).
- Integration with `mice`, `Amelia II`, or similar imputation libraries
  via a "Lacuna recommends" extension.
- Extension to time-series and longitudinal data with attention masks
  appropriate to those domains.

Each of these is its own scoping exercise, not part of the v1.0 plan.

## 5. Open Decisions

Things that need a deliberate call before Phase 1 starts. Listed without
prescribed resolution; flagged so they aren't decided by accident.

### 5.1 Single package or split package layout?

Option A: monolithic `lacuna` package, with `lacuna.survey`, `lacuna.records`,
etc. as submodules. Simpler distribution, larger install.

Option B: split packages (`lacuna-core`, `lacuna-survey`, etc.) with
inter-package dependencies. More complex distribution, smaller installs,
cleaner failure modes.

The split package model scales better but is more work to set up.

### 5.2 Where do the weights live?

Hugging Face Hub is the strong default. Alternatives include a self-hosted
S3 bucket (more control, more ops burden) or bundling into the wheel
(simpler, doesn't scale to multiple variants).

The HF Hub path is recommended unless there's a specific reason against it
(e.g. enterprise air-gap requirements that come up later).

### 5.3 Project name and namespace

"Lacuna" is the working name. For a public release the questions are:

- Is the package called `lacuna`, `pylacuna`, `project-lacuna`, or
  something else?
- Is the project's marketing name "Lacuna" or something more
  search-distinctive?
- Should the name be trademarked, and if so, in what jurisdictions?

These are commercial-strategy questions; defer to the right moment, not
to the casual default.

### 5.4 Inference-time dependency on training infrastructure

Currently `lacuna_survey/calibrate.py` and `lacuna_survey/ood.py` import
heavy training modules (catalog, registry builder, semisynthetic). For
inference-only users, these imports should not be required.

Practical implication: when refactoring for 1.0, separate the inference
codepath from the training codepath at the import level. Inference users
should be able to install a minimal `lacuna-survey-inference` and not pull
in the full training stack.

### 5.5 GPU vs CPU as the default deployment target

The current model runs in ~50-200 ms on CPU per inference. GPU is faster
but not necessary. The deployment story should clarify:

- CPU-only is the default; GPU is an optional acceleration.
- Determinism guarantees may differ between CPU and GPU; document if so.
- The wheel does not require a CUDA install; PyTorch handles that.

This is mostly a documentation decision but it sets user expectations.

## 6. Things This Roadmap Deliberately Doesn't Cover

- **Pricing or commercialisation model.** Out of scope for a code roadmap.
  Commercial strategy is a separate exercise involving market, customer,
  and legal considerations.
- **Hiring or team scaling.** Same reason.
- **Specific feature requests beyond v1.0.** The roadmap above is a
  capability roadmap, not a feature wishlist. Features get scoped against
  user demand and dissertation-cycle priorities, not pre-committed here.
- **Conference / publication strategy.** Adjacent to Phase 3 but is a
  separate planning exercise.

---

*Document history:*
- 2026-04-28: Initial draft, post v11. Captures the productization plan
  agreed in conversation; non-binding orientation document.
