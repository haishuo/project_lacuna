# Lacuna Refactoring Checklist

This document tracks all refactoring work needed to bring Lacuna into full compliance
with the coding standards defined in `CLAUDE.md`. Each item includes the rule violated,
what needs to change, and a suggested decomposition.

**Line counts are CODE ONLY** — comments, docstrings, and blank lines are excluded per
CLAUDE.md Rule 4: "comments and docstrings excluded."

**Environment note:** Lacuna runs on Forge (Nvidia GPU Linux box). Current dev environment
is macOS without RTX GPU. Refactors that change logic or interfaces must be validated on
Forge before merging. Pure structural splits (no logic changes) can be validated via
`pytest` on any machine with PyTorch CPU.

---

## Rule 4: LOC Limits (Hard limit: 500, Soft limit: 400)

### Hard Limit Violations (>500 LOC)

Only one file exceeds the hard limit:

- [ ] **`lacuna/training/trainer.py`** — 585 LOC (1088 total)
  - Split into:
    - `trainer.py` — TrainerConfig, Trainer.__init__, Trainer.train() (main loop shell)
    - `scheduling.py` — LR scheduling logic (cosine, linear, warmup)
    - `early_stopping.py` — EarlyStopping class and patience logic
    - `training_step.py` — Single train step, validation step, gradient clipping
  - Tests: `tests/unit/training/test_trainer.py` will also need splitting to match

### Soft Limit Zone (400-500 LOC) — Split When Touched

These files are approaching or in the 400-500 soft limit zone. They should be
actively considered for splitting when modified.

- [ ] `scripts/generate_dissertation_figures.py` — 431 LOC (589 total)
  - Consider splitting by figure type (bar charts, confusion matrices, LaTeX tables)
- [ ] `lacuna/models/assembly.py` — 415 LOC (716 total)
  - Consider extracting model builder/factory logic from orchestration
- [ ] `lacuna/generators/families/mar/multiple.py` — 370 LOC (608 total)
  - Multiple generator classes; could split by variant
- [ ] `scripts/download_datasets.py` — 368 LOC (509 total)
- [ ] `lacuna/generators/families/mnar/self_censoring.py` — 356 LOC (648 total)
- [ ] `scripts/diagnose_reconstruction.py` — 345 LOC (532 total)
- [ ] `lacuna/training/loss.py` — 338 LOC (918 total)
- [ ] `scripts/train_semisynthetic.py` — 332 LOC (448 total)
- [ ] `scripts/consolidate_results.py` — 332 LOC (394 total)
- [ ] `lacuna/training/report.py` — 315 LOC (570 total)

### Compliant Files (previously flagged, now confirmed OK)

These files looked large by total line count but are well under 500 LOC:

| File | Total Lines | Code Lines | Status |
|------|-------------|------------|--------|
| `training/loss.py` | 918 | 338 | OK |
| `models/encoder.py` | 752 | 262 | OK |
| `data/missingness_features.py` | 716 | 300 | OK |
| `training/checkpoint.py` | 648 | 236 | OK |
| `data/tokenization.py` | 571 | 195 | OK |
| `data/batching.py` | 523 | 244 | OK |
| `models/moe.py` | 511 | 258 | OK |
| `training/report.py` | 570 | 315 | OK |
| `reconstruction/heads.py` | 491 | 168 | OK |
| `reconstruction/__init__.py` | 469 | 171 | OK |

---

## Rule 3: One Module, One Job

- [ ] **`lacuna/models/reconstruction/__init__.py`** — 171 LOC of logic in __init__
  - `__init__.py` should primarily re-export. Move reconstruction orchestration logic
    to a dedicated file (e.g., `reconstruction/manager.py` or `reconstruction/pipeline.py`)

- [ ] **`lacuna/data/__init__.py`** (163 total lines) — Review for logic vs. re-exports
  - Verify this is only re-exporting, not implementing logic

---

## Dead Code Removal

- [ ] **`MNARThresholdHead`** and **`MNARLatentHead`** in `lacuna/models/reconstruction/heads.py`
  - These head classes are defined but never instantiated. The default (and only used)
    configuration is `mnar_variants=["self_censoring"]`, giving 3 heads: MCAR, MAR,
    MNAR-SelfCensoring. The threshold and latent heads are dead code.
  - Remove the classes, their entries in `HEAD_REGISTRY`, their imports/exports in
    `reconstruction/__init__.py`, and any test coverage that exists solely for them.
  - After removal, `heads.py` drops from 168 LOC to roughly ~100 LOC.

---

## Rule 7: Tests Are First-Class Citizens

### Missing or Placeholder Modules

- [ ] **`lacuna/cli/`** — Module exists with placeholder files (train.py, infer.py,
  eval.py). Either implement with tests or remove.

---

## Logging and Metrics Infrastructure

The current logging and metrics state is inadequate for producing presentation-quality
output (loss curves, AUC-over-epoch, calibration progression, etc.) without manual
post-hoc data wrangling.

### Current State

- `lacuna/core/logging.py` — empty stub (one-line docstring)
- `lacuna/training/logging.py` — minimal: prints epoch summaries to console, appends
  raw metric dicts as text lines to `logs/training.log`
- `lacuna/metrics/` — four files, all empty stubs (calibration.py, classification.py,
  uncertainty.py, __init__.py)
- `scripts/train_semisynthetic.py` — has a `--wandb` flag that is never wired up (dead code)
- Dissertation figures are generated post-hoc by loading consolidated JSON/`.pt` files
  and manually extracting nested dict values

### What Needs to Happen

- [ ] **Implement `lacuna/metrics/`** — The stub files already outline the right
  decomposition (calibration, classification, uncertainty). These should contain
  reusable metric computation functions that are currently scattered across
  `training/report.py` and various scripts. Extract, don't duplicate.

- [ ] **Implement structured training logging** — Replace the text-append logger in
  `lacuna/training/logging.py` with a structured system that:
  - Records per-step and per-epoch metrics in a machine-readable format (e.g.,
    JSON lines or CSV) that can be trivially loaded into pandas/matplotlib
  - Tracks at minimum: train loss (total + components), val loss, val accuracy
    (overall + per-class), learning rate, ECE, per-class precision/recall
  - Is a clean callback interface so the Trainer doesn't need to know the storage format

- [ ] **Implement `lacuna/core/logging.py`** — Decide scope: if this is meant to be
  the project-wide structured logger (vs. `training/logging.py` which is training-specific),
  implement it. If redundant with training logging, remove it.

- [ ] **Remove dead `--wandb` flag** from `scripts/train_semisynthetic.py`

- [ ] **Add visualization utilities** — Either in `lacuna/metrics/` or a new
  `lacuna/visualization/` module, provide functions that consume the structured logs
  and produce publication-ready figures:
  - Loss curves (train + val, with component breakdown)
  - Per-class accuracy over epochs
  - AUC/ROC curves (currently only in `scripts/generate_roc_curves.py`)
  - Calibration progression (ECE over epochs)
  - Confusion matrix heatmaps
  - This consolidates the figure logic currently spread across 3+ scripts

### Design Considerations

- The logging system must follow Rule 5 (No Hidden State): no global logger singletons.
  Pass the logger explicitly to the Trainer and any other consumer.
- The logging system must follow Rule 6 (Deterministic): logging itself has no
  non-deterministic behavior, but timestamping log entries should use an injectable
  clock if timestamps are included.
- Per Rule 7: every new module needs tests. Metric computation functions are pure
  functions and straightforward to test. Visualization functions can be tested by
  verifying they produce valid figure objects without rendering.

---

## Rules 1, 2, 5, 6: Audit Required

The codebase appears to generally comply with Rules 1 (Fail Fast), 2 (Validate at
Boundaries), 5 (No Hidden State), and 6 (Deterministic). A thorough audit should
be performed. Specific items to check:

- [ ] **Rule 1 audit** — Grep for bare `except:`, `pass` in except blocks, functions
  returning `None` to indicate errors instead of raising. Check that all validation
  failures raise descriptive exceptions.

- [ ] **Rule 2 audit** — Verify all external inputs (CSV loading, config loading, CLI
  args, environment variables) are validated at entry points. Check scripts/ especially,
  as they are boundary code.

- [ ] **Rule 5 audit** — Grep for module-level mutable state, global variables. Check
  that no function behavior depends on hidden state not in its signature.

- [ ] **Rule 6 audit** — Verify all non-deterministic operations (random sampling, time
  calls) are documented with `# NON-DETERMINISTIC:` comments and have injectable
  seeds/clocks. Check scripts/ for undocumented time.time() or random usage.

---

## Recommended Execution Order

1. **Split `trainer.py`** — The only hard limit violation. This is the mandatory fix.

2. **Remove dead code** — Delete `MNARThresholdHead`, `MNARLatentHead`, and the
   dead `--wandb` flag.

3. **Implement logging and metrics** — This is new functionality, not just cleanup.
   Build the structured logging system, implement `lacuna/metrics/`, and add
   visualization utilities. This is the highest-value work for committee presentations.

4. **Run Rule 1/2/5/6 audits** — These may reveal issues independent of file size.

5. **Address Rule 3** — Move logic out of `reconstruction/__init__.py`.

6. **Clean up remaining placeholders** — `lacuna/cli/`.

7. **Validate on Forge** — Run full test suite to confirm nothing broke.

---

## Constraints

- **No logic changes during structural splits.** Each split commit should be a pure
  refactor: move code, update imports, verify tests pass. Logic changes go in separate
  commits.
- **Maintain public API.** All `__init__.py` re-exports must continue to work so that
  existing scripts and tests are not broken by internal restructuring.
- **One module = one test file.** After splitting, every new module file must have a
  corresponding test file.
