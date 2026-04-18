"""
lacuna.analysis.ablation_harness

Sweep runner for missingness-feature ablation studies.

Given a base `LacunaConfig`, run training for each combination of
(ablation_spec, seed), evaluate the trained model, and emit a tidy
per-row record of the headline metrics. The resulting table is the
input to `lacuna.analysis.ablation_stats.paired_comparison` (and
friends) for significance testing.

Contract
--------
Input:
    - base_config  : LacunaConfig (shape of the training run; per-ablation
                     overrides are applied without mutating the input).
    - seeds        : sequence of int seeds. Each seed is used *identically*
                     across every ablation spec so paired tests are valid.
    - specs        : sequence of AblationSpec (default: DEFAULT_SPECS).

Output:
    - A list of AblationResult dataclasses, one per (spec, seed). Optionally
      written to a tidy CSV for downstream analysis.

This module is OFFLINE. It drives real training runs — it does not try to
preserve autograd or run inside a forward pass. Expect one full training
run per (spec, seed) pair.
"""

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Callable

import torch

from lacuna.config.schema import LacunaConfig
from lacuna.core.exceptions import ValidationError
from lacuna.data.catalog import create_default_catalog
from lacuna.data.missingness_features import MissingnessFeatureConfig
from lacuna.data.semisynthetic import SemiSyntheticDataLoader
from lacuna.generators.families.registry_builder import load_registry_from_config
from lacuna.generators.priors import GeneratorPrior
from lacuna.models.assembly import create_lacuna_model
from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.training.report import generate_eval_report


# =============================================================================
# Spec + result containers
# =============================================================================


@dataclass(frozen=True)
class AblationSpec:
    """One ablation configuration.

    Attributes:
        name: Short identifier used in output rows and logs. Keep filename-safe.
        feature_config: MissingnessFeatureConfig for this run. If None and
            use_missingness_features=True, the model defaults are used
            (all 5 feature groups enabled) — equivalent to the "baseline"
            spec.
        use_missingness_features: If False, the extractor is removed
            entirely — a full sanity-check ablation.
    """
    name: str
    feature_config: Optional[MissingnessFeatureConfig]
    use_missingness_features: bool = True


@dataclass(frozen=True)
class AblationResult:
    """One row of ablation output: metrics from a single (spec, seed) run.

    Attributes match the keys written to the tidy CSV.

    train_accuracy / generalization_gap are optional for backward-compat
    with CSVs written by earlier harness revisions (pre train-eval). New
    runs always populate them. A None value in either field means "not
    measured" — analysis code must handle this.
    """
    spec_name: str
    seed: int
    n_features: int
    accuracy: float
    mcar_acc: float
    mar_acc: float
    mnar_acc: float
    ece: float
    val_loss: float
    train_time_s: float
    train_accuracy: Optional[float] = None
    generalization_gap: Optional[float] = None


# =============================================================================
# Default spec list
# =============================================================================


def _spec_disable(name: str, **flags) -> AblationSpec:
    """Build a spec with all default flags on, then override with `flags`."""
    cfg = MissingnessFeatureConfig(**flags)
    return AblationSpec(name=name, feature_config=cfg)


DEFAULT_SPECS: List[AblationSpec] = [
    AblationSpec(name="baseline", feature_config=None),
    _spec_disable("disable_missing_rate", include_missing_rate_stats=False),
    _spec_disable("disable_pointbiserial", include_pointbiserial=False),
    _spec_disable("disable_cross_column", include_cross_column_corr=False),
    _spec_disable("disable_distributional", include_distributional=False),
    _spec_disable("disable_littles", include_littles_approx=False),
    AblationSpec(
        name="all_disabled",
        feature_config=None,
        use_missingness_features=False,
    ),
]


# =============================================================================
# Single-run execution
# =============================================================================


def _build_model(base_config: LacunaConfig, spec: AblationSpec):
    """Construct a fresh LacunaModel for this ablation spec.

    Same signature as `scripts/train.py` uses, with the spec's feature
    config injected.
    """
    return create_lacuna_model(
        hidden_dim=base_config.model.hidden_dim,
        evidence_dim=base_config.model.evidence_dim,
        n_layers=base_config.model.n_layers,
        n_heads=base_config.model.n_heads,
        max_cols=base_config.data.max_cols,
        dropout=base_config.model.dropout,
        use_missingness_features=spec.use_missingness_features,
        missingness_feature_config=spec.feature_config,
    )


def _load_raw_datasets_for_harness(names: Sequence[str], max_cols: int) -> list:
    """Load raw tabular datasets from the default catalog.

    Mirrors scripts/train.load_raw_datasets: datasets with
    more columns than `max_cols` are excluded (they would exceed the
    tokeniser capacity). Skips are printed to stdout so they are visible
    in run logs — this matches the existing semisynthetic training path's
    behavior, which is the reference configuration the ablation runs
    against.

    A missing dataset name IS a loud failure (catalog.load raises) — that
    indicates a config error, not a capacity limit.
    """
    catalog = create_default_catalog()
    datasets = []
    skipped = []
    for name in names:
        raw = catalog.load(name)  # raises on missing name
        if raw.d <= max_cols:
            datasets.append(raw)
        else:
            skipped.append((name, raw.d))
    if skipped:
        # Visible in logs, matches train_semisynthetic.py convention.
        print(
            f"  [ablation_harness] Skipping datasets exceeding max_cols="
            f"{max_cols}: {skipped}"
        )
    return datasets


def _build_loaders(base_config: LacunaConfig, seed: int):
    """Construct (train_loader, val_loader, registry) for one ablation run.

    Seed-derivation matches scripts/train.py (seed / seed + 1_000_000) so
    every spec sees identical data for a given seed — the paired design
    Phase 2 relies on.

    Raises:
        ValidationError: If the config lacks train_datasets or val_datasets.
    """
    generators_name = (
        base_config.generator.config_path or base_config.generator.config_name
    )
    registry = load_registry_from_config(generators_name)

    train_names = base_config.data.train_datasets
    val_names = base_config.data.val_datasets

    if not train_names:
        raise ValidationError(
            "config.data.train_datasets is required. Lacuna trains only on "
            "semi-synthetic data (real X + synthetic mechanism)."
        )
    if not val_names:
        raise ValidationError(
            "config.data.val_datasets is required whenever train_datasets is set. "
            "Semi-synthetic evaluation requires explicit held-out real datasets."
        )

    prior = GeneratorPrior.uniform(registry)
    train_raw = _load_raw_datasets_for_harness(train_names, base_config.data.max_cols)
    val_raw = _load_raw_datasets_for_harness(val_names, base_config.data.max_cols)
    if not train_raw or not val_raw:
        raise ValidationError(
            "After max_cols filtering, train_raw or val_raw is empty."
        )
    train_loader = SemiSyntheticDataLoader(
        raw_datasets=train_raw,
        registry=registry,
        prior=prior,
        max_rows=base_config.data.max_rows,
        max_cols=base_config.data.max_cols,
        batch_size=base_config.training.batch_size,
        batches_per_epoch=base_config.training.batches_per_epoch,
        seed=seed,
    )
    val_loader = SemiSyntheticDataLoader(
        raw_datasets=val_raw,
        registry=registry,
        prior=prior,
        max_rows=base_config.data.max_rows,
        max_cols=base_config.data.max_cols,
        batch_size=base_config.training.batch_size,
        batches_per_epoch=base_config.training.val_batches,
        seed=seed + 1_000_000,
    )
    return train_loader, val_loader, registry


def _build_train_eval_loader(base_config: LacunaConfig, seed: int) -> SemiSyntheticDataLoader:
    """Build an evaluation loader over TRAIN datasets with val-sized budget.

    Used to measure train-set accuracy as a memorization diagnostic. Uses
    the same dataset pool as training but with a different seed derivation
    (seed + 2_000_000) so the batches are fresh random samples — this
    measures generalization across mechanisms on the same X distributions,
    which is the relevant "train accuracy" for a semi-synthetic pipeline
    where the mechanism is re-randomised every batch.
    """
    generators_name = (
        base_config.generator.config_path or base_config.generator.config_name
    )
    registry = load_registry_from_config(generators_name)
    prior = GeneratorPrior.uniform(registry)
    train_raw = _load_raw_datasets_for_harness(
        base_config.data.train_datasets, base_config.data.max_cols
    )
    return SemiSyntheticDataLoader(
        raw_datasets=train_raw,
        registry=registry,
        prior=prior,
        max_rows=base_config.data.max_rows,
        max_cols=base_config.data.max_cols,
        batch_size=base_config.training.batch_size,
        batches_per_epoch=base_config.training.val_batches,
        seed=seed + 2_000_000,
    )


def run_single_ablation(
    base_config: LacunaConfig,
    spec: AblationSpec,
    seed: int,
    *,
    log_fn: Optional[Callable[[dict], None]] = None,
) -> AblationResult:
    """Train + evaluate one (spec, seed) combination.

    Args:
        base_config: Training config. NOT mutated.
        spec: Which ablation to run.
        seed: Seed used for torch + data loaders. Each spec sees the
            same data when seed is held constant.
        log_fn: Optional callable passed to Trainer for per-step logs.

    Returns:
        AblationResult with headline metrics.

    Raises:
        ValidationError: If the base config is structurally unsound for
            an ablation (e.g. training epochs == 0).
    """
    if base_config.training.epochs <= 0:
        raise ValidationError(
            f"base_config.training.epochs must be > 0; got {base_config.training.epochs}"
        )

    # Deterministic per-run seeding (Coding Bible rule 6).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = _build_model(base_config, spec)
    train_loader, val_loader, registry = _build_loaders(base_config, seed)

    trainer_cfg = TrainerConfig(
        lr=base_config.training.lr,
        weight_decay=base_config.training.weight_decay,
        grad_clip=base_config.training.grad_clip,
        epochs=base_config.training.epochs,
        warmup_steps=base_config.training.warmup_steps,
        patience=base_config.training.patience,
        min_delta=base_config.training.min_delta,
        # No checkpointing for ablation runs — metrics-only.
        checkpoint_dir=None,
        save_best_only=False,
    )
    trainer = Trainer(model, trainer_cfg, device=base_config.device, log_fn=log_fn)

    t0 = time.time()
    trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0

    # Detailed validation on held-out val datasets.
    detailed_val = trainer.validate_detailed(val_loader)
    report_val = generate_eval_report(detailed_val, registry=registry)

    # Train-set evaluation for the memorization diagnostic. Uses a
    # separately-built loader over the TRAIN datasets with val_batches
    # worth of fresh samples (same evaluation budget as val) — drawing
    # fresh random mechanisms and subsamples, so this measures how well
    # the model generalizes across missingness patterns on the same X
    # distributions it was trained on.
    train_eval_loader = _build_train_eval_loader(base_config, seed)
    detailed_train = trainer.validate_detailed(train_eval_loader)
    report_train = generate_eval_report(detailed_train, registry=registry)

    summary_val = report_val["summary"]
    calibration = report_val["calibration"]
    summary_train = report_train["summary"]

    val_accuracy = float(summary_val["accuracy"])
    train_accuracy = float(summary_train["accuracy"])
    generalization_gap = train_accuracy - val_accuracy

    n_features = 0
    if spec.use_missingness_features:
        cfg = spec.feature_config or MissingnessFeatureConfig()
        n_features = cfg.n_features

    return AblationResult(
        spec_name=spec.name,
        seed=seed,
        n_features=n_features,
        accuracy=val_accuracy,
        mcar_acc=float(summary_val["mcar_acc"]),
        mar_acc=float(summary_val["mar_acc"]),
        mnar_acc=float(summary_val["mnar_acc"]),
        ece=float(calibration["ece"]),
        val_loss=float(summary_val["loss"]),
        train_time_s=float(train_time),
        train_accuracy=train_accuracy,
        generalization_gap=generalization_gap,
    )


# =============================================================================
# Full sweep + CSV output
# =============================================================================


CSV_COLUMNS = [
    "spec_name",
    "seed",
    "n_features",
    "accuracy",
    "mcar_acc",
    "mar_acc",
    "mnar_acc",
    "ece",
    "val_loss",
    "train_time_s",
    "train_accuracy",
    "generalization_gap",
]

# Columns a pre-train-eval CSV is guaranteed to have. `load_ablation_csv`
# treats anything beyond this set as optional so legacy CSVs still load.
_LEGACY_CSV_COLUMNS = frozenset({
    "spec_name", "seed", "n_features", "accuracy",
    "mcar_acc", "mar_acc", "mnar_acc", "ece",
    "val_loss", "train_time_s",
})


def _write_row(csv_path: Path, row: AblationResult, write_header: bool) -> None:
    """Append a single result row to the CSV, creating the file if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with csv_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(row))


def run_ablation_sweep(
    base_config: LacunaConfig,
    seeds: Sequence[int],
    *,
    specs: Sequence[AblationSpec] = DEFAULT_SPECS,
    csv_path: Optional[Path] = None,
    on_result: Optional[Callable[[AblationResult], None]] = None,
) -> List[AblationResult]:
    """Run the full (spec × seed) cross-product.

    Writes results incrementally to CSV as each run finishes so a crash
    mid-sweep does not lose earlier runs.

    Args:
        base_config: Training config. NOT mutated.
        seeds: Seeds to use. Each spec is evaluated at every seed.
        specs: Ablation specs to run. Defaults to DEFAULT_SPECS (7 specs).
        csv_path: Optional path for incremental CSV output.
        on_result: Optional callback invoked with each AblationResult after
            it is produced (useful for live logging).

    Returns:
        Flat list of AblationResult, in run order: for each spec, for each seed.
    """
    if len(seeds) == 0:
        raise ValidationError("seeds must be non-empty")
    if len(specs) == 0:
        raise ValidationError("specs must be non-empty")

    results: List[AblationResult] = []
    first = True
    for spec in specs:
        for seed in seeds:
            result = run_single_ablation(base_config, spec, seed)
            results.append(result)
            if on_result is not None:
                on_result(result)
            if csv_path is not None:
                _write_row(Path(csv_path), result, write_header=first)
                first = False
    return results


def load_ablation_csv(csv_path: Path) -> List[AblationResult]:
    """Read a tidy CSV produced by run_ablation_sweep back into AblationResult rows.

    Backward-compatible with legacy CSVs written before train-set evaluation
    was added to the harness: `train_accuracy` and `generalization_gap`
    default to None when absent. Analysis code must handle None.
    """
    csv_path = Path(csv_path)
    out: List[AblationResult] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_acc_raw = row.get("train_accuracy", "")
            gap_raw = row.get("generalization_gap", "")
            train_accuracy = float(train_acc_raw) if train_acc_raw not in ("", None) else None
            gap = float(gap_raw) if gap_raw not in ("", None) else None
            out.append(
                AblationResult(
                    spec_name=row["spec_name"],
                    seed=int(row["seed"]),
                    n_features=int(row["n_features"]),
                    accuracy=float(row["accuracy"]),
                    mcar_acc=float(row["mcar_acc"]),
                    mar_acc=float(row["mar_acc"]),
                    mnar_acc=float(row["mnar_acc"]),
                    ece=float(row["ece"]),
                    val_loss=float(row["val_loss"]),
                    train_time_s=float(row["train_time_s"]),
                    train_accuracy=train_accuracy,
                    generalization_gap=gap,
                )
            )
    return out


def results_for(results: Sequence[AblationResult], spec_name: str, *, metric: str) -> List[float]:
    """Pull the per-seed metric values for one spec, sorted by seed.

    Returned ordering matches any other spec pulled via this helper —
    enabling paired comparisons downstream.
    """
    rows = [r for r in results if r.spec_name == spec_name]
    rows.sort(key=lambda r: r.seed)
    if not hasattr(AblationResult, metric) and metric not in {
        "accuracy", "mcar_acc", "mar_acc", "mnar_acc", "ece", "val_loss"
    }:
        raise ValueError(f"Unknown metric: {metric}")
    return [float(getattr(r, metric)) for r in rows]
