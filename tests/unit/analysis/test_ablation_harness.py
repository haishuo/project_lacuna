"""
Tests for lacuna.analysis.ablation_harness

Unit tests focus on spec construction and CSV I/O. A separate end-to-end
test runs one tiny ablation on the minimal config — marked `slow` so
the main suite stays fast.

Covers:
    Normal cases:
        - DEFAULT_SPECS has one baseline, five "disable_X", and one "all_disabled".
        - Each disable_X spec toggles exactly one flag off.
        - Round-trip: run a minimal sweep, write CSV, read it back identically.
        - `results_for` returns per-seed values in seed-ascending order.
    Edge cases:
        - Empty seeds / empty specs raise.
        - A single-seed, single-spec sweep produces one row.
    Failure cases:
        - base_config with epochs == 0 raises ValidationError.
"""

from pathlib import Path

import pytest

from lacuna.analysis.ablation_harness import (
    AblationSpec,
    AblationResult,
    DEFAULT_SPECS,
    CSV_COLUMNS,
    run_single_ablation,
    run_ablation_sweep,
    load_ablation_csv,
    results_for,
    _write_row,
)
from lacuna.core.exceptions import ValidationError
from lacuna.config.schema import LacunaConfig
from lacuna.data.missingness_features import MissingnessFeatureConfig


# =============================================================================
# Spec construction
# =============================================================================


def test_default_specs_shape():
    """DEFAULT_SPECS is: baseline + 5 disable_X + all_disabled."""
    names = [s.name for s in DEFAULT_SPECS]
    assert names[0] == "baseline"
    assert names[-1] == "all_disabled"
    assert len(DEFAULT_SPECS) == 7
    disable_specs = [s for s in DEFAULT_SPECS if s.name.startswith("disable_")]
    assert len(disable_specs) == 5


def test_baseline_spec_uses_defaults():
    baseline = DEFAULT_SPECS[0]
    assert baseline.feature_config is None
    assert baseline.use_missingness_features is True


def test_all_disabled_spec():
    all_dis = [s for s in DEFAULT_SPECS if s.name == "all_disabled"][0]
    assert all_dis.use_missingness_features is False


def test_disable_specs_toggle_exactly_one_flag():
    """Each 'disable_X' spec differs from the default in exactly one flag."""
    default = MissingnessFeatureConfig()
    default_flags = {
        "include_missing_rate_stats": default.include_missing_rate_stats,
        "include_pointbiserial": default.include_pointbiserial,
        "include_cross_column_corr": default.include_cross_column_corr,
        "include_distributional": default.include_distributional,
        "include_littles_approx": default.include_littles_approx,
    }
    for spec in DEFAULT_SPECS:
        if not spec.name.startswith("disable_"):
            continue
        cfg = spec.feature_config
        assert cfg is not None
        spec_flags = {
            "include_missing_rate_stats": cfg.include_missing_rate_stats,
            "include_pointbiserial": cfg.include_pointbiserial,
            "include_cross_column_corr": cfg.include_cross_column_corr,
            "include_distributional": cfg.include_distributional,
            "include_littles_approx": cfg.include_littles_approx,
        }
        differing = [k for k in default_flags if default_flags[k] != spec_flags[k]]
        assert len(differing) == 1, (
            f"Spec {spec.name} should toggle exactly one flag; "
            f"differs in: {differing}"
        )
        assert spec_flags[differing[0]] is False
        # And n_features should be lower than the default.
        assert cfg.n_features < default.n_features


# =============================================================================
# CSV round-trip
# =============================================================================


def _make_row(spec_name: str, seed: int, accuracy: float) -> AblationResult:
    return AblationResult(
        spec_name=spec_name,
        seed=seed,
        n_features=16,
        accuracy=accuracy,
        mcar_acc=0.9,
        mar_acc=0.8,
        mnar_acc=0.7,
        ece=0.05,
        val_loss=0.5,
        train_time_s=12.3,
    )


def test_csv_roundtrip(tmp_path: Path):
    csv = tmp_path / "out" / "ablation.csv"
    rows = [
        _make_row("baseline", 1, 0.91),
        _make_row("baseline", 2, 0.90),
        _make_row("disable_littles", 1, 0.89),
        _make_row("disable_littles", 2, 0.88),
    ]
    for i, row in enumerate(rows):
        _write_row(csv, row, write_header=(i == 0))

    restored = load_ablation_csv(csv)
    assert len(restored) == len(rows)
    for orig, got in zip(rows, restored):
        assert orig == got

    # CSV header must match CSV_COLUMNS exactly.
    with csv.open() as f:
        header = f.readline().strip().split(",")
    assert header == CSV_COLUMNS


def test_results_for_sorts_by_seed():
    rows = [
        _make_row("baseline", 3, 0.91),
        _make_row("baseline", 1, 0.89),
        _make_row("disable_littles", 1, 0.80),
        _make_row("baseline", 2, 0.90),
    ]
    accs = results_for(rows, "baseline", metric="accuracy")
    assert accs == [0.89, 0.90, 0.91]


def test_results_for_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        results_for([_make_row("baseline", 1, 0.9)], "baseline", metric="nonsense")


# =============================================================================
# Failure cases on run_single_ablation
# =============================================================================


def test_zero_epochs_raises():
    base = LacunaConfig.minimal()
    base.training.epochs = 0
    with pytest.raises(ValidationError):
        run_single_ablation(base, DEFAULT_SPECS[0], seed=1)


def test_empty_seeds_raises():
    base = LacunaConfig.minimal()
    with pytest.raises(ValidationError):
        run_ablation_sweep(base, seeds=[], specs=DEFAULT_SPECS[:1])


def test_empty_specs_raises():
    base = LacunaConfig.minimal()
    with pytest.raises(ValidationError):
        run_ablation_sweep(base, seeds=[1], specs=[])


def test_missing_train_datasets_raises():
    """Lacuna trains only on semi-synthetic; missing train_datasets must fail loud."""
    base = LacunaConfig.minimal()
    base.data.train_datasets = None
    with pytest.raises(ValidationError, match="train_datasets"):
        run_single_ablation(base, DEFAULT_SPECS[0], seed=1)


def test_missing_val_datasets_raises():
    base = LacunaConfig.minimal()
    base.data.val_datasets = None
    with pytest.raises(ValidationError, match="val_datasets"):
        run_single_ablation(base, DEFAULT_SPECS[0], seed=1)


# =============================================================================
# End-to-end (slow)
# =============================================================================


@pytest.mark.slow
def test_end_to_end_minimal_sweep(tmp_path: Path):
    """One seed × two specs on the minimal config — proves plumbing works."""
    base = LacunaConfig.minimal()
    # Shrink even further to keep this test under ~60s on CPU.
    base.training.epochs = 2
    base.training.batches_per_epoch = 5
    base.training.val_batches = 2
    base.training.warmup_steps = 2
    base.training.patience = 5

    specs = [
        DEFAULT_SPECS[0],  # baseline
        [s for s in DEFAULT_SPECS if s.name == "disable_littles"][0],
    ]

    csv = tmp_path / "ablation.csv"
    results = run_ablation_sweep(
        base_config=base,
        seeds=[7],
        specs=specs,
        csv_path=csv,
    )
    assert len(results) == 2
    assert {r.spec_name for r in results} == {"baseline", "disable_littles"}
    # Baseline gate has more features than disable_littles.
    base_row = [r for r in results if r.spec_name == "baseline"][0]
    dis_row = [r for r in results if r.spec_name == "disable_littles"][0]
    assert base_row.n_features > dis_row.n_features

    # CSV matches in-memory rows.
    restored = load_ablation_csv(csv)
    assert restored == results
