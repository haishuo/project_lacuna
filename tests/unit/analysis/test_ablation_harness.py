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
    """DEFAULT_SPECS after ADR 0004:
    baseline + baseline_legacy_mle + 2 disable_X + disable_littles alias +
    all_disabled (dissertation set), then the research-mode specs
    baseline_mom + baseline_heuristic + baseline_propensity +
    baseline_hsic + baseline_missmech.
    """
    names = [s.name for s in DEFAULT_SPECS]
    assert names[0] == "baseline"
    assert "baseline_legacy_mle" in names
    assert "disable_littles" in names
    assert "all_disabled" in names
    assert "baseline_mom" in names
    assert "baseline_heuristic" in names
    assert "baseline_propensity" in names
    assert "baseline_hsic" in names
    assert "baseline_missmech" in names
    assert len(DEFAULT_SPECS) == 11
    disable_specs = [s for s in DEFAULT_SPECS if s.name.startswith("disable_")]
    assert len(disable_specs) == 3  # disable_missing_rate, disable_cross_column, disable_littles


def test_bakeoff_specs_select_correct_cache_methods():
    """Each bakeoff spec reads its own cached scalar pair. After ADR
    0004 the default is include_littles_approx=False, so these specs now
    carry an explicit feature_config that re-enables the cached slot."""
    by_name = {s.name: s for s in DEFAULT_SPECS}
    assert by_name["baseline_propensity"].littles_method == "propensity"
    assert by_name["baseline_hsic"].littles_method == "hsic"
    assert by_name["baseline_missmech"].littles_method == "missmech"
    for name in ("baseline_propensity", "baseline_hsic", "baseline_missmech"):
        cfg = by_name[name].feature_config
        assert cfg is not None
        assert cfg.include_littles_approx is True
        assert by_name[name].use_missingness_features is True


def test_baseline_heuristic_spec_swaps_mcar_slot():
    spec = [s for s in DEFAULT_SPECS if s.name == "baseline_heuristic"][0]
    cfg = spec.feature_config
    assert cfg is not None
    assert cfg.include_littles_approx is False
    assert cfg.include_heuristic_littles is True


def test_baseline_mom_spec_uses_mom_method():
    baseline_mom = [s for s in DEFAULT_SPECS if s.name == "baseline_mom"][0]
    assert baseline_mom.littles_method == "mom"
    # Research-mode spec re-enables the cached slot (the default is off
    # after ADR 0004), so the feature_config must be explicit.
    assert baseline_mom.feature_config is not None
    assert baseline_mom.feature_config.include_littles_approx is True


def test_baseline_legacy_mle_spec_reenables_cached_slot():
    spec = [s for s in DEFAULT_SPECS if s.name == "baseline_legacy_mle"][0]
    assert spec.littles_method == "mle"
    assert spec.feature_config is not None
    assert spec.feature_config.include_littles_approx is True


def test_baseline_spec_uses_post_adr_defaults():
    # ADR 0004: baseline = the new default config (no Little's slot).
    baseline = DEFAULT_SPECS[0]
    assert baseline.name == "baseline"
    assert baseline.feature_config is None
    assert baseline.use_missingness_features is True


def test_disable_littles_alias_matches_baseline():
    # The alias resolves to the new default config — same as baseline.
    alias = [s for s in DEFAULT_SPECS if s.name == "disable_littles"][0]
    assert alias.feature_config is None
    assert alias.use_missingness_features is True


def test_all_disabled_spec():
    all_dis = [s for s in DEFAULT_SPECS if s.name == "all_disabled"][0]
    assert all_dis.use_missingness_features is False


def test_disable_specs_toggle_exactly_one_flag():
    """Each active 'disable_X' spec differs from the default in exactly
    one flag. `disable_littles` is an ADR-0004 alias for the new default
    (the cached slot is already off by default) and is exempt."""
    default = MissingnessFeatureConfig()
    default_flags = {
        "include_missing_rate_stats": default.include_missing_rate_stats,
        "include_cross_column_corr": default.include_cross_column_corr,
        "include_littles_approx": default.include_littles_approx,
    }
    for spec in DEFAULT_SPECS:
        if not spec.name.startswith("disable_"):
            continue
        if spec.name == "disable_littles":
            # Alias spec — intentionally identical to the new default.
            assert spec.feature_config is None
            continue
        cfg = spec.feature_config
        assert cfg is not None
        spec_flags = {
            "include_missing_rate_stats": cfg.include_missing_rate_stats,
            "include_cross_column_corr": cfg.include_cross_column_corr,
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


def _make_row(
    spec_name: str,
    seed: int,
    accuracy: float,
    *,
    train_accuracy: float = 0.95,
    generalization_gap: float = 0.05,
) -> AblationResult:
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
        train_accuracy=train_accuracy,
        generalization_gap=generalization_gap,
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


def test_csv_roundtrip_preserves_train_acc_and_gap(tmp_path: Path):
    """New fields round-trip cleanly through the CSV."""
    csv = tmp_path / "ablation.csv"
    row = _make_row("baseline", 1, 0.85, train_accuracy=0.97, generalization_gap=0.12)
    _write_row(csv, row, write_header=True)
    restored = load_ablation_csv(csv)
    assert restored[0].train_accuracy == pytest.approx(0.97)
    assert restored[0].generalization_gap == pytest.approx(0.12)


def test_load_legacy_csv_without_train_acc_columns(tmp_path: Path):
    """Legacy CSVs (no train_accuracy / generalization_gap columns) load with None."""
    csv = tmp_path / "legacy.csv"
    legacy_header = (
        "spec_name,seed,n_features,accuracy,mcar_acc,mar_acc,mnar_acc,"
        "ece,val_loss,train_time_s\n"
    )
    legacy_row = "baseline,1,16,0.85,0.9,0.8,0.7,0.05,0.5,12.3\n"
    csv.write_text(legacy_header + legacy_row)

    rows = load_ablation_csv(csv)
    assert len(rows) == 1
    assert rows[0].spec_name == "baseline"
    assert rows[0].accuracy == pytest.approx(0.85)
    assert rows[0].train_accuracy is None
    assert rows[0].generalization_gap is None


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
def test_end_to_end_minimal_sweep(tmp_path: Path, iris_littles_cache):
    """One seed × two specs on the minimal config — proves plumbing works."""
    base = LacunaConfig.minimal()
    # Shrink even further to keep this test under ~60s on CPU.
    base.training.epochs = 2
    base.training.batches_per_epoch = 5
    base.training.val_batches = 2
    base.training.warmup_steps = 2
    base.training.patience = 5

    specs = [
        DEFAULT_SPECS[0],  # baseline (post-ADR-0004: no Little's slot)
        [s for s in DEFAULT_SPECS if s.name == "baseline_legacy_mle"][0],
    ]

    csv = tmp_path / "ablation.csv"
    results = run_ablation_sweep(
        base_config=base,
        seeds=[7],
        specs=specs,
        csv_path=csv,
        littles_cache=iris_littles_cache,
    )
    assert len(results) == 2
    assert {r.spec_name for r in results} == {"baseline", "baseline_legacy_mle"}
    # Legacy MLE spec re-adds the 2-scalar cached slot, so it has more features.
    base_row = [r for r in results if r.spec_name == "baseline"][0]
    legacy_row = [r for r in results if r.spec_name == "baseline_legacy_mle"][0]
    assert legacy_row.n_features > base_row.n_features

    # Train-set evaluation populates the memorization-diagnostic fields.
    for r in results:
        assert r.train_accuracy is not None
        assert 0.0 <= r.train_accuracy <= 1.0
        assert r.generalization_gap is not None
        assert r.generalization_gap == pytest.approx(r.train_accuracy - r.accuracy)

    # CSV matches in-memory rows.
    restored = load_ablation_csv(csv)
    assert restored == results
