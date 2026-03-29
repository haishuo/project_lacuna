"""
Tests for the one-time migration scanner.

Uses tmp_path to create mock run directories with various artifact
combinations. Covers normal, edge, and failure cases (Rule 7).
"""

import json
import pytest
from pathlib import Path

from lacuna.experiments.registry import RunRegistry
from lacuna.experiments.migrate import scan_run_directory, migrate_existing_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dir(
    tmp_path: Path,
    name: str = "lacuna_semisyn_20260110_051028",
    meta: dict = None,
    eval_report: dict = None,
    calibrated_json: dict = None,
    config_yaml: bool = True,
) -> Path:
    """Create a mock run directory with optional artifacts."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    if config_yaml:
        (run_dir / "config.yaml").write_text("model:\n  hidden_dim: 128\n")

    if meta is not None:
        (run_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2))

    if eval_report is not None:
        (run_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2))

    if calibrated_json is not None:
        (run_dir / "checkpoints" / "calibrated.json").write_text(
            json.dumps(calibrated_json, indent=2)
        )
        # Also create calibrated.pt placeholder.
        (run_dir / "checkpoints" / "calibrated.pt").write_text("")

    return run_dir


# ---------------------------------------------------------------------------
# scan_run_directory
# ---------------------------------------------------------------------------

class TestScanRunDirectory:
    """Test metadata extraction from a single run directory."""

    def test_training_status_minimal(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, meta=None, eval_report=None)
        info = scan_run_directory(run_dir)
        assert info["status"] == "training"
        assert info["folder_path"] == str(run_dir.resolve())

    def test_evaluated_status(self, tmp_path):
        report = {
            "summary": {"accuracy": 0.85, "mar_acc": 0.70, "mnar_acc": 0.90},
        }
        run_dir = _make_run_dir(tmp_path, eval_report=report)
        info = scan_run_directory(run_dir)
        assert info["status"] == "evaluated"
        assert info["metrics"]["accuracy"] == 0.85
        assert info["metrics"]["mar_acc"] == 0.70

    def test_calibrated_status(self, tmp_path):
        report = {
            "summary": {"accuracy": 0.85},
        }
        cal = {"ece_before": 0.05, "ece_after": 0.02}
        run_dir = _make_run_dir(
            tmp_path, eval_report=report, calibrated_json=cal,
        )
        info = scan_run_directory(run_dir)
        assert info["status"] == "calibrated"
        assert info["metrics"]["ece_calibrated"] == 0.02
        assert info["metrics"]["ece_before_cal"] == 0.05

    def test_timestamp_from_meta(self, tmp_path):
        meta = {
            "timestamp": "2026-03-15T10:30:00",
            "mnar_variants": ["self_censoring", "threshold"],
            "config_path": "configs/training/semisynthetic.yaml",
        }
        run_dir = _make_run_dir(tmp_path, meta=meta)
        info = scan_run_directory(run_dir)
        assert info["timestamp"] == "2026-03-15T10:30:00"
        assert info["mnar_variants"] == ["self_censoring", "threshold"]
        assert info["n_experts"] == 4

    def test_timestamp_from_folder_name(self, tmp_path):
        run_dir = _make_run_dir(
            tmp_path,
            name="lacuna_semisyn_20260110_051028",
            meta=None,
        )
        info = scan_run_directory(run_dir)
        assert "2026-01-10" in info["timestamp"]

    def test_not_a_directory(self, tmp_path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        with pytest.raises(ValueError, match="Not a directory"):
            scan_run_directory(file_path)

    def test_non_standard_folder_name(self, tmp_path):
        """Folders without timestamp pattern still get scanned."""
        run_dir = _make_run_dir(tmp_path, name="mvp_test", meta=None)
        info = scan_run_directory(run_dir)
        assert info["status"] == "training"
        # Timestamp falls back to now (non-deterministic).
        assert info["timestamp"] != ""


# ---------------------------------------------------------------------------
# migrate_existing_runs
# ---------------------------------------------------------------------------

class TestMigrateExistingRuns:
    """Test bulk migration of run directories."""

    def test_basic_migration(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _make_run_dir(
            runs_dir,
            name="lacuna_20260110_051028",
            meta={"timestamp": "2026-01-10T05:10:28", "mnar_variants": ["self_censoring"]},
        )
        _make_run_dir(
            runs_dir,
            name="lacuna_20260110_051141",
            meta={"timestamp": "2026-01-10T05:11:41", "mnar_variants": ["self_censoring"]},
        )

        reg = RunRegistry(tmp_path / "registry.json")
        reg.load()
        new_entries = migrate_existing_runs(runs_dir, reg)

        assert len(new_entries) == 2
        assert len(reg) == 2
        assert new_entries[0].run_id == "RUN-001"
        assert new_entries[1].run_id == "RUN-002"

    def test_idempotent(self, tmp_path):
        """Running migration twice does not duplicate entries."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        _make_run_dir(runs_dir, name="lacuna_20260110_051028")

        reg = RunRegistry(tmp_path / "registry.json")
        reg.load()
        first = migrate_existing_runs(runs_dir, reg)
        second = migrate_existing_runs(runs_dir, reg)

        assert len(first) == 1
        assert len(second) == 0
        assert len(reg) == 1

    def test_nonexistent_runs_dir(self, tmp_path):
        reg = RunRegistry(tmp_path / "registry.json")
        reg.load()
        with pytest.raises(ValueError, match="does not exist"):
            migrate_existing_runs(tmp_path / "nonexistent", reg)

    def test_skips_files(self, tmp_path):
        """Non-directory entries in runs_dir are skipped."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "some_file.txt").write_text("not a run")
        _make_run_dir(runs_dir, name="lacuna_20260110_051028")

        reg = RunRegistry(tmp_path / "registry.json")
        reg.load()
        new_entries = migrate_existing_runs(runs_dir, reg)
        assert len(new_entries) == 1

    def test_chronological_ordering(self, tmp_path):
        """Runs are registered in chronological order by folder name."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        # Create in reverse order.
        _make_run_dir(runs_dir, name="lacuna_20260213_221103")
        _make_run_dir(runs_dir, name="lacuna_20260110_051028")

        reg = RunRegistry(tmp_path / "registry.json")
        reg.load()
        new_entries = migrate_existing_runs(runs_dir, reg)

        # RUN-001 should be the earlier run.
        assert new_entries[0].folder_name == "lacuna_20260110_051028"
        assert new_entries[1].folder_name == "lacuna_20260213_221103"
