"""
Tests for RunEntry validation and RunRegistry CRUD / persistence.

Covers normal cases, edge cases, and failure cases (Rule 7).
All tests use tmp_path and are independently runnable.
"""

import json
import pytest
from pathlib import Path

from lacuna.experiments.registry import (
    RunEntry,
    RunRegistry,
    RegistryError,
    VALID_STATUSES,
)


# ---------------------------------------------------------------------------
# RunEntry validation
# ---------------------------------------------------------------------------

class TestRunEntryValidation:
    """RunEntry __post_init__ validation (Rule 1 -- fail fast)."""

    def test_valid_entry(self):
        entry = RunEntry(
            run_id="RUN-001",
            folder_name="my_run",
            folder_path="/tmp/my_run",
            timestamp="2026-01-01T00:00:00",
            config_path="config.yaml",
            status="training",
            description="test run",
        )
        assert entry.run_id == "RUN-001"

    def test_valid_high_number_id(self):
        entry = RunEntry(
            run_id="RUN-9999",
            folder_name="run",
            folder_path="/tmp/run",
            timestamp="2026-01-01T00:00:00",
            config_path="",
            status="evaluated",
            description="",
        )
        assert entry.run_id == "RUN-9999"

    def test_invalid_run_id_no_prefix(self):
        with pytest.raises(RegistryError, match="Invalid run_id"):
            RunEntry(
                run_id="001",
                folder_name="run",
                folder_path="/tmp/run",
                timestamp="",
                config_path="",
                status="training",
                description="",
            )

    def test_invalid_run_id_too_few_digits(self):
        with pytest.raises(RegistryError, match="Invalid run_id"):
            RunEntry(
                run_id="RUN-01",
                folder_name="run",
                folder_path="/tmp/run",
                timestamp="",
                config_path="",
                status="training",
                description="",
            )

    def test_invalid_status(self):
        with pytest.raises(RegistryError, match="Invalid status"):
            RunEntry(
                run_id="RUN-001",
                folder_name="run",
                folder_path="/tmp/run",
                timestamp="",
                config_path="",
                status="unknown",
                description="",
            )

    def test_empty_folder_name(self):
        with pytest.raises(RegistryError, match="folder_name must not be empty"):
            RunEntry(
                run_id="RUN-001",
                folder_name="",
                folder_path="/tmp/run",
                timestamp="",
                config_path="",
                status="training",
                description="",
            )

    def test_empty_folder_path(self):
        with pytest.raises(RegistryError, match="folder_path must not be empty"):
            RunEntry(
                run_id="RUN-001",
                folder_name="run",
                folder_path="",
                timestamp="",
                config_path="",
                status="training",
                description="",
            )

    @pytest.mark.parametrize("status", VALID_STATUSES)
    def test_all_valid_statuses(self, status):
        entry = RunEntry(
            run_id="RUN-001",
            folder_name="run",
            folder_path="/tmp/run",
            timestamp="",
            config_path="",
            status=status,
            description="",
        )
        assert entry.status == status


class TestRunEntrySerialization:
    """to_dict / from_dict round-trip."""

    def test_round_trip(self):
        original = RunEntry(
            run_id="RUN-042",
            folder_name="lacuna_semisyn_20260101_120000",
            folder_path="/tmp/runs/lacuna_semisyn_20260101_120000",
            timestamp="2026-01-01T12:00:00",
            config_path="configs/training/semisynthetic.yaml",
            status="evaluated",
            description="test",
            metrics={"accuracy": 0.85, "mar_acc": 0.70},
            mnar_variants=["self_censoring", "threshold"],
            n_experts=4,
            tags=["ablation"],
        )
        d = original.to_dict()
        restored = RunEntry.from_dict(d)
        assert restored.run_id == original.run_id
        assert restored.metrics == original.metrics
        assert restored.mnar_variants == original.mnar_variants
        assert restored.tags == original.tags

    def test_from_dict_missing_key(self):
        with pytest.raises(RegistryError, match="Missing required keys"):
            RunEntry.from_dict({"run_id": "RUN-001"})

    def test_from_dict_defaults(self):
        data = {
            "run_id": "RUN-001",
            "folder_name": "run",
            "folder_path": "/tmp/run",
            "timestamp": "",
            "config_path": "",
            "status": "training",
            "description": "",
        }
        entry = RunEntry.from_dict(data)
        assert entry.metrics == {}
        assert entry.mnar_variants == []
        assert entry.tags == []
        assert entry.n_experts == 0


# ---------------------------------------------------------------------------
# RunRegistry CRUD
# ---------------------------------------------------------------------------

class TestRunRegistry:
    """RunRegistry load/save/CRUD operations."""

    def _make_registry(self, tmp_path: Path) -> RunRegistry:
        reg = RunRegistry(tmp_path / "registry.json")
        reg.load()
        return reg

    def test_empty_registry(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert len(reg) == 0
        assert reg.all_entries() == []

    def test_register_and_get(self, tmp_path):
        reg = self._make_registry(tmp_path)
        entry = reg.register(
            folder_path="/tmp/run_a",
            timestamp="2026-01-01T00:00:00",
            status="training",
        )
        assert entry.run_id == "RUN-001"
        assert len(reg) == 1
        assert reg.get("RUN-001") is entry

    def test_sequential_ids(self, tmp_path):
        reg = self._make_registry(tmp_path)
        e1 = reg.register(folder_path="/tmp/r1", timestamp="t1")
        e2 = reg.register(folder_path="/tmp/r2", timestamp="t2")
        e3 = reg.register(folder_path="/tmp/r3", timestamp="t3")
        assert e1.run_id == "RUN-001"
        assert e2.run_id == "RUN-002"
        assert e3.run_id == "RUN-003"

    def test_get_missing_raises(self, tmp_path):
        reg = self._make_registry(tmp_path)
        with pytest.raises(RegistryError, match="not found"):
            reg.get("RUN-999")

    def test_update(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(folder_path="/tmp/r1", timestamp="t1", status="training")
        updated = reg.update("RUN-001", status="evaluated", metrics={"accuracy": 0.9})
        assert updated.status == "evaluated"
        assert updated.metrics == {"accuracy": 0.9}

    def test_update_invalid_status(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(folder_path="/tmp/r1", timestamp="t1")
        with pytest.raises(RegistryError, match="Invalid status"):
            reg.update("RUN-001", status="bogus")

    def test_update_nonexistent_field(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(folder_path="/tmp/r1", timestamp="t1")
        with pytest.raises(RegistryError, match="no attribute"):
            reg.update("RUN-001", nonexistent_field="value")

    def test_find_by_folder(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(folder_path="/tmp/runs/abc", timestamp="t1")
        reg.register(folder_path="/tmp/runs/def", timestamp="t2")
        found = reg.find_by_folder("abc")
        assert found is not None
        assert found.run_id == "RUN-001"
        assert reg.find_by_folder("nonexistent") is None

    def test_all_entries_sorted(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.register(folder_path="/tmp/r1", timestamp="t1")
        reg.register(folder_path="/tmp/r2", timestamp="t2")
        reg.register(folder_path="/tmp/r3", timestamp="t3")
        entries = reg.all_entries()
        ids = [e.run_id for e in entries]
        assert ids == ["RUN-001", "RUN-002", "RUN-003"]


class TestRunRegistryPersistence:
    """Atomic save and reload (Rule 1 -- no silent data loss)."""

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "registry.json"

        reg1 = RunRegistry(path)
        reg1.load()
        reg1.register(folder_path="/tmp/r1", timestamp="t1", description="first")
        reg1.register(folder_path="/tmp/r2", timestamp="t2", description="second")

        # Reload from scratch.
        reg2 = RunRegistry(path)
        reg2.load()
        assert len(reg2) == 2
        assert reg2.get("RUN-001").description == "first"
        assert reg2.get("RUN-002").description == "second"

    def test_atomic_write_no_tmp_left(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = RunRegistry(path)
        reg.load()
        reg.register(folder_path="/tmp/r1", timestamp="t1")
        # .tmp file should not remain after save.
        tmp_file = path.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("not valid json")
        reg = RunRegistry(path)
        with pytest.raises(RegistryError, match="Failed to read"):
            reg.load()

    def test_load_missing_runs_key_raises(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"other": []}))
        reg = RunRegistry(path)
        with pytest.raises(RegistryError, match="expected top-level"):
            reg.load()

    def test_load_nonexistent_creates_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        reg = RunRegistry(path)
        reg.load()
        assert len(reg) == 0

    def test_next_id_after_reload(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = RunRegistry(path)
        reg.load()
        reg.register(folder_path="/tmp/r1", timestamp="t1")
        reg.register(folder_path="/tmp/r2", timestamp="t2")

        reg2 = RunRegistry(path)
        reg2.load()
        assert reg2.next_id() == "RUN-003"
