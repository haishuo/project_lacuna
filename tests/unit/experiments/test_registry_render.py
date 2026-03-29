"""
Tests for Markdown rendering of the run registry.

Covers output format, empty registry, metric formatting (Rule 7).
"""

import pytest
from pathlib import Path

from lacuna.experiments.registry import RunRegistry
from lacuna.experiments.registry_render import (
    render_registry_markdown,
    write_registry_markdown,
)


def _populate_registry(tmp_path: Path) -> RunRegistry:
    """Create a registry with a couple of entries for testing."""
    reg = RunRegistry(tmp_path / "registry.json")
    reg.load()
    reg.register(
        folder_path="/tmp/runs/run_a",
        timestamp="2026-01-15T10:30:00",
        status="evaluated",
        description="First run",
        metrics={"accuracy": 0.85, "mar_acc": 0.70, "ece": 0.045},
    )
    reg.register(
        folder_path="/tmp/runs/run_b",
        timestamp="2026-02-20T14:00:00",
        status="training",
        description="Second run",
    )
    return reg


class TestRenderRegistryMarkdown:
    """Test the rendered Markdown content."""

    def test_auto_generated_comment(self, tmp_path):
        reg = _populate_registry(tmp_path)
        md = render_registry_markdown(reg)
        assert md.startswith("<!-- AUTO-GENERATED from registry.json.")

    def test_title_present(self, tmp_path):
        reg = _populate_registry(tmp_path)
        md = render_registry_markdown(reg)
        assert "# Lacuna Run Registry" in md

    def test_total_count(self, tmp_path):
        reg = _populate_registry(tmp_path)
        md = render_registry_markdown(reg)
        assert "Total runs: 2" in md

    def test_table_header(self, tmp_path):
        reg = _populate_registry(tmp_path)
        md = render_registry_markdown(reg)
        assert "| Run ID | Date | Status | Accuracy | MAR Acc | ECE | Description |" in md

    def test_entry_rows(self, tmp_path):
        reg = _populate_registry(tmp_path)
        md = render_registry_markdown(reg)
        assert "RUN-001" in md
        assert "RUN-002" in md
        assert "2026-01-15" in md
        assert "evaluated" in md
        assert "85.0%" in md
        assert "70.0%" in md
        assert "0.0450" in md

    def test_missing_metrics_show_dash(self, tmp_path):
        reg = _populate_registry(tmp_path)
        md = render_registry_markdown(reg)
        lines = md.split("\n")
        # RUN-002 has no metrics -> dashes
        run002_line = [l for l in lines if "RUN-002" in l][0]
        # Should contain dashes for accuracy, mar_acc, ece
        assert run002_line.count("- |") >= 3

    def test_empty_registry(self, tmp_path):
        reg = RunRegistry(tmp_path / "empty.json")
        reg.load()
        md = render_registry_markdown(reg)
        assert "Total runs: 0" in md
        # Should still have header row but no data rows
        assert "RUN-" not in md

    def test_long_description_truncated(self, tmp_path):
        reg = RunRegistry(tmp_path / "reg.json")
        reg.load()
        long_desc = "A" * 100
        reg.register(
            folder_path="/tmp/runs/long_desc",
            timestamp="2026-01-01T00:00:00",
            description=long_desc,
        )
        md = render_registry_markdown(reg)
        # Should be truncated with ...
        assert "..." in md
        assert "A" * 100 not in md


class TestWriteRegistryMarkdown:
    """Test file writing."""

    def test_writes_file(self, tmp_path):
        reg = _populate_registry(tmp_path)
        output = tmp_path / "REGISTRY.md"
        write_registry_markdown(reg, output)
        assert output.exists()
        content = output.read_text()
        assert "RUN-001" in content

    def test_creates_parent_dirs(self, tmp_path):
        reg = _populate_registry(tmp_path)
        output = tmp_path / "sub" / "dir" / "REGISTRY.md"
        write_registry_markdown(reg, output)
        assert output.exists()
