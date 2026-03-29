"""
Markdown rendering for the run registry.

Single responsibility (Rule 3): turn a RunRegistry into a Markdown string
and optionally write it to disk.
"""

from datetime import datetime
from pathlib import Path

from lacuna.experiments.registry import RunRegistry


def render_registry_markdown(registry: RunRegistry) -> str:
    """
    Render the full registry as a Markdown document.

    Returns a string with an auto-generation warning, summary table,
    and per-run rows.
    """
    # NON-DETERMINISTIC: uses current wall-clock time for the generation timestamp.
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "<!-- AUTO-GENERATED from registry.json. Do not edit manually. -->",
        "",
        "# Lacuna Run Registry",
        "",
        f"Generated: {generated_at}  ",
        f"Total runs: {len(registry)}",
        "",
        "## Summary",
        "",
        "| Run ID | Date | Status | Accuracy | MAR Acc | ECE | Description |",
        "|--------|------|--------|----------|---------|-----|-------------|",
    ]

    for entry in registry.all_entries():
        # Extract date portion from ISO timestamp (first 10 chars).
        date_str = entry.timestamp[:10] if len(entry.timestamp) >= 10 else entry.timestamp

        accuracy = _fmt_pct(entry.metrics.get("accuracy"))
        mar_acc = _fmt_pct(entry.metrics.get("mar_acc"))
        ece = _fmt_float(entry.metrics.get("ece"))

        # Truncate long descriptions so the table stays readable.
        desc = entry.description
        if len(desc) > 60:
            desc = desc[:57] + "..."

        lines.append(
            f"| {entry.run_id} | {date_str} | {entry.status} "
            f"| {accuracy} | {mar_acc} | {ece} | {desc} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_registry_markdown(registry: RunRegistry, output_path: Path) -> None:
    """Render and write the registry Markdown to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_registry_markdown(registry)
    output_path.write_text(content, encoding="utf-8")


# -- helpers ----------------------------------------------------------------

def _fmt_pct(value) -> str:
    """Format a 0-1 float as a percentage string, or '-' if absent."""
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def _fmt_float(value) -> str:
    """Format a float to 4 decimal places, or '-' if absent."""
    if value is None:
        return "-"
    return f"{float(value):.4f}"
