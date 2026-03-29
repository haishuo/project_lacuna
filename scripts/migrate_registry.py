#!/usr/bin/env python3
"""
CLI wrapper for one-time migration of existing run directories into
the run registry.

Usage:
    python scripts/migrate_registry.py
    python scripts/migrate_registry.py --runs-dir /path/to/runs --registry experiments/registry.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.experiments.registry import RunRegistry
from lacuna.experiments.registry_render import write_registry_markdown
from lacuna.experiments.migrate import migrate_existing_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate existing run directories into the registry",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="/mnt/artifacts/project_lacuna/runs",
        help="Root directory containing run subdirectories",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="experiments/registry.json",
        help="Path to registry JSON file (relative to project root)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    registry_path = PROJECT_ROOT / args.registry

    print(f"Runs directory : {runs_dir}")
    print(f"Registry file  : {registry_path}")

    # Load or create registry.
    registry = RunRegistry(registry_path)
    registry.load()
    before_count = len(registry)

    # Run migration.
    new_entries = migrate_existing_runs(runs_dir, registry)

    after_count = len(registry)
    print(f"\nExisting entries : {before_count}")
    print(f"Newly registered : {len(new_entries)}")
    print(f"Total entries    : {after_count}")

    # List new entries.
    if new_entries:
        print("\nNew entries:")
        for e in new_entries:
            print(f"  {e.run_id}  {e.folder_name}  [{e.status}]")

    # Generate REGISTRY.md alongside the JSON file.
    md_path = registry_path.parent / "REGISTRY.md"
    write_registry_markdown(registry, md_path)
    print(f"\nRegistry markdown: {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
