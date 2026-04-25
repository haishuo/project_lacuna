#!/usr/bin/env bash
# Convenience wrapper: launches the Lacuna demo on http://localhost:8501.
# Run from the repo root or from inside demo/ — it locates itself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

cd "$REPO_ROOT"
exec streamlit run "$HERE/app.py" "$@"
