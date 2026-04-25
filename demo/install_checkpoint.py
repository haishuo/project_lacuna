"""Copy a trained checkpoint + metadata into the demo bundle.

Usage:
    python demo/install_checkpoint.py /mnt/artifacts/project_lacuna/runs/lacuna_demo_v1

Locates `checkpoints/best_model.pt` (preferred) or `final.pt` inside the
run directory, copies it to `demo/model.pt`, and writes a sidecar
`demo/model.json` with the train date, config name, and validation
accuracy pulled from `eval_report.json` if present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

DEMO_DIR = Path(__file__).parent
TARGET_PT = DEMO_DIR / "model.pt"
TARGET_META = DEMO_DIR / "model.json"


def find_checkpoint(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoints" / "best_model.pt",
        run_dir / "checkpoints" / "final.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No checkpoint found under {run_dir / 'checkpoints'}. "
        f"Expected best_model.pt or final.pt."
    )


def collect_metadata(run_dir: Path, ckpt_path: Path) -> dict:
    meta: dict = {
        "checkpoint_source": str(ckpt_path),
        "installed_at": datetime.now().isoformat(timespec="seconds"),
    }
    report_path = run_dir / "eval_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            meta["val_accuracy"] = float(report.get("summary", {}).get("accuracy", 0)) or None
            meta["config"] = report.get("config_path") or report.get("config")
            meta["trained_at"] = report.get("trained_at") or report.get("timestamp")
        except Exception as e:
            print(f"  (skipped eval_report.json: {e})", file=sys.stderr)

    if "trained_at" not in meta or not meta["trained_at"]:
        # Fall back to the checkpoint's mtime.
        ts = datetime.fromtimestamp(ckpt_path.stat().st_mtime)
        meta["trained_at"] = ts.isoformat(timespec="seconds")

    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path,
                    help="Path to the run directory under /mnt/artifacts/project_lacuna/runs/.")
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.run_dir}")

    ckpt_path = find_checkpoint(args.run_dir)
    print(f"Found checkpoint: {ckpt_path}")

    shutil.copy2(ckpt_path, TARGET_PT)
    print(f"Copied → {TARGET_PT}")

    meta = collect_metadata(args.run_dir, ckpt_path)
    TARGET_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote sidecar → {TARGET_META}")
    if meta.get("val_accuracy") is not None:
        print(f"  val_accuracy: {meta['val_accuracy']*100:.1f}%")
    print(f"  trained_at:   {meta['trained_at']}")


if __name__ == "__main__":
    main()
