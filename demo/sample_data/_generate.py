"""Generate three demo CSVs — one per missingness mechanism — using
Lacuna's actual generator registry applied to a real catalog dataset.

This is the right way: the model was trained on patterns produced by
these generators on real X, so the demo samples come from the same
distribution. Hand-rolled sigmoid drops can drift from that distribution
in subtle ways and end up misclassified.

Run from the repo root:
    python demo/sample_data/_generate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lacuna.core.rng import RNGState
from lacuna.data.catalog import create_default_catalog
from lacuna.data.semisynthetic import apply_missingness
from lacuna.generators.families.registry_builder import load_registry_from_config

OUT_DIR = Path(__file__).parent

# Pick generators that produce visually clear missingness for the demo.
GENERATORS = {
    "mcar.csv": "MCAR-Bernoulli-30",
    "mar.csv":  "MAR-Logistic",
    "mnar.csv": "MNAR-SelfCensor-High",
}

# Use a real catalog dataset as X. wine has 178 rows × 13 features —
# enough columns for cross-column correlation to matter, small enough
# that the saved CSV is easy to inspect.
DATASET_NAME = "wine"

SEED = 20260425


def to_csv(semi, path: Path) -> None:
    """Write a SemiSyntheticDataset to CSV using the COMPLETE values for
    observed cells and NaN for missing cells. We use `complete` (not
    `observed.x`) because ObservedDataset zeros out the missing entries
    in `x`, and zeroing in observed cells would give the model a
    deceptive view through the demo's CSV roundtrip."""
    x = semi.complete.numpy()
    r = semi.observed.r.numpy()
    values = np.where(r, x, np.nan)
    df = pd.DataFrame(values, columns=list(semi.observed.feature_names))
    df.to_csv(path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry_from_config("lacuna_tabular_110")
    by_name = {g.name: g for g in registry}

    catalog = create_default_catalog()
    raw = catalog.load(DATASET_NAME)

    for filename, gen_name in GENERATORS.items():
        if gen_name not in by_name:
            raise SystemExit(f"Generator {gen_name!r} not found in registry.")
        generator = by_name[gen_name]

        rng = RNGState(seed=SEED + generator.generator_id)
        semi = apply_missingness(raw, generator, rng)

        path = OUT_DIR / filename
        to_csv(semi, path)
        n_missing = int((~semi.observed.r).sum().item())
        n_total = semi.observed.n * semi.observed.d
        print(f"{path.name:10s} ({gen_name})  "
              f"{semi.observed.n}×{semi.observed.d}  "
              f"{n_missing}/{n_total} missing ({n_missing/n_total*100:.1f}%)")


if __name__ == "__main__":
    main()
