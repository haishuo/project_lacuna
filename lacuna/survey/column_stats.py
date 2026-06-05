"""
lacuna.survey.column_stats

Target-column cardinality characterization for the P2.2b cardinality probe.

ONE job: report, per candidate target column, how many distinct values it takes (a proxy for
continuity vs low-cardinality coded items) — so real-X examples can be stratified by cardinality
and we can ask whether high-cardinality/continuous targets recover δ signal while low-cardinality
ones floor. Mirrors `proxy_score.target_r2_table`.

Deterministic (Rule 6): pure function of the data. Fail loud (Rule 1) on an empty table.
"""

from typing import List

import numpy as np

from lacuna.data.ingestion import RawDataset


def target_cardinality_table(raw_datasets: List[RawDataset]) -> List[dict]:
    """Per non-constant candidate target: n_unique and unique_frac on the full data.

    Each row: {dataset, target_idx, target_name, n_unique, unique_frac, n, d}.
    """
    table: List[dict] = []
    for raw in raw_datasets:
        X = raw.data
        n, d = X.shape
        if d < 2:
            continue
        nonconst = [c for c in range(d) if float(np.std(X[:, c])) > 0.0]
        if len(nonconst) < 2:
            continue
        for t in nonconst:
            u = int(np.unique(X[:, t]).size)
            table.append({
                "dataset": raw.name, "target_idx": int(t),
                "target_name": str(raw.feature_names[t]),
                "n_unique": u, "unique_frac": float(u / n), "n": int(n), "d": int(d),
            })
    return table


def cardinality_distribution(table: List[dict]) -> dict:
    """min / q25 / median / q75 / max of n_unique across the table (safeguard report)."""
    if len(table) == 0:
        raise ValueError("empty cardinality table")
    vals = np.array([row["n_unique"] for row in table], dtype=np.float64)
    q = np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n_targets": len(table),
        "min": int(q[0]), "q25": float(q[1]), "median": float(q[2]),
        "q75": float(q[3]), "max": int(q[4]),
    }
