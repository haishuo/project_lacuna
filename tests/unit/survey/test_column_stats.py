"""Tests for lacuna.survey.column_stats."""

import numpy as np
import pytest

from lacuna.data.catalog import create_default_catalog
from lacuna.data.ingestion import RawDataset
from lacuna.survey.column_stats import cardinality_distribution, target_cardinality_table


def _raw(name="syn"):
    g = np.random.default_rng(0)
    cont = g.standard_normal(500)            # ~500 unique (continuous)
    binary = (g.random(500) > 0.5).astype(float)  # 2 unique
    coded = g.integers(0, 5, 500).astype(float)    # 5 unique
    data = np.stack([cont, binary, coded], axis=1).astype("float32")
    return RawDataset(data=data, feature_names=("cont", "binary", "coded"), name=name)


# ---------- normal ----------

def test_cardinality_table_values():
    table = target_cardinality_table([_raw()])
    by_name = {r["target_name"]: r for r in table}
    assert by_name["binary"]["n_unique"] == 2
    assert by_name["coded"]["n_unique"] == 5
    assert by_name["cont"]["n_unique"] > 100
    assert 0.0 < by_name["cont"]["unique_frac"] <= 1.0


def test_distribution_ordering():
    table = target_cardinality_table([_raw()])
    dist = cardinality_distribution(table)
    assert dist["min"] <= dist["median"] <= dist["max"]
    assert dist["n_targets"] == len(table)


def test_real_pool_spans_range():
    cat = create_default_catalog()
    pool = [cat.load(n) for n in ("survey_cps1988", "survey_computers", "survey_hmda")]
    table = target_cardinality_table(pool)
    dist = cardinality_distribution(table)
    assert dist["min"] <= 8            # has low-cardinality coded items
    assert dist["max"] > 100           # has continuous targets


# ---------- failure ----------

def test_empty_distribution_raises():
    with pytest.raises(ValueError):
        cardinality_distribution([])
