"""
Tests for the pure aggregation logic in scripts/stage5_true_mixtures.py.

The train/eval/forward paths are integration code (need a GPU + the baseline checkpoint) and are
exercised by `--quick`; here we unit-test the deterministic mean+/-sd and per-subtype aggregation
that produces the reported headline numbers, since a bug there would silently corrupt the result.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "stage5_true_mixtures.py"
_spec = importlib.util.spec_from_file_location("stage5_true_mixtures", _SCRIPT)
stage5 = importlib.util.module_from_spec(_spec)
sys.modules["stage5_true_mixtures"] = stage5
_spec.loader.exec_module(stage5)


# ---------------------------------------------------------------------------
# _mean_sd
# ---------------------------------------------------------------------------

def test_mean_sd_single_value_has_zero_sd():
    assert stage5._mean_sd([0.5]) == [0.5, 0.0]


def test_mean_sd_multi():
    m, sd = stage5._mean_sd([0.0, 1.0])
    assert m == pytest.approx(0.5)
    assert sd == pytest.approx(0.5)  # population sd of {0,1}


def test_mean_sd_ignores_none():
    assert stage5._mean_sd([0.4, None, 0.6])[0] == pytest.approx(0.5)


def test_mean_sd_all_none_returns_none():
    assert stage5._mean_sd([None, None]) == [None, None]


def test_mean_sd_empty_returns_none():
    assert stage5._mean_sd([]) == [None, None]


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def _metric(acc, rec, ece=0.05, comp=0.4, miss=0.25, subs=None):
    return {
        "per_column_accuracy": acc,
        "per_column_ece": ece,
        "composition_l1_mean": comp,
        "per_class_recall": rec,
        "per_class_precision": {k: v for k, v in rec.items()},
        "realised_miss_rate_by_class": {k: miss for k in ("MCAR", "MAR", "MNAR")},
        "per_subtype_recall": subs or {},
    }


def test_aggregate_scalars_mean_sd_and_per_seed_lists():
    seeds = [
        _metric(0.50, {"MCAR": 0.5, "MAR": 0.7, "MNAR": 0.3}),
        _metric(0.60, {"MCAR": 0.7, "MAR": 0.7, "MNAR": 0.5}),
    ]
    agg = stage5.aggregate(seeds)
    assert agg["per_column_accuracy"] == [0.55, 0.05]
    assert agg["recall_MNAR"] == [0.4, 0.1]
    assert agg["recall_MAR"] == [0.7, 0.0]
    assert agg["recall_MNAR_per_seed"] == [0.3, 0.5]   # distribution preserved (rigor: report it)
    assert agg["missrate_MNAR"] == [0.25, 0.0]


def test_aggregate_per_subtype_filters_small_n_but_keeps_total():
    """A subtype's mean+/-sd averages only seeds with n>=20, but n_total sums all seeds."""
    seeds = [
        _metric(0.5, {"MCAR": 0.5, "MAR": 0.5, "MNAR": 0.5},
                subs={"MNAR:threshold_left": {"recall": 0.9, "n": 40}}),
        _metric(0.5, {"MCAR": 0.5, "MAR": 0.5, "MNAR": 0.5},
                subs={"MNAR:threshold_left": {"recall": 0.1, "n": 5}}),  # too few -> excluded from mean
    ]
    agg = stage5.aggregate(seeds)
    tl = agg["per_subtype_recall"]["MNAR:threshold_left"]
    assert tl["recall_mean"] == 0.9       # only the n=40 seed counts toward the mean
    assert tl["n_seeds"] == 1
    assert tl["n_total"] == 45            # but the total n includes both


def test_aggregate_handles_missing_subtype_in_some_seeds():
    seeds = [
        _metric(0.5, {"MCAR": 0.5, "MAR": 0.5, "MNAR": 0.5},
                subs={"MNAR:gaming": {"recall": 0.8, "n": 30}}),
        _metric(0.5, {"MCAR": 0.5, "MAR": 0.5, "MNAR": 0.5}, subs={}),  # gaming absent this seed
    ]
    agg = stage5.aggregate(seeds)
    assert agg["per_subtype_recall"]["MNAR:gaming"]["n_seeds"] == 1


def test_arms_define_one_variable_progression():
    """Sanity: the three arms add exactly one diversity axis at a time."""
    assert stage5.ARMS["monoculture"] == {"mnar_diverse": False, "mar_diverse": False}
    assert stage5.ARMS["diverse_mnar"] == {"mnar_diverse": True, "mar_diverse": False}
    assert stage5.ARMS["full_diversity"] == {"mnar_diverse": True, "mar_diverse": True}
