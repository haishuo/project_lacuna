"""Tests for lacuna.survey.leakage — the matched-rate leakage gate."""

import pytest

from lacuna.core.rng import RNGState
from lacuna.survey.answer_sheet import GENERATOR_FAMILY, AnswerSheet
from lacuna.survey.delta_bins import assign_delta_bin
from lacuna.survey.leakage import assess_leakage, leakage_pass


def _sheet(delta, realized_rate, target_rate=0.3, i=0):
    return AnswerSheet(
        source_name="syn", n=256, d=5, target_col_idx=0, target_col_name="t",
        predictor_col_idx=1, predictor_col_name="p", beta0=-1.0, beta1=1.0,
        delta=delta, delta_bin=assign_delta_bin(delta), generator_family=GENERATOR_FAMILY,
        target_rate=target_rate, realized_rate=realized_rate,
        corr_target_predictor=0.3, seed=i,
    )


def _clean_corpus(rng):
    """Matched-rate: realized ≈ target with small symmetric jitter, independent of δ."""
    deltas = [0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5]
    sheets = []
    for i in range(140):
        d = deltas[i % len(deltas)]
        jitter = 0.01 * (1 if i % 2 else -1)  # zero-mean, δ-independent
        sheets.append(_sheet(d, 0.3 + jitter, i=i))
    return sheets


def _leaky_corpus():
    """Realized rate rises with δ — a clear rate cue."""
    deltas = [0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5]
    sheets = []
    for i in range(140):
        d = deltas[i % len(deltas)]
        rate = 0.15 + 0.1 * d  # monotonic in δ
        sheets.append(_sheet(d, rate, i=i))
    return sheets


# ---------- normal ----------

def test_clean_corpus_passes():
    report = assess_leakage(_clean_corpus(RNGState(seed=1)), RNGState(seed=2))
    assert abs(report.delta_rate_pearson) < 0.2
    assert leakage_pass(report)


def test_leaky_corpus_fails():
    report = assess_leakage(_leaky_corpus(), RNGState(seed=2))
    assert report.delta_rate_pearson > 0.5
    assert not leakage_pass(report)


def test_report_has_per_bin_table():
    report = assess_leakage(_clean_corpus(RNGState(seed=1)), RNGState(seed=2))
    bins = {row["bin"] for row in report.per_bin}
    assert bins == set(range(7))
    assert report.to_dict()["n_examples"] == 140


# ---------- failure ----------

def test_too_few_examples_raises():
    with pytest.raises(ValueError):
        assess_leakage([_sheet(0.0, 0.3)], RNGState(seed=1))
