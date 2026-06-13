"""Tests for lacuna.analysis.realism_gate.gate (orchestration + verdict)."""

import numpy as np
import pytest
import torch

from lacuna.core.rng import RNGState
from lacuna.analysis.realism_gate.corpora import MaskCorpus
from lacuna.analysis.realism_gate.gate import (
    VERDICT_FAIL,
    VERDICT_PASS,
    format_gate_table,
    run_realism_gate,
)


class _FakeMCAR:
    """Minimal generator stub: drops each cell independently at a fixed rate."""

    def __init__(self, rate, generator_id=0, name="fake"):
        self.rate = rate
        self.generator_id = generator_id
        self.name = name

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        p = rng.rand(*X.shape)
        return p >= self.rate  # True = observed


class _BadGen:
    generator_id = 9
    name = "bad"

    def apply_to(self, X, rng):
        return torch.zeros(X.shape[0], X.shape[1] + 1, dtype=torch.bool)  # wrong shape


def _corpus(real_rate, d=10, seed=0):
    rng = np.random.default_rng(seed)
    M_real = (rng.random((3000, d)) < real_rate).astype(np.uint8)
    X = rng.normal(size=(3000, d)).astype(np.float64)
    return MaskCorpus(
        "T", tuple(f"c{j}" for j in range(d)), M_real,
        rng.integers(0, 4, 3000), X, rng.integers(0, 4, 3000),
        ("0", "1", "2", "3"), True,
    )


def test_matched_generator_passes():
    corpus = _corpus(real_rate=0.10)
    res = run_realism_gate(_FakeMCAR(0.10), corpus, seed=2026)
    assert res.verdict == VERDICT_PASS
    assert res.c2st.auc < 0.60


def test_saturated_generator_fails():
    corpus = _corpus(real_rate=0.05)
    res = run_realism_gate(_FakeMCAR(0.60), corpus, seed=2026)
    assert res.verdict == VERDICT_FAIL
    assert res.c2st.auc > 0.60
    assert res.fidelity.overall_rate_gen > res.fidelity.overall_rate_real


def test_degenerate_output_raises():
    corpus = _corpus(real_rate=0.10)
    with pytest.raises(ValueError):
        run_realism_gate(_BadGen(), corpus, seed=1)


def test_determinism():
    corpus = _corpus(real_rate=0.10)
    a = run_realism_gate(_FakeMCAR(0.15), corpus, seed=42)
    b = run_realism_gate(_FakeMCAR(0.15), corpus, seed=42)
    assert a.c2st.auc == b.c2st.auc


def test_format_table_runs():
    corpus = _corpus(real_rate=0.10)
    results = [run_realism_gate(_FakeMCAR(r, generator_id=i, name=f"g{i}"), corpus, seed=1)
               for i, r in enumerate([0.10, 0.50])]
    table = format_gate_table(results)
    assert "Verdict" in table and "PASS" in table
    assert format_gate_table([]) == "(no results)"
