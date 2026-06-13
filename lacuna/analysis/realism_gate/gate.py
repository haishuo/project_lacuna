"""
lacuna.analysis.realism_gate.gate

Orchestrate the L1 realism gate for one (generator, corpus) pair.

One job: inject a generator's missingness into the corpus's complete-case real-X
substrate (``generator.apply_to``), then score the resulting masks against the
real masks with the C2ST (the gate) and the fidelity + authenticity panel
(context), and render a verdict.

Verdict policy (charter §6.1 — "discriminator near chance"): the practical gate
is the C2ST effect size, because with tens of thousands of rows any real
difference is statistically significant (the p-value is reported, not gated).
``PASS`` (realistic) iff pooled OOF AUC < ``auc_pass``; otherwise ``FAIL``. A
generator failing is a *finding*, reported, never tuned away.

Fail loud (Coding Bible §1) on a degenerate generator output (wrong shape/dtype).
Deterministic given ``seed``.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from lacuna.core.rng import RNGState
from lacuna.generators.base import Generator

from .c2st import C2STResult, run_c2st
from .corpora import MaskCorpus
from .fidelity import FidelityResult, run_fidelity

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"


@dataclass(frozen=True)
class RealismGateResult:
    """The realism-gate verdict + evidence for one (generator, corpus) pair."""

    generator_id: int
    generator_name: str
    corpus_name: str
    d: int
    c2st: C2STResult
    fidelity: FidelityResult
    auc_pass: float
    verdict: str

    @property
    def auc(self) -> float:
        return self.c2st.auc


def _gen_mask(generator: Generator, corpus: MaskCorpus, seed: int) -> np.ndarray:
    """Apply a generator to the complete-X substrate; return a [n_cc, d] uint8 mask.

    Raises:
        ValueError: if ``apply_to`` returns the wrong shape/dtype (fail loud).
    """
    X = torch.from_numpy(corpus.X_complete.astype(np.float32))
    R = generator.apply_to(X, RNGState(seed=seed))
    if not torch.is_tensor(R):
        raise ValueError(f"{generator.name}.apply_to must return a tensor, got {type(R)}")
    if R.dtype != torch.bool:
        raise ValueError(f"{generator.name}.apply_to must return bool R, got {R.dtype}")
    if tuple(R.shape) != tuple(X.shape):
        raise ValueError(
            f"{generator.name}.apply_to returned shape {tuple(R.shape)}, "
            f"expected {tuple(X.shape)}"
        )
    return (~R).cpu().numpy().astype(np.uint8)  # mask: 1 = missing


def run_realism_gate(
    generator: Generator,
    corpus: MaskCorpus,
    *,
    seed: int = 2026,
    auc_pass: float = 0.60,
    n_splits: int = 5,
) -> RealismGateResult:
    """Run the full realism gate for one generator against one corpus.

    Args:
        generator: a Lacuna generator (must implement ``apply_to``).
        corpus: a ``MaskCorpus`` (real masks + complete-X substrate).
        seed: deterministic seed for generation, balancing, and authenticity.
        auc_pass: C2ST AUC below which the generator is judged realistic.
        n_splits: C2ST cross-validation folds.

    Raises:
        ValueError: on degenerate generator output (propagated from ``_gen_mask``)
            or C2ST/fidelity contract violations.
    """
    M_gen = _gen_mask(generator, corpus, seed)
    c2st = run_c2st(
        corpus.M_real, M_gen,
        corpus.real_block_ids, corpus.cc_block_ids, corpus.columns,
        seed=seed, n_splits=n_splits,
    )
    fidelity = run_fidelity(corpus.M_real, M_gen, seed=seed)
    verdict = VERDICT_PASS if c2st.auc < auc_pass else VERDICT_FAIL
    return RealismGateResult(
        generator_id=generator.generator_id,
        generator_name=generator.name,
        corpus_name=corpus.name,
        d=corpus.d,
        c2st=c2st,
        fidelity=fidelity,
        auc_pass=auc_pass,
        verdict=verdict,
    )


def format_gate_table(results: List[RealismGateResult]) -> str:
    """Render gate results as a fixed-width table, sorted by AUC ascending."""
    if not results:
        return "(no results)"
    rows = sorted(results, key=lambda r: r.c2st.auc)
    header = (
        f"{'ID':>3}  {'Generator':<34}  {'Corpus':<7}  "
        f"{'rReal':>6}  {'rGen':>6}  {'AUC':>6}  {'p':>8}  "
        f"{'attrGap':>7}  {'biGap':>6}  {'Verdict':<7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        f = r.fidelity
        lines.append(
            f"{r.generator_id:>3}  {r.generator_name[:34]:<34}  {r.corpus_name[:7]:<7}  "
            f"{f.overall_rate_real:>6.3f}  {f.overall_rate_gen:>6.3f}  "
            f"{r.c2st.auc:>6.3f}  {r.c2st.p_value:>8.1e}  "
            f"{f.attr_max_gap:>7.3f}  {f.bivar_max_gap:>6.3f}  {r.verdict:<7}"
        )
    n_pass = sum(1 for r in results if r.verdict == VERDICT_PASS)
    lines.append("-" * len(header))
    lines.append(f"PASS {n_pass} / {len(results)}  (realistic = AUC < {results[0].auc_pass})")
    return "\n".join(lines)
