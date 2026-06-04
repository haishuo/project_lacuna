"""
lacuna.survey.example_source

Pluggable example sources for the δ-prior training loop (P2.2b ladder; audit §8).

ONE job: produce `batching.DeltaExample`s (real-X or synthetic-X) behind a single interface so the
training/eval loop in `train.py` is data-source-agnostic and every ladder rung shares the SAME
head / loss / metrics / leakage / manifest harness — only the data source changes between rungs
(that is what makes a rung-to-rung difference attributable).

Two sources for rung 1:
  - SurveyExampleSource     — the P2.2 path: subsample a real survey dataset + own-value
    self-censoring (preserves existing behavior).
  - SyntheticTwoColSource   — rung 1: the P1-aligned controlled setting. 2-column standard
    bivariate-normal X (ConditionalGaussian(rho)) + own-value self-censoring on the KNOWN target
    column (col 1), matched rate. Removes real-X geometry, table width, and target localization.

Determinism (Rule 6): both sources draw only from the injected RNGState. Fail loud (Rule 1) on
empty pools / invalid grids.
"""

from abc import ABC, abstractmethod
from typing import List

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset
from lacuna.data.ingestion import RawDataset
from lacuna.feasibility.delta_generator import apply_self_censor
from lacuna.feasibility.xmodel import ConditionalGaussian

from .answer_sheet import GENERATOR_FAMILY, AnswerSheet
from .batching import DeltaExample, make_example
from .delta_bins import NUM_BINS, assign_delta_bin


class ExampleSource(ABC):
    """Produces one δ-self-censoring example for a given (delta, beta1). num_bins fixes the head width."""

    num_bins: int = NUM_BINS

    @abstractmethod
    def make_one(self, cfg, rng: RNGState, *, delta: float, beta1: float) -> DeltaExample:
        """Return one DeltaExample. `cfg` supplies target_rate / max_rows (TrainConfig)."""

    @abstractmethod
    def describe(self) -> dict:
        """JSON-serializable description for the manifest split_scheme."""


class SurveyExampleSource(ExampleSource):
    """Real survey X + own-value self-censoring (the P2.2 path). num_bins = 7."""

    def __init__(self, pool: List[RawDataset], num_bins: int = NUM_BINS):
        if len(pool) == 0:
            raise ValueError("SurveyExampleSource requires a non-empty dataset pool")
        self.pool = pool
        self.num_bins = num_bins

    def make_one(self, cfg, rng: RNGState, *, delta: float, beta1: float) -> DeltaExample:
        raw = self.pool[rng.randint(0, len(self.pool), (1,)).item()]
        return make_example(
            raw, beta1=beta1, delta=delta, target_rate=cfg.target_rate,
            rng=rng.spawn(), max_rows=cfg.max_rows,
        )

    def describe(self) -> dict:
        return {"x_source": "real_survey", "datasets": [r.name for r in self.pool]}


class SyntheticTwoColSource(ExampleSource):
    """Rung 1: 2-column standard bivariate-normal X (corr rho) + own-value self-censoring.

    The censored column (target) is col 1 and the predictor is col 0 — both KNOWN by construction,
    so the head faces ordered-δ estimation with geometry/width/localization all removed. n rows per
    example = cfg.max_rows (not subsampled; rung 1 fixes an adequate, non-starved n).
    """

    def __init__(self, rho_grid: List[float], num_bins: int = NUM_BINS):
        if len(rho_grid) == 0:
            raise ValueError("SyntheticTwoColSource requires a non-empty rho_grid")
        for rho in rho_grid:
            if not (-1.0 < float(rho) < 1.0):
                raise ValueError(f"rho values must be in (-1, 1), got {rho}")
        self.rho_grid = [float(r) for r in rho_grid]
        self.num_bins = num_bins

    def make_one(self, cfg, rng: RNGState, *, delta: float, beta1: float) -> DeltaExample:
        n = cfg.max_rows
        rho = self.rho_grid[rng.randint(0, len(self.rho_grid), (1,)).item()]
        xm = ConditionalGaussian.synthetic(rho)
        z_p = xm.sample_predictor(n, rng.spawn())
        z_t = xm.sample_conditional(z_p, rng.spawn())
        X = torch.stack([z_p, z_t], dim=1).float()  # [n, 2]; col 0 = predictor, col 1 = target

        res = apply_self_censor(
            X_complete=X, target_idx=1, predictor_idx=0,
            beta1=beta1, delta=delta, target_rate=cfg.target_rate, rng=rng.spawn(),
        )
        x_observed = X * res.mask.float()
        name = f"synth2col_rho{rho:+.2f}"
        sheet = AnswerSheet(
            source_name=name, n=int(n), d=2,
            target_col_idx=1, target_col_name="target",
            predictor_col_idx=0, predictor_col_name="predictor",
            beta0=float(res.params.beta0), beta1=float(beta1), delta=float(delta),
            delta_bin=assign_delta_bin(delta), generator_family=GENERATOR_FAMILY,
            target_rate=float(cfg.target_rate), realized_rate=float(res.realized_rate),
            corr_target_predictor=float(rho), seed=int(rng.seed),
        )
        observed = ObservedDataset(
            x=x_observed, r=res.mask, n=int(n), d=2,
            feature_names=("predictor", "target"), dataset_id=name,
            meta={"x_source": "synthetic_2col", "rho": rho},
        )
        return DeltaExample(observed=observed, answer_sheet=sheet)

    def describe(self) -> dict:
        return {"x_source": "synthetic_2col", "rho_grid": list(self.rho_grid)}
