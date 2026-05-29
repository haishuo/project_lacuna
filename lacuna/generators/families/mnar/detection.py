"""MNAR detection limit generators (lab assay-style).

Every generator here accepts an optional `target_col_idx` param (in addition to its documented
params). When set, the detection-limit rule is applied to EXACTLY that one column (negative
indices wrap from the end, matching `self_censoring.MNARLogistic`); when absent, behaviour is
unchanged — a random `affected_frac` fraction of columns. This lets diverse MNAR subtypes be
spliced into per-column mechanism mixtures (ADR-0006). See `_affected_cols.resolve_affected_cols`.
"""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian
from ._affected_cols import resolve_affected_cols


class MNARDetectionLower(Generator):
    """MNAR Detection Lower Limit - values below detection limit go missing.

    Models lab assays where instruments cannot measure below a detection limit.
    P(R_j=0 | X_j) = 1 if X_j < quantile(detection_percentile/100)

    Optional params:
        detection_percentile: Percentile for detection limit (default: 15)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        # Affected columns: random fraction (legacy) or a single targeted column when
        # `target_col_idx` is set (per-column mixtures; ADR-0006). Default behaviour unchanged.
        affected_cols = resolve_affected_cols(self.params, d, rng)

        det_pct = self.params.get("detection_percentile", 15)

        for col in affected_cols:
            vals = X[:, col]
            limit = torch.quantile(vals, det_pct / 100.0)
            R[:, col] = vals >= limit

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARDetectionUpper(Generator):
    """MNAR Detection Upper Limit - values above upper limit go missing.

    Models saturation limits where instruments max out.
    P(R_j=0 | X_j) = 1 if X_j > quantile(detection_percentile/100)

    Optional params:
        detection_percentile: Percentile for upper limit (default: 85)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        # Affected columns: random fraction (legacy) or a single targeted column when
        # `target_col_idx` is set (per-column mixtures; ADR-0006). Default behaviour unchanged.
        affected_cols = resolve_affected_cols(self.params, d, rng)

        det_pct = self.params.get("detection_percentile", 85)

        for col in affected_cols:
            vals = X[:, col]
            limit = torch.quantile(vals, det_pct / 100.0)
            R[:, col] = vals <= limit

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)


class MNARDetectionBoth(Generator):
    """MNAR Detection Both Limits - values outside both limits go missing.

    Models instruments with both lower and upper detection limits.

    Optional params:
        lower_percentile: Lower detection limit percentile (default: 10)
        upper_percentile: Upper detection limit percentile (default: 90)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        # Affected columns: random fraction (legacy) or a single targeted column when
        # `target_col_idx` is set (per-column mixtures; ADR-0006). Default behaviour unchanged.
        affected_cols = resolve_affected_cols(self.params, d, rng)

        lo_pct = self.params.get("lower_percentile", 10)
        hi_pct = self.params.get("upper_percentile", 90)

        for col in affected_cols:
            vals = X[:, col]
            lo = torch.quantile(vals, lo_pct / 100.0)
            hi = torch.quantile(vals, hi_pct / 100.0)
            R[:, col] = (vals >= lo) & (vals <= hi)

        if R.sum() == 0:
            R[0, 0] = True

        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(X, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        return self._compute_missingness(X, rng)
