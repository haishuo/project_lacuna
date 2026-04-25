"""MAR generators tuned for real-world missingness patterns.

The pre-existing MAR family covers the canonical MAR fingerprints
(single-target single-predictor sigmoid; multi-column block dropouts;
survey skip-logic). What it does NOT cover well is the *moderate
multi-column* regime that real datasets like NHANES, mass surveys, and
clinical lab panels actually live in:

  - airquality   (R datasets) :  1–2 columns missing at ~5–25 %
                                 driven by observed weather variables.
  - mass_survey  (MASS::survey):  6 columns missing at ~5 % each,
                                 driven by demographic predictors.
  - pbc          (survival::pbc): 12 columns missing at ~18 % each,
                                 driven by clinical state.

The 2026-04-25 MAR-detection diagnostic
(`scripts/diagnose_mar.py`) found these textbook-MAR cases routing
to MNAR with high model confidence — a distribution-shift symptom
rather than a fundamental identifiability issue.

The three generators below fill that gap by producing multi-column
moderate-rate missingness with patterns the existing MNAR family
does NOT mimic:

  MARRealisticSingle   single-column MAR with z-scored predictor
                       so the realised target missing-rate stays in
                       the realistic 15–35 % range, not the saturated
                       100 % the un-normalised generators produce.

  MARPartialResponse   form-completion drop-off — a subset of rows
                       (gated by an observed predictor) have a block
                       of trailing columns simultaneously missing,
                       like surveys where some respondents don't
                       finish the second half.

  MARDemographicGated  administrative cutoff — rows past a threshold
                       on an observed predictor have a sparse pattern
                       of missingness across many columns. Models the
                       NHANES / lab-panel pattern: clinical state
                       drives whether each test is ordered.

All three call `_zscore_predictor` so their behaviour is invariant to
the absolute scale of the catalog dataset — without this, alpha-style
linear predictors saturate immediately on real X (e.g. wine alcohol on
the 11–15 scale would push every logit to ±20).
"""

from __future__ import annotations

from typing import Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


def _zscore(x: torch.Tensor) -> torch.Tensor:
    """Standardise a 1-D tensor; returns zeros if std == 0."""
    mean = x.mean()
    std = x.std(unbiased=False)
    if std.item() == 0.0:
        return torch.zeros_like(x)
    return (x - mean) / std


def _resolve_idx(idx: int, d: int) -> int:
    """Wrap a possibly-negative column index into [0, d)."""
    if idx < 0:
        idx = d + idx
    return idx % d


# =============================================================================
# 1. MARRealisticSingle — single-target MAR at realistic rates
# =============================================================================


class MARRealisticSingle(Generator):
    """Single-column MAR with z-scored predictor.

    Same shape as MARLogistic, but the predictor is z-scored before the
    linear combination so the realised missing rate on real catalog
    datasets stays in the 15-35% range rather than saturating to ~100%
    of the target column.

    Required params:
        target_miss_rate:  Desired marginal P(missing) for the target
                           column. The intercept is computed so the
                           realised rate matches this on a standard
                           normal predictor.
        slope:             Slope on the z-scored predictor.
                           Larger ⇒ more pronounced MAR signal.

    Optional params:
        target_col_idx:    Column to drop (default: -1).
        predictor_col_idx: Column whose z-score drives missingness
                           (default: 0).
        base_mean / base_std: Synthetic-X parameters for `sample`.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ("target_miss_rate", "slope"):
            if key not in params:
                raise ValueError(f"MARRealisticSingle requires '{key}' parameter")
        rate = params["target_miss_rate"]
        if not 0.01 <= rate <= 0.95:
            raise ValueError(f"target_miss_rate must be in [0.01, 0.95]; got {rate}")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARRealisticSingle requires d >= 2")
        target = _resolve_idx(self.params.get("target_col_idx", -1), d)
        predictor = _resolve_idx(self.params.get("predictor_col_idx", 0), d)
        if predictor == target:
            predictor = (target + 1) % d

        z = _zscore(X[:, predictor])
        slope = float(self.params["slope"])
        rate = float(self.params["target_miss_rate"])
        # logit(rate) hits the target marginal exactly when the
        # z-scored predictor's contribution averages to zero; the
        # slope adds the MAR signal on top.
        intercept = float(torch.special.logit(torch.tensor(rate)).item())
        p_missing = torch.sigmoid(intercept + slope * z)

        R = torch.ones(n, d, dtype=torch.bool)
        R[:, target] = ~(rng.rand(n) < p_missing)
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


# =============================================================================
# 2. MARPartialResponse — survey form drop-off
# =============================================================================


class MARPartialResponse(Generator):
    """Form-completion drop-off MAR.

    Each row is independently flagged as a "partial responder" with
    probability that depends on a z-scored observed predictor. Partial
    responders have a contiguous trailing block of columns marked
    missing; full responders have no missingness from this generator.

    This produces patterns like mass-survey item nonresponse where some
    respondents skip the back half of the form: many columns have
    moderate per-cell missing rates that are highly correlated across
    those columns within each row.

    Required params:
        partial_rate:     Marginal P(row is a partial responder).
        slope:            Slope on the z-scored predictor for partial
                          probability. Larger ⇒ stronger MAR signal.
        block_frac:       Fraction of columns at the end of the row
                          that get dropped for partial responders.

    Optional params:
        predictor_col_idx: Predictor column (default: 0).
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ("partial_rate", "slope", "block_frac"):
            if key not in params:
                raise ValueError(f"MARPartialResponse requires '{key}' parameter")
        if not 0.05 <= params["partial_rate"] <= 0.80:
            raise ValueError("partial_rate must be in [0.05, 0.80]")
        if not 0.10 <= params["block_frac"] <= 0.80:
            raise ValueError("block_frac must be in [0.10, 0.80]")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 3:
            raise ValueError("MARPartialResponse requires d >= 3")
        predictor = _resolve_idx(self.params.get("predictor_col_idx", 0), d)

        z = _zscore(X[:, predictor])
        rate = float(self.params["partial_rate"])
        slope = float(self.params["slope"])
        intercept = float(torch.special.logit(torch.tensor(rate)).item())
        p_partial = torch.sigmoid(intercept + slope * z)
        is_partial = rng.rand(n) < p_partial

        block = max(1, int(round(d * float(self.params["block_frac"]))))
        # Drop the last `block` columns (excluding predictor) for partial responders.
        drop_cols = [j for j in range(d - block, d) if j != predictor]
        if not drop_cols:
            drop_cols = [(predictor + 1) % d]

        R = torch.ones(n, d, dtype=torch.bool)
        for j in drop_cols:
            R[is_partial, j] = False
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


# =============================================================================
# 3. MARDemographicGated — admin / lab-panel pattern
# =============================================================================


class MARDemographicGated(Generator):
    """Administrative MAR — clinical-state-style sparse multi-col dropout.

    For each non-predictor column, an independent Bernoulli draw decides
    whether that cell is dropped for rows where the observed predictor
    crosses a threshold. The result is moderate per-column missing
    rates spread across many columns, with cross-column missingness
    correlation that is positive but not block-uniform — closer to
    pbc / NHANES patterns than to MAR-MultiCol's tight blocks.

    Required params:
        gate_quantile:     Predictor quantile that defines the gated
                           subgroup (e.g. 0.5 = top half).
        rate_above:        Per-cell P(missing) for cells in the gated
                           subgroup, drawn independently across columns.
        rate_below:        Per-cell P(missing) for the other subgroup.

    Optional params:
        predictor_col_idx: Predictor column (default: 0).
        cols_affected_frac: Fraction of non-predictor cols subject to
                           the differential rate (default: 0.7).
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ("gate_quantile", "rate_above", "rate_below"):
            if key not in params:
                raise ValueError(f"MARDemographicGated requires '{key}' parameter")
        for key in ("rate_above", "rate_below"):
            if not 0.0 <= params[key] <= 0.95:
                raise ValueError(f"{key} must be in [0, 0.95]")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 3:
            raise ValueError("MARDemographicGated requires d >= 3")
        predictor = _resolve_idx(self.params.get("predictor_col_idx", 0), d)

        q = float(self.params["gate_quantile"])
        threshold = torch.quantile(X[:, predictor], q)
        in_gate = X[:, predictor] >= threshold

        cols_frac = float(self.params.get("cols_affected_frac", 0.7))
        non_pred = [j for j in range(d) if j != predictor]
        n_affected = max(1, int(round(len(non_pred) * cols_frac)))
        # Take the first n_affected non-predictor cols (deterministic).
        affected = non_pred[:n_affected]

        rate_a = float(self.params["rate_above"])
        rate_b = float(self.params["rate_below"])

        R = torch.ones(n, d, dtype=torch.bool)
        for j in affected:
            p = torch.where(in_gate, torch.tensor(rate_a), torch.tensor(rate_b))
            R[:, j] = ~(rng.rand(n) < p)
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
