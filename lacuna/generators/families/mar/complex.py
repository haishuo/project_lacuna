"""MAR generators with complex nonlinear dependency patterns."""

from typing import List, Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MARPolynomialMulti(Generator):
    """MAR mechanism with multi-predictor polynomial.

    Includes terms up to given degree for each predictor column.
    logit = alpha0 + sum_k sum_p alphas[k][p] * X_k^(p+1)

    Required params: degree, alpha0, alphas (list of lists: [predictor][degree])
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["degree", "alpha0", "alphas"]:
            if key not in params:
                raise ValueError(f"MARPolynomialMulti requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        alphas = self.params["alphas"]
        n_predictors = len(alphas)
        if d < n_predictors + 1:
            raise ValueError(
                f"MARPolynomialMulti with {n_predictors} predictors requires d >= {n_predictors + 1}"
            )

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:n_predictors]
        degree = self.params["degree"]

        logits = torch.full((n,), self.params["alpha0"])
        for k in range(n_predictors):
            xp = X[:, preds[k]]
            coeffs = alphas[k]
            for p in range(min(degree, len(coeffs))):
                logits = logits + coeffs[p] * xp ** (p + 1)

        p_missing = torch.sigmoid(logits)

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


class MARSplineMulti(Generator):
    """MAR mechanism with spline basis for each predictor.

    Each predictor gets a piecewise linear spline, and the total logit
    is the sum of all spline outputs.

    Required params: knots (list of knot positions, shared), alphas (list of slope
                     lists, one per predictor; each has len(knots)-1 slopes)
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["knots", "alphas"]:
            if key not in params:
                raise ValueError(f"MARSplineMulti requires '{key}' parameter")

    def _spline_eval(self, x: torch.Tensor, knots: list, slopes: list) -> torch.Tensor:
        """Evaluate piecewise linear spline."""
        n = x.shape[0]
        f = torch.zeros(n)
        for i in range(len(slopes)):
            lo, hi = knots[i], knots[i + 1]
            mask = (x >= lo) & (x < hi)
            f = f + mask.float() * slopes[i] * (x - lo)
        # Boundary extension
        below = x < knots[0]
        f = f + below.float() * slopes[0] * (x - knots[0])
        above = x >= knots[-1]
        f = f + above.float() * slopes[-1] * (x - knots[-1])
        return f

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        alphas = self.params["alphas"]
        knots = self.params["knots"]
        n_predictors = len(alphas)
        if d < n_predictors + 1:
            raise ValueError(
                f"MARSplineMulti with {n_predictors} predictors requires d >= {n_predictors + 1}"
            )

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:n_predictors]

        logits = torch.zeros(n)
        for k in range(n_predictors):
            logits = logits + self._spline_eval(X[:, preds[k]], knots, alphas[k])

        p_missing = torch.sigmoid(logits)

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


class MARTreeRules(Generator):
    """MAR mechanism via tree-like rules.

    Each rule specifies {col, threshold, rate}: if X[col] > threshold,
    the target column gets missingness with the given rate.
    Rules are applied independently and combined via max probability.

    Required params: rules (list of dicts with keys: col, threshold, rate)
    Optional params: target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        if "rules" not in params:
            raise ValueError("MARTreeRules requires 'rules' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARTreeRules requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        rules = self.params["rules"]
        p_missing = torch.zeros(n)

        for rule in rules:
            col = rule["col"] % d
            if col == target:
                col = (col + 1) % d
            threshold = rule["threshold"]
            rate = rule["rate"]
            active = X[:, col] > threshold
            # Combine via max: take the highest miss probability from any rule
            p_missing = torch.max(p_missing, active.float() * rate)

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


class MARKernel(Generator):
    """MAR mechanism via RBF kernel on predictor.

    P(miss) = sigmoid(scale * exp(-||X_pred - reference||^2 / bandwidth^2))

    Reference is the value at the specified quantile of X_pred.

    Required params: reference_quantile, bandwidth, scale
    Optional params: target_col_idx, predictor_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["reference_quantile", "bandwidth", "scale"]:
            if key not in params:
                raise ValueError(f"MARKernel requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        if d < 2:
            raise ValueError("MARKernel requires d >= 2")

        target = self.params.get("target_col_idx", -1)
        predictor = self.params.get("predictor_col_idx", 0)
        if target < 0:
            target = d + target
        if predictor < 0:
            predictor = d + predictor
        target = target % d
        predictor = predictor % d
        if predictor == target:
            predictor = (target + 1) % d

        xp = X[:, predictor]
        ref_q = self.params["reference_quantile"]
        reference = torch.quantile(xp, ref_q)
        bandwidth = self.params["bandwidth"]
        scale = self.params["scale"]

        dist_sq = (xp - reference) ** 2
        kernel_val = torch.exp(-dist_sq / (bandwidth ** 2))
        logits = scale * kernel_val
        p_missing = torch.sigmoid(logits)

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


class MARDistance(Generator):
    """MAR mechanism based on distance from random centers.

    Generates n_clusters random centers in predictor space. Miss probability
    depends on distance to nearest center.

    P(miss) = sigmoid(scale * min_k ||X_preds - center_k||)

    Required params: n_clusters, scale
    Optional params: n_predictor_cols (default 2), target_col_idx, base_mean, base_std
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MAR, params)
        for key in ["n_clusters", "scale"]:
            if key not in params:
                raise ValueError(f"MARDistance requires '{key}' parameter")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        n_pred_cols = self.params.get("n_predictor_cols", 2)
        if d < n_pred_cols + 1:
            raise ValueError(
                f"MARDistance requires d >= {n_pred_cols + 1}"
            )

        target = self.params.get("target_col_idx", -1)
        if target < 0:
            target = d + target
        target = target % d

        preds = [j for j in range(d) if j != target][:n_pred_cols]
        X_preds = X[:, preds]  # [n, n_pred_cols]

        n_clusters = self.params["n_clusters"]
        scale = self.params["scale"]

        # Generate random centers from the data range
        centers = rng.randn(n_clusters, n_pred_cols)  # [n_clusters, n_pred_cols]

        # Compute distance to nearest center
        # X_preds: [n, p], centers: [k, p]
        diffs = X_preds.unsqueeze(1) - centers.unsqueeze(0)  # [n, k, p]
        dists = torch.sqrt((diffs ** 2).sum(dim=2) + 1e-8)  # [n, k]
        min_dist = dists.min(dim=1).values  # [n]

        logits = scale * min_dist
        p_missing = torch.sigmoid(logits)

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
