"""MNAR social desirability generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MNAR
from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from ..base_data import sample_gaussian


class MNARUnderReport(Generator):
    """MNAR Under-Reporting - high values go missing (income, spending).

    Models social desirability where high values are under-reported.
    Values above threshold_percentile go missing with under_report_prob.

    Optional params:
        threshold_percentile: Percentile above which under-reporting occurs (default: 75)
        under_report_prob: Probability of under-reporting (default: 0.6)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        threshold_pct = self.params.get("threshold_percentile", 75)
        under_report_prob = self.params.get("under_report_prob", 0.6)

        for col in affected_cols:
            vals = X[:, col]
            threshold = torch.quantile(vals, threshold_pct / 100.0)
            above = vals > threshold
            missing_mask = above & (rng.rand(n) < under_report_prob)
            R[:, col] = ~missing_mask

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


class MNAROverReport(Generator):
    """MNAR Over-Reporting - low values go missing (exercise, healthy eating).

    Models social desirability where low values are suppressed. People
    with low exercise amounts do not report.

    Optional params:
        threshold_percentile: Percentile below which over-reporting occurs (default: 25)
        over_report_prob: Probability of suppressing low values (default: 0.6)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        threshold_pct = self.params.get("threshold_percentile", 25)
        over_report_prob = self.params.get("over_report_prob", 0.6)

        for col in affected_cols:
            vals = X[:, col]
            threshold = torch.quantile(vals, threshold_pct / 100.0)
            below = vals < threshold
            missing_mask = below & (rng.rand(n) < over_report_prob)
            R[:, col] = ~missing_mask

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


class MNARNonLinearSocial(Generator):
    """MNAR Non-Linear Social Desirability - both extremes go missing.

    P(miss) = sigmoid(sensitivity * (X - center)^2 - 1)
    Values far from center_value are more likely missing.

    Optional params:
        center_value: Center around which values are acceptable (default: 0.0)
        sensitivity: How strongly extremes are censored (default: 1.0)
        affected_frac: Fraction of columns affected (default: 0.5)
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        R = torch.ones(n, d, dtype=torch.bool)

        affected_frac = self.params.get("affected_frac", 0.5)
        n_affected = max(1, int(d * affected_frac))
        affected_cols = rng.choice(d, size=n_affected, replace=False)

        center = self.params.get("center_value", 0.0)
        sensitivity = self.params.get("sensitivity", 1.0)

        for col in affected_cols:
            vals = X[:, col]
            deviation_sq = (vals - center) ** 2
            logits = sensitivity * deviation_sq - 1.0
            p_missing = torch.sigmoid(logits)

            missing_mask = rng.rand(n) < p_missing
            R[:, col] = ~missing_mask

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


class MNARModuleRefusal(Generator):
    """MNAR module-level refusal — entire question batteries skipped.

    Models survey item-batteries (PHQ-9 depression screener, drug-use
    modules, sexual-behavior modules, income/wealth supplements) where
    the missingness is row-aligned across the module: a respondent who
    refuses one item in the module typically refuses all of them. The
    refusal probability depends on the latent value of the module
    itself (e.g. severely depressed respondents are less likely to
    complete the depression screener), making this MNAR by design.

    Distinct from cell-level MNAR (UnderReport/OverReport/Threshold)
    in that the missingness is column-bimodal and row-aligned, which
    produces extreme cross-column correlation in the missingness mask.

    Required params:
        module_frac: Fraction of columns that form the refusable module.
            The remainder are "demographic" cols (always observed).
        baseline_refusal: Baseline probability of module refusal
            (controls overall NaN rate).
        selection_strength: Magnitude of the value→refusal coupling.
            0 = MCAR-like (random refusal); larger = stronger MNAR.
        direction: "high" (refuse when module values are high) or "low"
            (refuse when low). Defaults to "high" (e.g. depression).

    Optional params:
        demo_strength: Magnitude of an additional demographic-gate
            coupling. When > 0, refusal also depends on an observed
            non-module column. This mimics real-world module-refusal
            patterns (e.g. NHANES PHQ-9 module refusal correlates
            with both depressive symptoms AND age/income demographics).
            The mechanism remains MNAR as long as selection_strength > 0
            because there is residual selection on the unobserved values
            beyond what the demographic explains. Defaults to 0.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MNAR, params)
        if "module_frac" not in params or "baseline_refusal" not in params:
            raise ValueError("MNARModuleRefusal requires 'module_frac' and 'baseline_refusal'")
        if not (0.0 < params["module_frac"] < 1.0):
            raise ValueError("MNARModuleRefusal requires 0 < module_frac < 1")
        if not (0.0 <= params["baseline_refusal"] < 1.0):
            raise ValueError("MNARModuleRefusal requires 0 <= baseline_refusal < 1")

    def _compute_missingness(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        module_frac = float(self.params["module_frac"])
        baseline = float(self.params["baseline_refusal"])
        strength = float(self.params.get("selection_strength", 1.5))
        demo_strength = float(self.params.get("demo_strength", 0.0))
        direction = str(self.params.get("direction", "high"))

        n_module = max(1, min(d - 1, int(round(d * module_frac))))
        perm = rng.shuffle_indices(d)
        module_cols = perm[:n_module]
        non_module_cols = perm[n_module:]

        # Latent score per row: standardized mean of module-column values.
        module_vals = X[:, module_cols]
        col_mean = module_vals.mean(dim=0, keepdim=True)
        col_std = module_vals.std(dim=0, keepdim=True).clamp(min=1e-6)
        z_module = ((module_vals - col_mean) / col_std).mean(dim=1)
        if direction == "low":
            z_module = -z_module

        # Optional demographic-gate component (mimics real-world
        # confounding between module refusal and demographics).
        z_demo = torch.zeros(n, dtype=z_module.dtype)
        if demo_strength > 0.0 and len(non_module_cols) > 0:
            demo_col = int(non_module_cols[0])
            d_vals = X[:, demo_col]
            z_demo = (d_vals - d_vals.mean()) / d_vals.std().clamp(min=1e-6)

        from math import log
        bias = log(max(baseline, 1e-4) / max(1.0 - baseline, 1e-4))
        p_refuse = torch.sigmoid(bias + strength * z_module + demo_strength * z_demo)
        refused = rng.rand(n) < p_refuse

        R = torch.ones(n, d, dtype=torch.bool)
        for c in module_cols:
            R[refused, c] = False

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
