"""
lacuna.data.composition_target

Draw a TARGET missingness COMPOSITION for the Stage-B composition-controlled sampler (ADR-0007).

Lacuna's estimand is a calibrated distribution over a dataset's missingness *composition* — the
by-missing-cell fractions `(f_MCAR, f_MAR, f_MNAR)` on the 2-simplex (ADR-0007). The generator is
the operational definition of that estimand, so training data must be produced at compositions
drawn from a BROAD / near-flat prior over the simplex (train prior-agnostic; a deployment may later
supply a real-world prior — ADR-0007 commitment 4). This module owns ONE job: drawing that target.

A target bundles two independent quantities:
  - the composition `(f_MCAR, f_MAR, f_MNAR)` (sums to 1) — *which* mechanisms produced the missing
    cells, in what proportion (the by-cell denominator, LOCKED in ADR-0007);
  - the overall `miss_rate` — *how much* of the matrix is missing. This is orthogonal to the
    composition (a ratio is scale-free) and is drawn broadly so the realised per-column rates VARY
    into the realistic survey regime (Stage-A Band 2). The downstream allocator/sampler turn this
    target into per-column/per-cell mechanism assignments.

Determinism (Coding Bible Rule 6): the Dirichlet / uniform draws flow through the injected RNGState;
given a seed the target is reproducible. Fails loud (Rule 1) on an invalid prior specification.
"""

from dataclasses import dataclass
from typing import Tuple

from lacuna.core.rng import RNGState


@dataclass(frozen=True)
class CompositionTarget:
    """A drawn target for one composed dataset (the by-cell estimand the sampler must realise).

    Attributes:
        f_mcar, f_mar, f_mnar: target fraction of MISSING CELLS attributable to each mechanism.
            Non-negative, sum to 1 (within a small tolerance). This is the by-cell denominator
            LOCKED in ADR-0007 — the same denominator the ground-truth tags and the eval metric use.
        miss_rate: target overall fraction of cells that are missing (in (0, 1)). Orthogonal to the
            composition; drives the per-column rate level (Stage-A Band 2 realism).
    """
    f_mcar: float
    f_mar: float
    f_mnar: float
    miss_rate: float

    _SUM_TOL = 1e-6

    def __post_init__(self) -> None:
        for name, f in (("f_mcar", self.f_mcar), ("f_mar", self.f_mar), ("f_mnar", self.f_mnar)):
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {f}")
        total = self.f_mcar + self.f_mar + self.f_mnar
        if abs(total - 1.0) > self._SUM_TOL:
            raise ValueError(f"composition fractions must sum to 1, got {total} "
                             f"({self.f_mcar}, {self.f_mar}, {self.f_mnar})")
        if not 0.0 < self.miss_rate < 1.0:
            raise ValueError(f"miss_rate must be in (0, 1), got {self.miss_rate}")

    def as_fractions(self) -> Tuple[float, float, float]:
        """Return the composition as an (MCAR, MAR, MNAR) tuple, indexable by the class ids."""
        return (self.f_mcar, self.f_mar, self.f_mnar)


def sample_composition_target(
    rng: RNGState,
    *,
    concentration: float = 1.0,
    miss_rate_range: Tuple[float, float] = (0.05, 0.6),
) -> CompositionTarget:
    """Draw a target composition + overall miss rate from a broad simplex × rate prior.

    Args:
        rng: Explicit RNG state (the Dirichlet/uniform draws flow through its numpy generator).
        concentration: Symmetric Dirichlet concentration for `(f_MCAR, f_MAR, f_MNAR)`. 1.0
            (default) = uniform over the simplex (the broad / prior-agnostic training prior of
            ADR-0007). Values < 1 push mass toward the corners (near-pure datasets); > 1 toward the
            centre (balanced mixtures). Must be > 0.
        miss_rate_range: (lo, hi) for the uniform overall-miss-rate draw. Defaults span the realised
            range of the Stage-A real survey corpus (~0.01–0.60). Requires 0 < lo < hi < 1.

    Returns:
        CompositionTarget.

    Raises:
        ValueError: on concentration <= 0 or an invalid miss_rate_range.
    """
    if concentration <= 0.0:
        raise ValueError(f"concentration must be > 0, got {concentration}")
    lo, hi = miss_rate_range
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"miss_rate_range must satisfy 0 < lo < hi < 1, got {miss_rate_range}")

    np_rng = rng.numpy_rng
    f_mcar, f_mar, f_mnar = (float(x) for x in np_rng.dirichlet([concentration] * 3))
    # Renormalise to absorb float drift so __post_init__'s sum check is exact.
    total = f_mcar + f_mar + f_mnar
    f_mcar, f_mar, f_mnar = f_mcar / total, f_mar / total, f_mnar / total
    miss_rate = float(np_rng.uniform(lo, hi))
    return CompositionTarget(f_mcar=f_mcar, f_mar=f_mar, f_mnar=f_mnar, miss_rate=miss_rate)
