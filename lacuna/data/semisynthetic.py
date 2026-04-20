"""
lacuna.data.semisynthetic

Semi-synthetic data generation: real data + synthetic missingness.

This module provides utilities for applying synthetic missingness mechanisms
to complete real-world datasets, creating semi-synthetic training data with
known ground-truth mechanism labels.

CRITICAL FIX (2026-01-10):
--------------------------
The original implementation called generator.sample(rng, n, d) which generates
BOTH synthetic X and missingness R based on that synthetic X. We then threw
away the synthetic X and applied R to real X. This broke MAR mechanisms because:

    MAR: Missingness in target depends on value in predictor column
    
    Old (broken): R computed from synthetic_X[:, predictor], applied to real_X
                  -> No actual MAR relationship in the data model sees!
    
    New (fixed):  R computed from real_X[:, predictor]
                  -> True MAR relationship preserved

The fix: Add generator.apply_to(X, rng) method that computes missingness
based on the PROVIDED data X, not internally generated synthetic data.
For generators that don't support this, fall back to the old behavior
with a warning.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import torch

from lacuna.core.types import ObservedDataset
from lacuna.core.rng import RNGState
from lacuna.generators.base import Generator
from lacuna.generators.registry import GeneratorRegistry
from lacuna.generators.priors import GeneratorPrior

from .ingestion import RawDataset


@dataclass(frozen=True)
class SemiSyntheticDataset:
    """A dataset with synthetic missingness applied to real data.
    
    Tracks the source data and the mechanism used.
    """
    observed: ObservedDataset      # The dataset with missingness
    complete: torch.Tensor         # Original complete data [n, d]
    generator_id: int              # Which generator was used
    generator_name: str
    class_id: int                  # MCAR/MAR/MNAR
    source_name: str               # Original dataset name


def apply_missingness(
    raw: RawDataset,
    generator: Generator,
    rng: RNGState,
    dataset_id: Optional[str] = None,
) -> SemiSyntheticDataset:
    """Apply a missingness mechanism to complete data.
    
    CRITICAL FIX: For MAR/MNAR mechanisms to work correctly, we must compute
    missingness based on the ACTUAL data values, not synthetic data.
    
    Args:
        raw: Complete dataset (no missing values).
        generator: Missingness generator to apply.
        rng: RNG state for reproducibility.
        dataset_id: ID for the resulting dataset.
    
    Returns:
        SemiSyntheticDataset with known mechanism.
    """
    n, d = raw.n, raw.d
    
    # Get the complete data as tensor
    X_complete = torch.from_numpy(raw.data.astype('float32'))
    
    # CRITICAL FIX: Use apply_to() if available, which computes missingness
    # based on the provided X, not internally generated synthetic data.
    # This is essential for MAR/MNAR where missingness depends on data values.
    if hasattr(generator, 'apply_to'):
        R = generator.apply_to(X_complete, rng)
    else:
        # Fallback to old behavior with warning
        # The generator.sample() returns (synthetic_X, R) where R was computed
        # based on synthetic_X. This breaks MAR/MNAR for semi-synthetic data.
        warnings.warn(
            f"Generator '{generator.name}' does not support apply_to(). "
            f"Using sample() which may not preserve MAR/MNAR relationships "
            f"in semi-synthetic data. Consider implementing apply_to().",
            UserWarning
        )
        _, R = generator.sample(rng, n, d)
    
    # Ensure at least one observed value per column (prevents degenerate cases)
    for col in range(d):
        if R[:, col].sum() == 0:
            # Make a random row observed
            rand_row = rng.randint(0, n, (1,)).item()
            R[rand_row, col] = True
    
    # Zero out missing values (missing = 0, observed = original value)
    X_observed = X_complete * R.float()
    
    observed_ds = ObservedDataset(
        x=X_observed,
        r=R,
        n=n,
        d=d,
        feature_names=raw.feature_names,
        dataset_id=dataset_id or f"{raw.name}_{generator.name}",
        meta={
            "source": raw.source,
            "generator_id": generator.generator_id,
            "generator_name": generator.name,
            "class_id": generator.class_id,
            "is_semisynthetic": True,
        },
    )
    
    return SemiSyntheticDataset(
        observed=observed_ds,
        complete=X_complete,
        generator_id=generator.generator_id,
        generator_name=generator.name,
        class_id=generator.class_id,
        source_name=raw.name,
    )


def subsample_rows(
    dataset: ObservedDataset,
    max_rows: int,
    rng: RNGState,
) -> ObservedDataset:
    """Subsample rows from a dataset if it exceeds max_rows.
    
    Args:
        dataset: Input dataset.
        max_rows: Maximum number of rows to keep.
        rng: RNG state for reproducibility.
    
    Returns:
        Dataset with at most max_rows rows.
    """
    if dataset.n <= max_rows:
        return dataset
    
    # Random sample without replacement
    indices = rng.choice(dataset.n, size=max_rows, replace=False)
    indices = torch.from_numpy(indices).long()
    
    return ObservedDataset(
        x=dataset.x[indices],
        r=dataset.r[indices],
        n=max_rows,
        d=dataset.d,
        feature_names=dataset.feature_names,
        dataset_id=dataset.dataset_id,
        meta=dataset.meta,
    )


def generate_semisynthetic_batch(
    raw_datasets: List[RawDataset],
    registry: GeneratorRegistry,
    prior: GeneratorPrior,
    rng: RNGState,
    samples_per_dataset: int = 1,
) -> List[SemiSyntheticDataset]:
    """Generate batch of semi-synthetic datasets.
    
    For each raw dataset, applies random missingness mechanisms.
    
    Args:
        raw_datasets: List of complete datasets.
        registry: Generator registry.
        prior: Prior over generators.
        rng: RNG state.
        samples_per_dataset: How many different missingness patterns per dataset.
    
    Returns:
        List of SemiSyntheticDataset objects.
    """
    results = []
    
    for raw in raw_datasets:
        for i in range(samples_per_dataset):
            # Sample a compatible generator (retry if d is too small)
            for attempt in range(20):
                gen_id = prior.sample(rng.spawn())
                generator = registry[gen_id]
                try:
                    ss_dataset = apply_missingness(
                        raw=raw,
                        generator=generator,
                        rng=rng.spawn(),
                        dataset_id=f"{raw.name}_gen{gen_id}_sample{i}",
                    )
                    results.append(ss_dataset)
                    break
                except ValueError:
                    if attempt == 19:
                        raise RuntimeError(
                            f"Could not find compatible generator for "
                            f"dataset '{raw.name}' (d={raw.d}) after 20 attempts"
                        )
    
    return results


class SemiSyntheticDataLoader:
    """Data loader for semi-synthetic data — Lacuna's only data path.

    Takes a pool of real tabular datasets and generates training batches
    by applying synthetic missingness mechanisms per batch. Supports row
    subsampling for large datasets.

    Why semi-synthetic: per Molenberghs, real missingness mechanisms are
    unidentifiable from observed data, so the mechanism has to be synthetic.
    But the underlying X must be real — pure Gaussian-tensor data is too
    "pat" to ground any downstream accuracy claim.

    Optional Little's MCAR cache: if provided at construction, each batch
    receives `little_mcar_stat` and `little_mcar_pvalue` tensors populated
    from the cache keyed on (raw_dataset_name, generator_id). Every
    (raw_dataset, generator) pair the loader can produce MUST be present
    in the cache — missing keys raise KeyError. Build the cache with
    `scripts/build_littles_cache.py`.
    """
    
    def __init__(
        self,
        raw_datasets: List[RawDataset],
        registry: GeneratorRegistry,
        prior: GeneratorPrior,
        max_rows: int,
        max_cols: int,
        batch_size: int,
        batches_per_epoch: int,
        seed: int = 42,
        littles_cache=None,  # Optional[LittlesCache]; late-bound to avoid circular import
        littles_method: str = "mle",
    ):
        """Initialize the semi-synthetic data loader.

        Args:
            raw_datasets: Pool of complete datasets to draw from.
            registry: Generator registry for missingness mechanisms.
            prior: Prior distribution over generators.
            max_rows: Maximum rows per dataset (subsample if larger).
            max_cols: Maximum columns (pad/truncate).
            batch_size: Number of datasets per batch.
            batches_per_epoch: Number of batches per epoch.
            seed: Random seed for reproducibility.
            littles_cache: Optional LittlesCache. If provided, emitted
                TokenBatches carry precomputed MCAR-test scalars. Every
                (raw_dataset.name, generator_id) pair reachable by this
                loader must be present in the cache.
            littles_method: Which cached MCAR test to emit. "mle" (default,
                Little 1988) or "mom" (method-of-moments). Ignored when
                littles_cache is None.
        """
        from .littles_cache import VALID_METHODS

        if littles_method not in VALID_METHODS:
            raise ValueError(
                f"littles_method must be one of {VALID_METHODS}; "
                f"got {littles_method!r}"
            )

        self.raw_datasets = raw_datasets
        self.registry = registry
        self.prior = prior
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.littles_cache = littles_cache
        self.littles_method = littles_method
        self._epoch_counter = 0

        self._class_mapping = registry.get_class_mapping()

        if len(raw_datasets) == 0:
            raise ValueError("Need at least one raw dataset")

        # Validate datasets
        for ds in raw_datasets:
            if ds.d > max_cols:
                raise ValueError(
                    f"Dataset '{ds.name}' has {ds.d} columns, "
                    f"but max_cols={max_cols}. Either increase max_cols "
                    f"or exclude this dataset."
                )

        # Validate cache coverage (fail loud per Coding Bible rule 1).
        if littles_cache is not None:
            missing = [
                (ds.name, gen.generator_id)
                for ds in raw_datasets
                for gen in registry
                if (ds.name, gen.generator_id) not in littles_cache
            ]
            if missing:
                raise ValueError(
                    f"Little's cache is missing {len(missing)} "
                    f"(dataset, generator) pair(s). First 5: {missing[:5]}. "
                    f"Rebuild the cache."
                )
    
    def __len__(self) -> int:
        return self.batches_per_epoch
    
    def __iter__(self):
        import dataclasses
        import torch
        from .tokenization import tokenize_and_batch

        # Use a different seed each epoch so the model sees fresh data
        epoch_seed = self.seed + self._epoch_counter * 1_000_000
        self._epoch_counter += 1
        rng = RNGState(seed=epoch_seed)
        n_datasets = len(self.raw_datasets)

        for batch_idx in range(self.batches_per_epoch):
            batch_rng = rng.spawn()

            datasets = []
            generator_ids = []
            raw_names: List[str] = []

            for i in range(self.batch_size):
                # Pick a random raw dataset
                ds_idx = batch_rng.randint(0, n_datasets, (1,)).item()
                raw = self.raw_datasets[ds_idx]

                # Sample a compatible generator (retry if d is too small)
                max_retries = 20
                for attempt in range(max_retries):
                    gen_id = self.prior.sample(batch_rng.spawn())
                    generator = self.registry[gen_id]
                    try:
                        ss = apply_missingness(
                            raw=raw,
                            generator=generator,
                            rng=batch_rng.spawn(),
                            dataset_id=f"batch{batch_idx}_item{i}",
                        )
                        break
                    except ValueError:
                        # Generator incompatible with this dataset's d
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"Could not find compatible generator for "
                                f"dataset '{raw.name}' (d={raw.d}) after "
                                f"{max_retries} attempts"
                            )

                # Subsample rows if needed
                observed = subsample_rows(
                    ss.observed,
                    max_rows=self.max_rows,
                    rng=batch_rng.spawn(),
                )

                datasets.append(observed)
                generator_ids.append(gen_id)
                raw_names.append(raw.name)

            # Tokenize and batch
            batch = tokenize_and_batch(
                datasets=datasets,
                max_rows=self.max_rows,
                max_cols=self.max_cols,
                generator_ids=generator_ids,
                class_mapping=self._class_mapping,
            )

            # Attach cached MCAR scalars, if a cache was provided. The
            # method selector ("mle" or "mom") picks which pair is emitted.
            if self.littles_cache is not None:
                stats = torch.empty(self.batch_size, dtype=torch.float32)
                pvals = torch.empty(self.batch_size, dtype=torch.float32)
                for i, (ds_name, gid) in enumerate(zip(raw_names, generator_ids)):
                    stat, pval = self.littles_cache.get(
                        ds_name, gid, method=self.littles_method
                    )
                    stats[i] = stat
                    pvals[i] = pval
                batch = dataclasses.replace(
                    batch,
                    little_mcar_stat=stats,
                    little_mcar_pvalue=pvals,
                )

            yield batch
    
    def reset_seed(self, new_seed: int) -> None:
        """Reset the random seed for a new epoch."""
        self.seed = new_seed


