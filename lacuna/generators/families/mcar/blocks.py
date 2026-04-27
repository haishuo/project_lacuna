"""MCAR block-structured missingness generators."""

from typing import Tuple
import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR
from ...base import Generator
from ...params import GeneratorParams
from ..base_data import sample_gaussian


class MCARRandomBlocks(Generator):
    """MCAR with random rectangular blocks of missing data.

    Random rectangular blocks are placed in the matrix; cells inside blocks
    are missing with probability miss_prob.

    Required params:
        block_size: Size of each block (block_size x block_size).
        miss_prob: Probability of placing a block at each grid position.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "block_size" not in params or "miss_prob" not in params:
            raise ValueError("MCARRandomBlocks requires 'block_size' and 'miss_prob'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        block_size = self.params["block_size"]
        miss_prob = self.params["miss_prob"]

        R = torch.ones(n, d, dtype=torch.bool)
        # Iterate over possible block top-left positions on a grid
        for i in range(0, n, block_size):
            for j in range(0, d, block_size):
                if rng.rand(1).item() < miss_prob:
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, d)
                    R[i:i_end, j:j_end] = False

        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARScattered(Generator):
    """MCAR with scattered small blocks of missing data.

    Places n_blocks small rectangular blocks at random positions.

    Required params:
        n_blocks: Number of blocks to place.
        block_rows: Number of rows per block.
        block_cols: Number of columns per block.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "n_blocks" not in params or "block_rows" not in params or "block_cols" not in params:
            raise ValueError("MCARScattered requires 'n_blocks', 'block_rows', and 'block_cols'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        n_blocks = self.params["n_blocks"]
        block_rows = self.params["block_rows"]
        block_cols = self.params["block_cols"]

        R = torch.ones(n, d, dtype=torch.bool)
        for _ in range(n_blocks):
            row_start = rng.randint(0, max(1, n - block_rows + 1), (1,)).item()
            col_start = rng.randint(0, max(1, d - block_cols + 1), (1,)).item()
            row_end = min(row_start + block_rows, n)
            col_end = min(col_start + block_cols, d)
            R[row_start:row_end, col_start:col_end] = False

        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARClustered(Generator):
    """MCAR with spatially clustered missingness around random centers.

    Picks n_clusters random center cells, then makes cells missing within
    cluster_radius (Manhattan distance) of each center.

    Required params:
        n_clusters: Number of cluster centers.
        cluster_radius: Manhattan distance radius around each center.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "n_clusters" not in params or "cluster_radius" not in params:
            raise ValueError("MCARClustered requires 'n_clusters' and 'cluster_radius'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        n_clusters = self.params["n_clusters"]
        cluster_radius = self.params["cluster_radius"]

        R = torch.ones(n, d, dtype=torch.bool)
        for _ in range(n_clusters):
            ci = rng.randint(0, n, (1,)).item()
            cj = rng.randint(0, d, (1,)).item()
            for i in range(max(0, ci - cluster_radius), min(n, ci + cluster_radius + 1)):
                for j in range(max(0, cj - cluster_radius), min(d, cj + cluster_radius + 1)):
                    if abs(i - ci) + abs(j - cj) <= cluster_radius:
                        R[i, j] = False

        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARDiagonal(Generator):
    """MCAR with a diagonal band of missingness.

    Cells near the diagonal (where |i/n - j/d| is small) are missing
    with probability miss_prob within the bandwidth.

    Required params:
        bandwidth: Width of the diagonal band (as fraction of matrix dimension).
        miss_prob: Probability of missingness within the band.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "bandwidth" not in params or "miss_prob" not in params:
            raise ValueError("MCARDiagonal requires 'bandwidth' and 'miss_prob'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        bandwidth = self.params["bandwidth"]
        miss_prob = self.params["miss_prob"]

        # Normalized positions: row i maps to i/(n-1), col j maps to j/(d-1)
        row_pos = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
        col_pos = torch.arange(d, dtype=torch.float32) / max(d - 1, 1)
        # Distance from diagonal
        dist = torch.abs(row_pos.unsqueeze(1) - col_pos.unsqueeze(0))
        in_band = dist <= bandwidth

        # Within the band, each cell is missing with miss_prob
        noise = rng.rand(n, d)
        R = ~(in_band & (noise < miss_prob))

        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARRotatedBooklet(Generator):
    """MCAR rotated-booklet planned-missing design.

    Models the missingness pattern produced by surveys that randomly
    assign each respondent to one of K "booklets" (e.g. PISA, NAEP,
    ECLS-K). Some columns are universal (asked of every respondent);
    the remaining columns are partitioned into K blocks, and each row
    is randomly assigned to one block whose columns it sees observed.
    All other rotated columns are NaN for that row.

    Random booklet assignment is independent of values, so the
    mechanism is MCAR-by-design even though within-row missingness
    is highly structured (entire blocks of columns absent together).

    Required params:
        n_blocks: Number of rotation blocks (booklets). Must be >= 2.
        universal_frac: Fraction of columns that are universal (always
            observed). Must be in [0, 1).
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "n_blocks" not in params or "universal_frac" not in params:
            raise ValueError("MCARRotatedBooklet requires 'n_blocks' and 'universal_frac'")
        if params["n_blocks"] < 2:
            raise ValueError("MCARRotatedBooklet requires n_blocks >= 2")
        if not (0.0 <= params["universal_frac"] < 1.0):
            raise ValueError("MCARRotatedBooklet requires 0 <= universal_frac < 1")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        n_blocks = int(self.params["n_blocks"])
        universal_frac = float(self.params["universal_frac"])

        n_universal = max(0, min(d - 1, int(round(d * universal_frac))))
        n_rotated = d - n_universal

        # If too few rotated cols for the requested blocks, fall back to
        # whatever blocks actually fit; keep at least 2 blocks.
        effective_blocks = min(n_blocks, max(2, n_rotated))

        # Random column permutation: first n_universal are universal,
        # remainder split into effective_blocks roughly-equal partitions.
        perm = rng.shuffle_indices(d).tolist()
        universal_cols = perm[:n_universal]
        rotated_cols = perm[n_universal:]

        block_of_col = torch.full((d,), -1, dtype=torch.long)
        for c in universal_cols:
            block_of_col[c] = -2  # sentinel: always observed
        if n_rotated > 0:
            base, rem = divmod(n_rotated, effective_blocks)
            offset = 0
            for b in range(effective_blocks):
                size = base + (1 if b < rem else 0)
                for c in rotated_cols[offset:offset + size]:
                    block_of_col[c] = b
                offset += size

        # Random booklet assignment per row, uniform over effective_blocks.
        row_block = rng.randint(0, effective_blocks, (n,))

        R = torch.zeros(n, d, dtype=torch.bool)
        for c in range(d):
            bc = block_of_col[c].item()
            if bc == -2:
                R[:, c] = True
            else:
                R[:, c] = (row_block == bc)

        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)


class MCARCheckerboard(Generator):
    """MCAR with alternating blocks of observed/missing in a checkerboard pattern.

    The matrix is divided into blocks of size block_size x block_size.
    Alternating blocks are missing with probability miss_prob.

    Required params:
        block_size: Size of each checkerboard square.
        miss_prob: Probability of missingness within "missing" squares.
    """

    def __init__(self, generator_id: int, name: str, params: GeneratorParams):
        super().__init__(generator_id, name, MCAR, params)
        if "block_size" not in params or "miss_prob" not in params:
            raise ValueError("MCARCheckerboard requires 'block_size' and 'miss_prob'")

    def _compute_missingness(self, n: int, d: int, rng: RNGState) -> torch.Tensor:
        block_size = self.params["block_size"]
        miss_prob = self.params["miss_prob"]

        R = torch.ones(n, d, dtype=torch.bool)
        noise = rng.rand(n, d)
        for i in range(n):
            for j in range(d):
                block_i = i // block_size
                block_j = j // block_size
                if (block_i + block_j) % 2 == 1:
                    if noise[i, j] < miss_prob:
                        R[i, j] = False

        if R.sum() == 0:
            R[0, 0] = True
        return R

    def sample(self, rng: RNGState, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.params.get("base_mean", 0.0)
        std = self.params.get("base_std", 1.0)
        X = sample_gaussian(rng.spawn(), n, d, mean=mean, std=std)
        R = self._compute_missingness(n, d, rng.spawn())
        return X, R

    def apply_to(self, X: torch.Tensor, rng: RNGState) -> torch.Tensor:
        n, d = X.shape
        return self._compute_missingness(n, d, rng)
