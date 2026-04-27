"""
Tests for lacuna.generators.families (MCAR, MAR, MNAR)
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mcar import MCARBernoulli, MCARColumnGaussian
from lacuna.generators.families.mcar.blocks import MCARRotatedBooklet
from lacuna.generators.families.mar import MARLogistic, MARMultiPredictor
from lacuna.generators.families.mnar import MNARLogistic, MNARSelfCensorHigh


class TestMCARBernoulli:
    """Tests for MCARBernoulli generator."""
    
    def test_construction(self):
        gen = MCARBernoulli(0, "test", GeneratorParams(miss_rate=0.2))
        assert gen.class_id == MCAR
    
    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="miss_rate"):
            MCARBernoulli(0, "test", GeneratorParams())
    
    def test_invalid_miss_rate_raises(self):
        with pytest.raises(ValueError, match="miss_rate"):
            MCARBernoulli(0, "test", GeneratorParams(miss_rate=1.5))
        with pytest.raises(ValueError, match="miss_rate"):
            MCARBernoulli(0, "test", GeneratorParams(miss_rate=-0.1))
    
    def test_sample_shapes(self):
        gen = MCARBernoulli(0, "test", GeneratorParams(miss_rate=0.3))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)
        assert R.dtype == torch.bool
    
    def test_approximate_miss_rate(self):
        gen = MCARBernoulli(0, "test", GeneratorParams(miss_rate=0.3))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=10000, d=10)
        
        empirical_rate = 1.0 - R.float().mean().item()
        assert abs(empirical_rate - 0.3) < 0.02
    
    def test_at_least_one_observed(self):
        gen = MCARBernoulli(0, "test", GeneratorParams(miss_rate=0.99))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=10, d=2)
        
        assert R.sum() >= 1


class TestMCARColumnGaussian:
    """Tests for MCARColumnGaussian generator."""
    
    def test_sample_different_column_rates(self):
        gen = MCARColumnGaussian(0, "test", GeneratorParams(miss_rate_range=(0.1, 0.5)))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=5000, d=5)
        
        col_rates = 1.0 - R.float().mean(dim=0)
        
        # All rates should be in range
        assert (col_rates >= 0.05).all()  # Allow some margin
        assert (col_rates <= 0.55).all()
        
        # Rates should vary between columns
        assert col_rates.std() > 0.01


class TestMCARRotatedBooklet:
    """Tests for MCARRotatedBooklet generator (planned-missing rotated-booklet design)."""

    def test_construction(self):
        gen = MCARRotatedBooklet(0, "test",
            GeneratorParams(n_blocks=4, universal_frac=0.3))
        assert gen.class_id == MCAR

    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="n_blocks"):
            MCARRotatedBooklet(0, "test", GeneratorParams(universal_frac=0.3))
        with pytest.raises(ValueError, match="universal_frac"):
            MCARRotatedBooklet(0, "test", GeneratorParams(n_blocks=4))

    def test_invalid_n_blocks_raises(self):
        with pytest.raises(ValueError, match="n_blocks >= 2"):
            MCARRotatedBooklet(0, "test",
                GeneratorParams(n_blocks=1, universal_frac=0.3))

    def test_invalid_universal_frac_raises(self):
        with pytest.raises(ValueError, match="universal_frac"):
            MCARRotatedBooklet(0, "test",
                GeneratorParams(n_blocks=3, universal_frac=1.0))
        with pytest.raises(ValueError, match="universal_frac"):
            MCARRotatedBooklet(0, "test",
                GeneratorParams(n_blocks=3, universal_frac=-0.1))

    def test_sample_shapes(self):
        gen = MCARRotatedBooklet(0, "test",
            GeneratorParams(n_blocks=3, universal_frac=0.3))
        X, R = gen.sample(RNGState(seed=42), n=100, d=10)
        assert X.shape == (100, 10)
        assert R.shape == (100, 10)
        assert R.dtype == torch.bool

    def test_universal_columns_always_observed(self):
        # 50% universal of 10 cols = 5 universal cols always observed.
        gen = MCARRotatedBooklet(0, "test",
            GeneratorParams(n_blocks=4, universal_frac=0.5))
        X, R = gen.sample(RNGState(seed=42), n=200, d=10)
        col_observed_rate = R.float().mean(dim=0)
        # Exactly n_universal columns should be 100% observed.
        n_fully_observed = int((col_observed_rate == 1.0).sum())
        assert n_fully_observed == 5

    def test_rotated_columns_have_block_rate(self):
        # K=4 blocks → each rotated col observed in ~1/4 of rows.
        gen = MCARRotatedBooklet(0, "test",
            GeneratorParams(n_blocks=4, universal_frac=0.0))
        X, R = gen.sample(RNGState(seed=42), n=2000, d=12)
        col_rates = R.float().mean(dim=0)
        # Every column is rotated; expected rate ~0.25 per col.
        assert (col_rates >= 0.20).all() and (col_rates <= 0.30).all()

    def test_rows_assigned_to_one_block(self):
        # With universal_frac=0 and K blocks of equal size, every row
        # should observe exactly one block's worth of columns.
        gen = MCARRotatedBooklet(0, "test",
            GeneratorParams(n_blocks=3, universal_frac=0.0))
        X, R = gen.sample(RNGState(seed=42), n=300, d=9)  # 3 blocks of 3 cols
        per_row_observed = R.sum(dim=1)
        # Every row should have exactly 3 cols observed (one block).
        assert (per_row_observed == 3).all()

    def test_at_least_one_observed_with_extreme_params(self):
        # Even with very few rotated cols, output must not be all-missing.
        gen = MCARRotatedBooklet(0, "test",
            GeneratorParams(n_blocks=10, universal_frac=0.0))
        X, R = gen.sample(RNGState(seed=42), n=5, d=2)
        assert R.sum() >= 1


class TestMARLogistic:
    """Tests for MARLogistic generator."""
    
    def test_construction(self):
        gen = MARLogistic(0, "test", GeneratorParams(alpha0=0.0, alpha1=1.0))
        assert gen.class_id == MAR
    
    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="alpha0"):
            MARLogistic(0, "test", GeneratorParams(alpha1=1.0))
    
    def test_requires_d_ge_2(self):
        gen = MARLogistic(0, "test", GeneratorParams(alpha0=0.0, alpha1=1.0))
        rng = RNGState(seed=42)
        
        with pytest.raises(ValueError, match="d >= 2"):
            gen.sample(rng, n=100, d=1)
    
    def test_sample_shapes(self):
        gen = MARLogistic(0, "test", GeneratorParams(alpha0=0.0, alpha1=1.0))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)
    
    def test_missingness_depends_on_predictor(self):
        """Verify MAR: missingness correlates with predictor column."""
        gen = MARLogistic(
            0, "test",
            GeneratorParams(
                alpha0=0.0,
                alpha1=2.0,  # Strong positive relationship
                target_col_idx=-1,  # Last column is target
                predictor_col_idx=0,  # First column is predictor
            )
        )
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=5000, d=3)
        
        # Split by predictor value
        predictor = X[:, 0]
        target_missing = ~R[:, -1]
        
        high_pred = predictor > predictor.median()
        
        miss_rate_high = target_missing[high_pred].float().mean()
        miss_rate_low = target_missing[~high_pred].float().mean()
        
        # Higher predictor -> higher missingness (positive alpha1)
        assert miss_rate_high > miss_rate_low + 0.1


class TestMARMultiPredictor:
    """Tests for MARMultiPredictor generator."""
    
    def test_sample_shapes(self):
        gen = MARMultiPredictor(
            0, "test",
            GeneratorParams(alpha0=0.0, alphas=[1.0, 0.5])
        )
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)


class TestMNARLogistic:
    """Tests for MNARLogistic generator."""
    
    def test_construction(self):
        gen = MNARLogistic(0, "test", GeneratorParams(beta0=0.0, beta2=1.0))
        assert gen.class_id == MNAR
    
    def test_beta2_zero_raises(self):
        with pytest.raises(ValueError, match="beta2 must be non-zero"):
            MNARLogistic(0, "test", GeneratorParams(beta0=0.0, beta2=0.0))
    
    def test_sample_shapes(self):
        gen = MNARLogistic(0, "test", GeneratorParams(beta0=0.0, beta2=1.0))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)
    
    def test_missingness_depends_on_target(self):
        """Verify MNAR: missingness correlates with target column value."""
        gen = MNARLogistic(
            0, "test",
            GeneratorParams(
                beta0=0.0,
                beta1=0.0,  # No predictor effect
                beta2=2.0,  # Strong target effect
                target_col_idx=-1,
                predictor_col_idx=0,
            )
        )
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=5000, d=3)
        
        # For MNAR, we need to look at the COMPLETE data X
        # (before missingness is applied)
        target_vals = X[:, -1]
        target_missing = ~R[:, -1]
        
        high_vals = target_vals > target_vals.median()
        
        miss_rate_high = target_missing[high_vals].float().mean()
        miss_rate_low = target_missing[~high_vals].float().mean()
        
        # Higher target value -> higher missingness (positive beta2)
        assert miss_rate_high > miss_rate_low + 0.1


class TestMNARSelfCensorHigh:
    """Tests for MNARSelfCensorHigh generator."""
    
    def test_construction(self):
        gen = MNARSelfCensorHigh(0, "test", GeneratorParams(beta0=0.0, beta1=1.0))
        assert gen.class_id == MNAR
    
    def test_beta1_zero_raises(self):
        with pytest.raises(ValueError, match="beta1 must be non-zero"):
            MNARSelfCensorHigh(0, "test", GeneratorParams(beta0=0.0, beta1=0.0))
    
    def test_sample_shapes(self):
        gen = MNARSelfCensorHigh(0, "test", GeneratorParams(beta0=0.0, beta1=1.0))
        rng = RNGState(seed=42)
        
        X, R = gen.sample(rng, n=100, d=5)
        
        assert X.shape == (100, 5)
        assert R.shape == (100, 5)


class TestBaseDataSamplers:
    """Tests for base data sampling functions."""
    
    def test_sample_gaussian(self):
        from lacuna.generators.families.base_data import sample_gaussian
        
        rng = RNGState(seed=42)
        X = sample_gaussian(rng, n=1000, d=5, mean=10.0, std=2.0)
        
        assert X.shape == (1000, 5)
        assert abs(X.mean().item() - 10.0) < 0.2
        assert abs(X.std().item() - 2.0) < 0.2
    
    def test_sample_gaussian_correlated(self):
        from lacuna.generators.families.base_data import sample_gaussian_correlated
        
        rng = RNGState(seed=42)
        X = sample_gaussian_correlated(rng, n=5000, d=5, rho=0.8)
        
        # Check adjacent columns are correlated
        corr = torch.corrcoef(X.T)
        assert corr[0, 1].item() > 0.7  # Should be close to 0.8
    
    def test_sample_uniform(self):
        from lacuna.generators.families.base_data import sample_uniform
        
        rng = RNGState(seed=42)
        X = sample_uniform(rng, n=1000, d=5, low=2.0, high=5.0)
        
        assert X.shape == (1000, 5)
        assert (X >= 2.0).all()
        assert (X <= 5.0).all()
    
    def test_sample_mixed(self):
        from lacuna.generators.families.base_data import sample_mixed
        
        rng = RNGState(seed=42)
        X = sample_mixed(rng, n=100, d=6, gaussian_cols=4)
        
        assert X.shape == (100, 6)
