"""
Tests for lacuna.models.reconstruction

Tests the reconstruction heads for self-supervised pretraining:
    - BaseReconstructionHead: Abstract base class
    - MCARHead: Simple MLP reconstruction
    - MARHead: Cross-attention to observed cells
    - MNARSelfCensoringHead: Censoring-aware reconstruction
    - ReconstructionHeads: Container for all heads
"""

import pytest
import torch
import torch.nn as nn

from lacuna.models.reconstruction import (
    # Config
    ReconstructionConfig,
    # Base class
    BaseReconstructionHead,
    # Head implementations
    MCARHead,
    MARHead,
    MNARSelfCensoringHead,
    # Registry
    HEAD_REGISTRY,
    create_head,
    # Container
    ReconstructionHeads,
    ExtendedReconstructionResult,
    create_reconstruction_heads,
)
from lacuna.core.types import ReconstructionResult
from lacuna.data.tokenization import TOKEN_DIM, IDX_OBSERVED


# =============================================================================
# Test ReconstructionConfig
# =============================================================================

class TestReconstructionConfig:
    """Tests for ReconstructionConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ReconstructionConfig()
        
        assert config.hidden_dim == 128
        assert config.head_hidden_dim == 64
        assert config.n_head_layers == 2
        assert config.dropout == 0.1
        assert config.mnar_variants == ["self_censoring"]

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReconstructionConfig(
            hidden_dim=256,
            head_hidden_dim=128,
            n_head_layers=3,
            dropout=0.2,
            mnar_variants=["self_censoring"],
        )

        assert config.hidden_dim == 256
        assert config.head_hidden_dim == 128
        assert config.n_head_layers == 3
        assert config.dropout == 0.2
        assert config.mnar_variants == ["self_censoring"]
    
    def test_mnar_variants_default_initialization(self):
        """Test that mnar_variants defaults to single variant if None."""
        config = ReconstructionConfig(mnar_variants=None)

        assert config.mnar_variants == ["self_censoring"]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return ReconstructionConfig(
        hidden_dim=64,
        head_hidden_dim=32,
        n_head_layers=2,
        dropout=0.0,
        mnar_variants=["self_censoring"],
    )


@pytest.fixture
def sample_inputs():
    """Create sample input tensors."""
    B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
    
    token_repr = torch.randn(B, max_rows, max_cols, hidden_dim)
    
    # Create tokens with observed indicator
    tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
    tokens[..., IDX_OBSERVED] = (torch.rand(B, max_rows, max_cols) > 0.2).float()
    
    row_mask = torch.ones(B, max_rows, dtype=torch.bool)
    col_mask = torch.ones(B, max_cols, dtype=torch.bool)
    
    original_values = torch.randn(B, max_rows, max_cols)
    reconstruction_mask = torch.rand(B, max_rows, max_cols) < 0.15  # 15% masked
    
    return {
        "token_repr": token_repr,
        "tokens": tokens,
        "row_mask": row_mask,
        "col_mask": col_mask,
        "original_values": original_values,
        "reconstruction_mask": reconstruction_mask,
    }


# =============================================================================
# Test MCARHead
# =============================================================================

class TestMCARHead:
    """Tests for MCARHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MCARHead."""
        return MCARHead(config)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_compute_error(self, head, sample_inputs):
        """Test error computation."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        per_sample_error, per_cell_error = head.compute_error(
            predictions,
            sample_inputs["original_values"],
            sample_inputs["reconstruction_mask"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert per_sample_error.shape == (4,)
        assert per_cell_error.shape == (4, 32, 16)
        assert (per_sample_error >= 0).all()
    
    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test MARHead
# =============================================================================

class TestMARHead:
    """Tests for MARHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MARHead."""
        return MARHead(config)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_uses_cross_attention(self, head, sample_inputs):
        """Test that cross-attention layers exist."""
        assert hasattr(head, 'query_proj')
        assert hasattr(head, 'key_proj')
        assert hasattr(head, 'value_proj')
    
    def test_attends_to_observed_only(self, head, sample_inputs):
        """Test that attention only considers observed cells."""
        # Create input where only first column is observed
        tokens = sample_inputs["tokens"].clone()
        tokens[..., IDX_OBSERVED] = 0.0
        tokens[:, :, 0, IDX_OBSERVED] = 1.0  # Only first column observed
        
        predictions = head(
            sample_inputs["token_repr"],
            tokens,
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        # Should still produce valid output
        assert predictions.shape == (4, 32, 16)
        assert not torch.isnan(predictions).any()
    
    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test MNARSelfCensoringHead
# =============================================================================

class TestMNARSelfCensoringHead:
    """Tests for MNARSelfCensoringHead."""
    
    @pytest.fixture
    def head(self, config):
        """Create MNARSelfCensoringHead."""
        return MNARSelfCensoringHead(config)
    
    def test_output_shape(self, head, sample_inputs):
        """Test output tensor shape."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        B, max_rows, max_cols = 4, 32, 16
        assert predictions.shape == (B, max_rows, max_cols)
    
    def test_no_nan_or_inf(self, head, sample_inputs):
        """Test that output contains no NaN or Inf values."""
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_has_censoring_components(self, head):
        """Test that censoring-related components exist."""
        assert hasattr(head, 'censoring_estimator')
        assert hasattr(head, 'value_predictor')
        assert hasattr(head, 'censoring_scale')

    def test_gradients_flow(self, head, sample_inputs):
        """Test that gradients flow through the head."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        predictions = head(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        loss = predictions.sum()
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test HEAD_REGISTRY and create_head
# =============================================================================

class TestHeadRegistry:
    """Tests for HEAD_REGISTRY and create_head factory."""
    
    def test_registry_contains_all_heads(self):
        """Test that registry contains all head types."""
        expected_heads = ["mcar", "mar", "self_censoring"]

        for name in expected_heads:
            assert name in HEAD_REGISTRY
    
    def test_create_head_mcar(self, config):
        """Test creating MCAR head via factory."""
        head = create_head("mcar", config)
        
        assert isinstance(head, MCARHead)
    
    def test_create_head_mar(self, config):
        """Test creating MAR head via factory."""
        head = create_head("mar", config)
        
        assert isinstance(head, MARHead)
    
    def test_create_head_self_censoring(self, config):
        """Test creating self-censoring head via factory."""
        head = create_head("self_censoring", config)
        
        assert isinstance(head, MNARSelfCensoringHead)
    
    def test_create_head_invalid_raises(self, config):
        """Test that creating invalid head raises error."""
        with pytest.raises(KeyError, match="Unknown head type"):
            create_head("invalid_head", config)


# =============================================================================
# Test ReconstructionHeads Container
# =============================================================================

class TestReconstructionHeads:
    """Tests for ReconstructionHeads container."""
    
    @pytest.fixture
    def heads(self, config):
        """Create ReconstructionHeads container."""
        return ReconstructionHeads(config)
    
    def test_n_heads_property(self, heads):
        """Test n_heads property."""
        # MCAR + MAR + 1 MNAR variant = 3
        assert heads.n_heads == 3
    
    def test_head_names_property(self, heads):
        """Test head_names ordering."""
        expected = ["mcar", "mar", "self_censoring"]
        assert heads.head_names == expected
    
    def test_forward_returns_dict(self, heads, sample_inputs):
        """Test that forward returns dict of ExtendedReconstructionResult."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )

        assert isinstance(results, dict)
        assert len(results) == 3

        for name in heads.head_names:
            assert name in results
            assert isinstance(results[name], ExtendedReconstructionResult)
    
    def test_forward_with_targets(self, heads, sample_inputs):
        """Test forward with original values and reconstruction mask."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
        )
        
        for name in heads.head_names:
            result = results[name]
            assert result.predictions.shape == (4, 32, 16)
            assert result.errors.shape == (4,)
            assert result.per_cell_errors is not None
    
    def test_forward_without_targets(self, heads, sample_inputs):
        """Test forward without targets returns zero errors."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            # No original_values or reconstruction_mask
        )

        for name in heads.head_names:
            result = results[name]
            assert result.predictions.shape == (4, 32, 16)
            assert (result.errors == 0).all()
            # per_cell_errors is zero tensor (not None) when no targets provided
            assert (result.per_cell_errors == 0).all()
    
    def test_get_error_tensor(self, heads, sample_inputs):
        """Test get_error_tensor method."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
        )
        
        error_tensor = heads.get_error_tensor(results)
        
        assert error_tensor.shape == (4, 3)  # B, n_heads
    
    def test_get_predictions_dict(self, heads, sample_inputs):
        """Test get_predictions_dict method."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        predictions = heads.get_predictions_dict(results)
        
        assert isinstance(predictions, dict)
        assert len(predictions) == 3
        
        for name in heads.head_names:
            assert predictions[name].shape == (4, 32, 16)
    
    def test_no_nan_or_inf_in_results(self, heads, sample_inputs):
        """Test that all results contain no NaN or Inf values."""
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
            original_values=sample_inputs["original_values"],
            reconstruction_mask=sample_inputs["reconstruction_mask"],
        )
        
        for name in heads.head_names:
            result = results[name]
            assert not torch.isnan(result.predictions).any()
            assert not torch.isinf(result.predictions).any()
            assert not torch.isnan(result.errors).any()
    
    def test_gradients_flow_through_all_heads(self, heads, sample_inputs):
        """Test that gradients flow through all heads."""
        sample_inputs["token_repr"].requires_grad_(True)
        
        results = heads(
            sample_inputs["token_repr"],
            sample_inputs["tokens"],
            sample_inputs["row_mask"],
            sample_inputs["col_mask"],
        )
        
        # Sum all predictions for loss
        loss = sum(results[name].predictions.sum() for name in heads.head_names)
        loss.backward()
        
        assert sample_inputs["token_repr"].grad is not None


# =============================================================================
# Test create_reconstruction_heads Factory
# =============================================================================

class TestCreateReconstructionHeads:
    """Tests for create_reconstruction_heads factory function."""
    
    def test_default_configuration(self):
        """Test creating heads with default configuration."""
        heads = create_reconstruction_heads()

        assert isinstance(heads, ReconstructionHeads)
        assert heads.n_heads == 3

    def test_custom_configuration(self):
        """Test creating heads with custom configuration."""
        heads = create_reconstruction_heads(
            hidden_dim=256,
            head_hidden_dim=128,
            n_head_layers=3,
            dropout=0.2,
            mnar_variants=["self_censoring"],
        )

        assert heads.n_heads == 3  # MCAR + MAR + 1 variant
        assert heads.config.hidden_dim == 256
        assert heads.config.head_hidden_dim == 128
    
    def test_forward_pass(self):
        """Test forward pass with factory-created heads."""
        heads = create_reconstruction_heads(
            hidden_dim=64,
            head_hidden_dim=32,
            mnar_variants=["self_censoring"],
        )
        
        B, max_rows, max_cols, hidden_dim = 4, 32, 16, 64
        token_repr = torch.randn(B, max_rows, max_cols, hidden_dim)
        tokens = torch.randn(B, max_rows, max_cols, TOKEN_DIM)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        results = heads(token_repr, tokens, row_mask, col_mask)
        
        assert len(results) == 3  # MCAR + MAR + self_censoring


# =============================================================================
# Test BaseReconstructionHead.compute_error
# =============================================================================

class TestComputeError:
    """Tests for error computation in reconstruction heads."""
    
    @pytest.fixture
    def head(self, config):
        """Create a head for testing."""
        return MCARHead(config)
    
    def test_error_only_on_masked_cells(self, head):
        """Test that error is only computed on masked cells."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        # Only mask first cell
        reconstruction_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.bool)
        reconstruction_mask[:, 0, 0] = True
        
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        per_sample_error, per_cell_error = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Per-sample error should be error at [0,0] only
        expected = (predictions[:, 0, 0] - targets[:, 0, 0]) ** 2
        assert torch.allclose(per_sample_error, expected)
    
    def test_respects_row_mask(self, head):
        """Test that row mask is respected."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        # Mask many cells
        reconstruction_mask = torch.ones(B, max_rows, max_cols, dtype=torch.bool)
        
        # But only first 2 rows are valid
        row_mask = torch.zeros(B, max_rows, dtype=torch.bool)
        row_mask[:, :2] = True
        
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        per_sample_error, _ = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Error should only consider first 2 rows
        valid_errors = (predictions[:, :2, :] - targets[:, :2, :]) ** 2
        expected = valid_errors.mean(dim=(1, 2))
        assert torch.allclose(per_sample_error, expected)
    
    def test_respects_col_mask(self, head):
        """Test that column mask is respected."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        reconstruction_mask = torch.ones(B, max_rows, max_cols, dtype=torch.bool)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        
        # Only first 2 columns valid
        col_mask = torch.zeros(B, max_cols, dtype=torch.bool)
        col_mask[:, :2] = True
        
        per_sample_error, _ = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Error should only consider first 2 columns
        valid_errors = (predictions[:, :, :2] - targets[:, :, :2]) ** 2
        expected = valid_errors.mean(dim=(1, 2))
        assert torch.allclose(per_sample_error, expected)
    
    def test_handles_no_masked_cells(self, head):
        """Test handling of no masked cells."""
        B, max_rows, max_cols = 2, 8, 4
        
        predictions = torch.randn(B, max_rows, max_cols)
        targets = torch.randn(B, max_rows, max_cols)
        
        # No cells masked
        reconstruction_mask = torch.zeros(B, max_rows, max_cols, dtype=torch.bool)
        row_mask = torch.ones(B, max_rows, dtype=torch.bool)
        col_mask = torch.ones(B, max_cols, dtype=torch.bool)
        
        per_sample_error, _ = head.compute_error(
            predictions, targets, reconstruction_mask, row_mask, col_mask
        )
        
        # Should handle gracefully (clamp min=1.0 prevents div by zero)
        assert not torch.isnan(per_sample_error).any()