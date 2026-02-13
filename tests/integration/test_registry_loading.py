"""
Integration tests for registry loading.

Tests that all YAML configs load correctly and all generators sample without errors.
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators import load_registry_from_config, list_available_configs


class TestAllConfigsLoad:
    """Test that every YAML config file loads successfully."""

    def test_all_configs_loadable(self):
        configs = list_available_configs()
        for config_name in configs:
            registry = load_registry_from_config(config_name)
            assert registry.K > 0, f"Config {config_name} loaded empty registry"

    def test_all_configs_have_all_classes(self):
        """All configs should have at least one generator per class."""
        configs = list_available_configs()
        for config_name in configs:
            if "fingerprint" in config_name:
                continue  # Fingerprint configs may be specialized
            registry = load_registry_from_config(config_name)
            counts = registry.class_counts()
            assert MCAR in counts, f"{config_name}: no MCAR generators"
            assert MAR in counts, f"{config_name}: no MAR generators"
            assert MNAR in counts, f"{config_name}: no MNAR generators"


class TestMinimal6Sampling:
    """Test that all generators in lacuna_minimal_6 sample correctly."""

    def test_all_generators_sample(self):
        registry = load_registry_from_config("lacuna_minimal_6")
        rng = RNGState(seed=42)

        for gen in registry:
            X, R = gen.sample(rng.spawn(), n=50, d=5)

            assert X.shape == (50, 5), f"Generator {gen.name}: wrong X shape"
            assert R.shape == (50, 5), f"Generator {gen.name}: wrong R shape"
            assert R.dtype == torch.bool, f"Generator {gen.name}: R not bool"
            assert R.sum() >= 1, f"Generator {gen.name}: all values missing"


class TestMinimal18Sampling:
    """Test that all generators in lacuna_minimal_18 sample correctly."""

    def test_load_minimal_18(self):
        registry = load_registry_from_config("lacuna_minimal_18")
        assert registry.K == 18

    def test_class_balance(self):
        registry = load_registry_from_config("lacuna_minimal_18")
        counts = registry.class_counts()
        assert counts[MCAR] == 6
        assert counts[MAR] == 6
        assert counts[MNAR] == 6

    def test_all_generators_sample(self):
        registry = load_registry_from_config("lacuna_minimal_18")
        rng = RNGState(seed=42)

        for gen in registry:
            X, R = gen.sample(rng.spawn(), n=50, d=8)

            assert X.shape == (50, 8), f"Generator {gen.name}: wrong X shape"
            assert R.shape == (50, 8), f"Generator {gen.name}: wrong R shape"
            assert R.dtype == torch.bool, f"Generator {gen.name}: R not bool"
            assert R.sum() >= 1, f"Generator {gen.name}: all values missing"


class TestFullRegistrySampling:
    """Test that the full 110-generator config loads and all generators sample."""

    def test_load_tabular_110(self):
        registry = load_registry_from_config("lacuna_tabular_110")
        assert registry.K >= 100  # Allow some flexibility

    def test_class_distribution(self):
        registry = load_registry_from_config("lacuna_tabular_110")
        counts = registry.class_counts()
        assert counts[MCAR] >= 25
        assert counts[MAR] >= 25
        assert counts[MNAR] >= 25

    @pytest.mark.slow
    def test_all_generators_sample(self):
        """Sample from every generator in the full registry."""
        registry = load_registry_from_config("lacuna_tabular_110")
        rng = RNGState(seed=42)

        failures = []
        for gen in registry:
            try:
                X, R = gen.sample(rng.spawn(), n=50, d=10)
                assert X.shape == (50, 10)
                assert R.shape == (50, 10)
                assert R.dtype == torch.bool
                assert R.sum() >= 1
            except Exception as e:
                failures.append(f"{gen.name}: {e}")

        if failures:
            msg = f"{len(failures)} generators failed:\n" + "\n".join(failures)
            pytest.fail(msg)

    @pytest.mark.slow
    def test_apply_to_all_generators(self):
        """Test apply_to for every generator that supports it."""
        registry = load_registry_from_config("lacuna_tabular_110")
        rng = RNGState(seed=42)

        # Generate some base data
        X = torch.randn(50, 10)

        failures = []
        for gen in registry:
            try:
                R = gen.apply_to(X, rng.spawn())
                assert R.shape == (50, 10)
                assert R.dtype == torch.bool
                assert R.sum() >= 1
            except Exception as e:
                failures.append(f"{gen.name}: {e}")

        if failures:
            msg = f"{len(failures)} generators failed apply_to:\n" + "\n".join(failures)
            pytest.fail(msg)
