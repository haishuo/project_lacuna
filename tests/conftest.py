"""
Pytest configuration and shared fixtures for Lacuna tests.
"""

import pytest
import torch
from lacuna.core.rng import RNGState
from lacuna.config.schema import LacunaConfig


@pytest.fixture(scope="session")
def iris_littles_cache():
    """Tiny session-scoped Little's MCAR cache for (iris × lacuna_minimal_6).

    Built once per test session. Gives model-forward tests a valid cache
    to point at — the alternative would be mocking or disabling the feature,
    both of which would skip the integration.
    """
    from lacuna.data.catalog import create_default_catalog
    from lacuna.data.littles_cache import build_cache
    from lacuna.generators.families.registry_builder import load_registry_from_config

    iris = create_default_catalog().load("iris")
    registry = load_registry_from_config("lacuna_minimal_6")
    return build_cache(
        raw_datasets=[iris],
        generators=list(registry.generators),
        generator_registry_name="lacuna_minimal_6",
        sample_rows=150,
        seed_base=99999,
    )


@pytest.fixture
def rng():
    """Provide seeded RNG for reproducible tests."""
    return RNGState(seed=42)


@pytest.fixture
def default_config():
    """Provide default LacunaConfig."""
    return LacunaConfig()


@pytest.fixture
def minimal_config():
    """Provide minimal LacunaConfig for fast tests."""
    return LacunaConfig.minimal()


@pytest.fixture
def sample_observed_dataset(rng):
    """Provide sample ObservedDataset."""
    from lacuna.core.types import ObservedDataset
    
    n, d = 100, 5
    x = rng.randn(n, d)
    r = rng.rand(n, d) > 0.2  # ~20% missing
    r[0, 0] = True  # Ensure at least one observed
    
    return ObservedDataset(
        x=x * r.float(),  # Zero out missing
        r=r,
        n=n,
        d=d,
        feature_names=tuple(f"col_{i}" for i in range(d)),
        dataset_id="test_dataset",
    )


@pytest.fixture
def minimal_registry():
    """Provide minimal 6-generator registry."""
    from lacuna.generators import load_registry_from_config
    return load_registry_from_config("lacuna_minimal_6")
