"""
Tests for lacuna.generators.families.registry_builder

Tests auto-discovery, YAML loading, and error handling.
"""

import pytest
import torch
from pathlib import Path
from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators.families.registry_builder import (
    _discover_generators,
    _load_generator_class,
    load_registry_from_config,
    load_registry_from_yaml,
    list_available_configs,
    list_discovered_generators,
)


class TestAutoDiscovery:
    """Tests for automatic generator class discovery."""

    def test_discovers_mcar_generators(self):
        generators = list_discovered_generators()
        mcar_keys = [k for k in generators if k.startswith("mcar.")]
        assert len(mcar_keys) >= 30

    def test_discovers_mar_generators(self):
        generators = list_discovered_generators()
        mar_keys = [k for k in generators if k.startswith("mar.")]
        assert len(mar_keys) >= 36

    def test_discovers_mnar_generators(self):
        generators = list_discovered_generators()
        mnar_keys = [k for k in generators if k.startswith("mnar.")]
        assert len(mnar_keys) >= 42

    def test_total_discovered(self):
        generators = list_discovered_generators()
        assert len(generators) >= 108

    def test_key_format(self):
        generators = list_discovered_generators()
        for key in generators:
            parts = key.split(".")
            assert len(parts) == 2
            assert parts[0] in ("mcar", "mar", "mnar")
            assert len(parts[1]) > 0

    def test_known_generators_discovered(self):
        generators = list_discovered_generators()
        assert "mcar.Bernoulli" in generators
        assert "mcar.ColumnGaussian" in generators
        assert "mar.Logistic" in generators
        assert "mar.MultiColumn" in generators
        assert "mnar.SelfCensorHigh" in generators
        assert "mnar.ThresholdLeft" in generators


class TestLoadGeneratorClass:
    """Tests for _load_generator_class."""

    def test_load_known_class(self):
        cls = _load_generator_class("mcar", "Bernoulli")
        assert cls.__name__ == "MCARBernoulli"

    def test_load_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            _load_generator_class("mcar", "NonexistentGenerator")

    def test_load_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            _load_generator_class("invalid", "Bernoulli")


class TestLoadRegistryFromConfig:
    """Tests for loading registries from named configs."""

    def test_load_minimal_6(self):
        registry = load_registry_from_config("lacuna_minimal_6")
        assert registry.K == 6

    def test_load_minimal_6_class_balance(self):
        registry = load_registry_from_config("lacuna_minimal_6")
        counts = registry.class_counts()
        assert counts[MCAR] == 2
        assert counts[MAR] == 2
        assert counts[MNAR] == 2

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_registry_from_config("nonexistent_config_xyz")

    def test_sequential_ids(self):
        registry = load_registry_from_config("lacuna_minimal_6")
        ids = [gen.generator_id for gen in registry]
        assert ids == list(range(registry.K))


class TestListAvailableConfigs:
    """Tests for listing available configs."""

    def test_lists_configs(self):
        configs = list_available_configs()
        assert isinstance(configs, list)
        assert "lacuna_minimal_6" in configs

    def test_configs_are_sorted(self):
        configs = list_available_configs()
        assert configs == sorted(configs)
