"""
Registry builder with automatic generator discovery.

Scans mcar/, mar/, mnar/ folders and auto-discovers all Generator subclasses.
No manual registration needed — follow the naming convention and it just works.

Naming convention:
    Class name: {FAMILY}{Variant} (e.g., MCARBernoulli, MNARThresholdLeft)
    Registry key: {family}.{Variant} (e.g., mcar.Bernoulli, mnar.ThresholdLeft)
"""

from typing import Dict, Type, List
from pathlib import Path
import yaml
import importlib
import pkgutil

from lacuna.generators.base import Generator
from lacuna.generators.params import GeneratorParams
from lacuna.generators.registry import GeneratorRegistry

# Cache for discovered generator classes
_GENERATOR_CLASSES: Dict[str, Type[Generator]] = {}
_DISCOVERED = False


def _discover_generators() -> None:
    """Auto-discover all generator classes in mcar/, mar/, mnar/ submodules.

    Scans each family directory, imports all non-private modules, and registers
    any Generator subclass found. Classes are keyed as "{family}.{Variant}"
    where Variant is the class name with the family prefix stripped.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    families_path = Path(__file__).parent

    for family_name in ["mcar", "mar", "mnar"]:
        family_path = families_path / family_name

        if not family_path.is_dir():
            continue

        for module_info in pkgutil.iter_modules([str(family_path)]):
            module_name = module_info.name

            if module_name.startswith("_"):
                continue

            module_path = f"lacuna.generators.families.{family_name}.{module_name}"
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                print(f"Warning: Could not import {module_path}: {e}")
                continue

            family_upper = family_name.upper()
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                if (
                    isinstance(attr, type)
                    and issubclass(attr, Generator)
                    and attr is not Generator
                    and attr_name.startswith(family_upper)
                ):
                    variant = attr_name[len(family_upper):]
                    key = f"{family_name}.{variant}"
                    _GENERATOR_CLASSES[key] = attr

    _DISCOVERED = True


def _load_generator_class(family: str, variant: str) -> Type[Generator]:
    """Load generator class by family and variant.

    Args:
        family: "mcar", "mar", or "mnar"
        variant: Variant name without family prefix (e.g., "Bernoulli")

    Returns:
        Generator class

    Raises:
        ValueError: If generator not found
    """
    _discover_generators()

    key = f"{family}.{variant}"
    if key not in _GENERATOR_CLASSES:
        available = sorted(_GENERATOR_CLASSES.keys())
        raise ValueError(
            f"Unknown generator: {key}\n"
            f"Available generators:\n" + "\n".join(f"  - {k}" for k in available)
        )

    return _GENERATOR_CLASSES[key]


def load_registry_from_yaml(config_path: Path) -> GeneratorRegistry:
    """Load generator registry from YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        GeneratorRegistry built from config
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    generators = []
    for i, spec in enumerate(config["generators"]):
        family = spec["family"]
        variant = spec["variant"]
        name = spec["name"]
        params = spec.get("params", {})

        cls = _load_generator_class(family, variant)
        gen = cls(
            generator_id=i,
            name=name,
            params=GeneratorParams(**params),
        )
        generators.append(gen)

    return GeneratorRegistry(tuple(generators))


def load_registry_from_config(config_name: str) -> GeneratorRegistry:
    """Load registry from named config in configs/generators/.

    Also accepts absolute paths or paths ending in .yaml.

    Args:
        config_name: Name of config (without .yaml extension) or path to YAML file

    Returns:
        GeneratorRegistry
    """
    config_path = Path(config_name)

    # If it looks like a path (absolute or has .yaml), use directly
    if config_path.is_absolute() or config_name.endswith(".yaml"):
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        return load_registry_from_yaml(config_path)

    # Otherwise resolve from configs/generators/
    config_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "generators"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        available = list_available_configs()
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Available configs: {available}"
        )

    return load_registry_from_yaml(config_path)


def list_available_configs() -> List[str]:
    """List all available generator config names."""
    config_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "generators"
    if not config_dir.exists():
        return []
    return sorted([p.stem for p in config_dir.glob("*.yaml")])


def list_discovered_generators() -> Dict[str, Type[Generator]]:
    """Return all discovered generator classes (useful for debugging)."""
    _discover_generators()
    return dict(_GENERATOR_CLASSES)
