"""Minimal setup.py so `pip install -e .` works.

Lacuna is research code; this exists only to make the package importable
from sibling tools (the demo, scripts, notebooks). No version metadata,
no entry points — pin those if/when the project is packaged for release.
"""

from setuptools import find_packages, setup

setup(
    name="lacuna",
    version="0.0.0",
    description="Project Lacuna — missing data mechanism classification",
    packages=find_packages(include=["lacuna", "lacuna.*"]),
    python_requires=">=3.11",
)
