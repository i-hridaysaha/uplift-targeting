"""Smoke test: the package imports and its version matches the project metadata."""

from importlib.metadata import version

import uplift


def test_package_imports() -> None:
    assert uplift.__version__ == version("uplift-targeting")
