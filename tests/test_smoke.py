"""Smoke test: the package imports and exposes a version."""

import uplift


def test_package_imports() -> None:
    assert uplift.__version__ == "0.1.0"
