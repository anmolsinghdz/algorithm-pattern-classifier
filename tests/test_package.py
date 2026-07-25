"""Smoke tests for the package."""

import algorithm_pattern_classifier as apc


def test_package_importable() -> None:
    assert apc is not None


def test_version_exposed() -> None:
    assert apc.__version__ == "0.1.0"
