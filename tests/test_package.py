"""
Package import tests.
"""

import importlib


def test_package_imports() -> None:
    """
    Returns: None, because this function only smoke tests package importing
    """
    package = importlib.import_module("custom_canny_xpu")

    assert package.__name__ == "custom_canny_xpu"
