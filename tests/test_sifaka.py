"""
Tests for the main Sifaka package.
"""

import pytest
import sifaka


def test_version():
    """Test that the version is defined."""
    assert hasattr(sifaka, "__version__")
    assert isinstance(sifaka.__version__, str)
    assert sifaka.__version__ != ""


def test_lazy_imports():
    """Test that lazy imports work correctly."""
    # Test that Chain is lazily imported
    assert "Chain" in sifaka.__all__
    chain = sifaka.Chain
    assert chain.__name__ == "Chain"

    # Test that ChainResult is lazily imported
    assert "ChainResult" in sifaka.__all__
    chain_result = sifaka.ChainResult
    assert chain_result.__name__ == "ChainResult"

    # Test that Rule is lazily imported
    assert "Rule" in sifaka.__all__
    rule = sifaka.Rule
    assert rule.__name__ == "Rule"


def test_all_exports():
    """Test that all exports are defined."""
    for name in sifaka.__all__:
        assert hasattr(sifaka, name), f"sifaka does not have attribute {name}"
