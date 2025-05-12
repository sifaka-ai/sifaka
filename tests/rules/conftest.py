"""
Pytest fixtures for rule tests.
"""

import pytest
from sifaka.utils.config import RuleConfig


@pytest.fixture
def rule_config():
    """Fixture for a rule configuration."""
    config = RuleConfig(
        params={
            "min_chars": 10,
            "max_chars": 100,
            "min_words": 2,
            "max_words": 20,
        }
    )
    return config


@pytest.fixture
def length_rule_params():
    """Fixture for length rule parameters."""
    return {
        "min_chars": 10,
        "max_chars": 100,
        "min_words": 2,
        "max_words": 20,
    }


@pytest.fixture
def short_text():
    """Fixture for a short text."""
    return "Short."


@pytest.fixture
def valid_text():
    """Fixture for a valid text."""
    return "This is a valid text that meets all criteria."


@pytest.fixture
def long_text():
    """Fixture for a long text."""
    return "This is a very long text that exceeds the maximum character limit. " * 10
