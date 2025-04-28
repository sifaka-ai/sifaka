"""
Configuration and fixtures for Sifaka test suite.

This module provides common fixtures and utilities for testing Sifaka components.
"""

import pytest
from unittest.mock import MagicMock

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.models.base import ModelProvider


@pytest.fixture
def sample_text():
    """Return a sample text for testing."""
    return "This is a sample text to use for testing."


@pytest.fixture
def long_text():
    """Return a long text for testing."""
    return " ".join(["word"] * 1000)


@pytest.fixture
def empty_text():
    """Return an empty text for testing."""
    return ""


@pytest.fixture
def mock_model_provider():
    """Return a mock model provider."""
    provider = MagicMock(spec=ModelProvider)
    provider.generate.return_value = "Sample generated text"
    return provider


@pytest.fixture
def mock_rule():
    """Return a mock rule that always passes."""
    rule = MagicMock(spec=Rule)
    rule.validate.return_value = RuleResult(passed=True, message="Validation passed")
    rule.name = "mock_rule"
    rule.description = "Mock rule for testing"
    rule.config = RuleConfig()
    return rule


@pytest.fixture
def mock_failing_rule():
    """Return a mock rule that always fails."""
    rule = MagicMock(spec=Rule)
    rule.validate.return_value = RuleResult(passed=False, message="Validation failed")
    rule.name = "mock_failing_rule"
    rule.description = "Mock failing rule for testing"
    rule.config = RuleConfig()
    return rule
