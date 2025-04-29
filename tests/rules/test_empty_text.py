"""
Tests for empty text handling in validators.
"""

import pytest

from sifaka.rules.base import RuleResult
from sifaka.rules.formatting.length import LengthConfig, DefaultLengthValidator
from sifaka.rules.formatting.style import StyleConfig, DefaultStyleValidator, CapitalizationStyle


def test_base_validator_empty_text():
    """Test that BaseValidator handles empty text correctly."""
    # Test with LengthValidator
    config = LengthConfig(min_chars=10, max_chars=100)
    validator = DefaultLengthValidator(config)

    # Test with empty string
    result = validator.validate("")
    assert result.passed is True
    assert result.metadata.get("reason") == "empty_input"

    # Test with whitespace-only string
    result = validator.validate("   \n   ")
    assert result.passed is True
    assert result.metadata.get("reason") == "empty_input"

    # Test with non-empty string that fails validation
    result = validator.validate("short")
    assert result.passed is False
    assert "too short" in result.message.lower()

    # Test with non-empty string that passes validation
    result = validator.validate("This is a string with more than 10 characters")
    assert result.passed is True


def test_style_validator_empty_text():
    """Test that StyleValidator handles empty text correctly."""
    config = StyleConfig(capitalization=CapitalizationStyle.SENTENCE_CASE)
    validator = DefaultStyleValidator(config)

    # Test with empty string
    result = validator.validate("")
    assert result.passed is True
    assert result.metadata.get("reason") == "empty_input"

    # Test with whitespace-only string
    result = validator.validate("   \n   ")
    assert result.passed is True
    assert result.metadata.get("reason") == "empty_input"

    # Test with non-empty string that fails validation
    result = validator.validate("lowercase sentence")
    assert result.passed is False
    assert "sentence case" in result.message.lower()

    # Test with non-empty string that passes validation
    result = validator.validate("Sentence case text")
    assert result.passed is True
