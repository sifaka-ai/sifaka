"""
Tests for empty text handling in validators.
"""

from sifaka.rules.formatting.length import create_length_validator
from sifaka.rules.formatting.style import create_style_validator, CapitalizationStyle


def test_base_validator_empty_text():
    """Test that BaseValidator handles empty text correctly."""
    # Test with LengthValidator using factory function
    validator = create_length_validator(min_chars=10, max_chars=100)

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
    # Test with StyleValidator using factory function
    validator = create_style_validator(capitalization=CapitalizationStyle.SENTENCE_CASE)

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
