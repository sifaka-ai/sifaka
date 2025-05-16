"""
Tests for the LengthValidator.
"""

import pytest
from sifaka.validators import LengthValidator
from sifaka.types import ValidationResult


def test_max_length():
    """Test LengthValidator with max_chars constraint."""
    validator = LengthValidator(max_chars=10)

    # Test text within the limit
    result = validator.validate("Short text")
    assert result.passed is True
    assert result.score > 0.0
    assert len(result.issues) == 0

    # Test text exceeding the limit
    result = validator.validate("This text is too long for the validator")
    assert result.passed is False
    assert result.score == 0.0
    assert len(result.issues) == 1
    assert "exceeds maximum length" in result.issues[0]


def test_min_length():
    """Test LengthValidator with min_chars constraint."""
    validator = LengthValidator(min_chars=10)

    # Test text exceeding the minimum
    result = validator.validate("This text is long enough")
    assert result.passed is True
    assert result.score == 1.0
    assert len(result.issues) == 0

    # Test text below the minimum
    result = validator.validate("Too short")
    assert result.passed is False
    assert result.score == 0.0
    assert len(result.issues) == 1
    assert "shorter than minimum length" in result.issues[0]


def test_min_and_max_length():
    """Test LengthValidator with both min_chars and max_chars constraints."""
    validator = LengthValidator(min_chars=5, max_chars=15)

    # Test text within the range
    result = validator.validate("Good length")
    assert result.passed is True
    assert result.score > 0.0
    assert len(result.issues) == 0

    # Test text below the minimum
    result = validator.validate("Hi")
    assert result.passed is False
    assert "shorter than minimum length" in result.issues[0]

    # Test text exceeding the maximum
    result = validator.validate("This text is way too long for the validator")
    assert result.passed is False
    assert "exceeds maximum length" in result.issues[0]


def test_no_constraints():
    """Test LengthValidator with no constraints."""
    validator = LengthValidator()

    # Any text should pass with no constraints
    result = validator.validate("Any text should pass")
    assert result.passed is True
    assert result.score == 1.0
    assert len(result.issues) == 0


def test_metadata():
    """Test metadata in ValidationResult."""
    validator = LengthValidator(max_chars=10)
    text = "Test text"
    result = validator.validate(text)

    assert "length" in result.metadata
    assert result.metadata["length"] == len(text)
