"""
Tests for the ContentValidator.
"""

import pytest
from sifaka.validators import ContentValidator
from sifaka.types import ValidationResult


def test_prohibit_single_pattern():
    """Test ContentValidator in prohibit mode with a single pattern."""
    validator = ContentValidator(patterns=r"bad", mode="prohibit")

    # Test text without the prohibited pattern
    result = validator.validate("This is good text")
    assert result.passed is True
    assert result.score == 1.0

    # Test text with the prohibited pattern
    result = validator.validate("This is bad text")
    assert result.passed is False
    assert result.score < 1.0
    assert len(result.issues) > 0
    assert "prohibited pattern" in result.issues[0]


def test_prohibit_multiple_patterns():
    """Test ContentValidator in prohibit mode with multiple patterns."""
    validator = ContentValidator(
        patterns=["bad", "terrible", "awful"],
        mode="prohibit",
        match_all=False,  # Default - any match fails
    )

    # Test text without any prohibited patterns
    result = validator.validate("This is good text")
    assert result.passed is True

    # Test text with one prohibited pattern
    result = validator.validate("This is terrible text")
    assert result.passed is False
    assert len(result.issues) == 1

    # Test text with multiple prohibited patterns
    result = validator.validate("This is bad and awful text")
    assert result.passed is False
    assert len(result.issues) == 2


def test_prohibit_match_all():
    """Test ContentValidator in prohibit mode with match_all=True."""
    validator = ContentValidator(
        patterns=["bad", "terrible", "awful"],
        mode="prohibit",
        match_all=True,  # All patterns must match to fail
    )

    # Test text with only one prohibited pattern (should pass)
    result = validator.validate("This is terrible text")
    assert result.passed is True

    # Test text with all prohibited patterns (should fail)
    result = validator.validate("This is bad, terrible, and awful text")
    assert result.passed is False
    assert len(result.issues) == 3


def test_require_single_pattern():
    """Test ContentValidator in require mode with a single pattern."""
    validator = ContentValidator(patterns=r"good", mode="require")

    # Test text with the required pattern
    result = validator.validate("This is good text")
    assert result.passed is True
    assert result.score == 1.0

    # Test text without the required pattern
    result = validator.validate("This is text")
    assert result.passed is False
    assert result.score < 1.0
    assert len(result.issues) > 0
    assert "required pattern" in result.issues[0]


def test_require_multiple_patterns():
    """Test ContentValidator in require mode with multiple patterns."""
    validator = ContentValidator(
        patterns=["good", "excellent", "great"],
        mode="require",
        match_all=False,  # Default - any match passes
    )

    # Test text with one required pattern
    result = validator.validate("This is good text")
    assert result.passed is True

    # Test text with multiple required patterns
    result = validator.validate("This is excellent and great text")
    assert result.passed is True

    # Test text without any required patterns
    result = validator.validate("This is some text")
    assert result.passed is False
    assert len(result.issues) == 3


def test_require_match_all():
    """Test ContentValidator in require mode with match_all=True."""
    validator = ContentValidator(
        patterns=["good", "excellent", "great"],
        mode="require",
        match_all=True,  # All patterns must match to pass
    )

    # Test text with only one required pattern (should fail)
    result = validator.validate("This is good text")
    assert result.passed is False
    assert len(result.issues) == 2

    # Test text with all required patterns (should pass)
    result = validator.validate("This text is good, excellent, and great")
    assert result.passed is True


def test_case_sensitivity():
    """Test case sensitivity of ContentValidator."""
    # Case-insensitive (default)
    validator1 = ContentValidator(patterns=r"test")
    assert validator1.validate("This is a TEST").passed is False

    # Case-sensitive
    validator2 = ContentValidator(patterns=r"test", case_sensitive=True)
    assert (
        validator2.validate("This is a TEST").passed is True
    )  # Should pass because "TEST" doesn't match "test"
    assert (
        validator2.validate("This is a test").passed is False
    )  # Should fail because "test" matches "test"


def test_metadata():
    """Test metadata in ValidationResult."""
    validator = ContentValidator(patterns=["good", "excellent"], mode="require")
    text = "This is good but not excellent"
    result = validator.validate(text)

    assert "matches" in result.metadata
    assert "non_matches" in result.metadata
    assert "patterns" in result.metadata
    assert "mode" in result.metadata

    # Verify the matches contains the correct index
    assert 0 in result.metadata["matches"]  # "good" matched

    # Check that non_matches exists but don't assert specific contents
    # as implementation might differ in how it handles non-matches
