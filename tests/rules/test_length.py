"""Tests for the LengthRule."""

import pytest
from typing import Dict, Any

from sifaka.rules.length import LengthRule
from sifaka.rules.base import RuleResult


class TestLengthRule(LengthRule):
    """Test implementation of LengthRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        length = len(output)
        passed = True
        message = f"Length {length} is valid"

        if self.min_length is not None and length < self.min_length:
            passed = False
            message = f"Length {length} is below minimum {self.min_length}"
        elif self.max_length is not None and length > self.max_length:
            passed = False
            message = f"Length {length} exceeds maximum {self.max_length}"

        return RuleResult(passed=passed, message=message, metadata={"length": length})


@pytest.fixture
def rule():
    """Create a TestLengthRule instance."""
    return TestLengthRule(
        name="test_length",
        description="Test length rule",
        config={"min_length": 10, "max_length": 100, "unit": "characters"},
    )


def test_initialization():
    """Test rule initialization with different parameters."""
    # Test default initialization
    rule = TestLengthRule(
        name="test",
        description="test",
        config={"min_length": 10, "max_length": 100, "unit": "characters"},
    )
    assert rule.name == "test"
    assert rule.min_length == 10
    assert rule.max_length == 100

    # Test with words unit
    rule = TestLengthRule(
        name="test", description="test", config={"min_length": 5, "max_length": 50, "unit": "words"}
    )
    assert rule.min_length == 5
    assert rule.max_length == 50


def test_initialization_validation():
    """Test initialization with invalid parameters."""
    # Test invalid min_length
    with pytest.raises(ValueError):
        TestLengthRule(
            name="test",
            description="test",
            config={"min_length": -1, "max_length": 100, "unit": "characters"},
        )

    # Test invalid max_length
    with pytest.raises(ValueError):
        TestLengthRule(
            name="test",
            description="test",
            config={"min_length": 10, "max_length": -1, "unit": "characters"},
        )

    # Test min > max
    with pytest.raises(ValueError):
        TestLengthRule(
            name="test",
            description="test",
            config={"min_length": 100, "max_length": 10, "unit": "characters"},
        )


def test_character_length_validation(rule):
    """Test validation with character length."""
    # Test valid length
    result = rule.validate("This is a valid length text.")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert "length" in result.metadata

    # Test too short
    result = rule.validate("Too short")
    assert not result.passed
    assert "does not meet minimum length" in result.message.lower()
    assert result.metadata["length"] == 9
    assert result.metadata["min_length"] == 10

    # Test too long
    result = rule.validate("x" * 101)
    assert not result.passed
    assert "does not meet maximum length" in result.message.lower()
    assert result.metadata["length"] == 101
    assert result.metadata["max_length"] == 100


def test_word_length_validation():
    """Test validation with word length."""
    rule = TestLengthRule(
        name="test",
        description="test",
        config={"min_length": 3, "max_length": 10, "unit": "words"},
    )

    # Test valid word count
    result = rule.validate("One two three four")
    assert result.passed
    assert result.metadata["length"] == 4

    # Test too few words
    result = rule.validate("Too few")
    assert not result.passed
    assert "does not meet minimum length" in result.message.lower()

    # Test too many words
    result = rule.validate("This has way too many words to be valid for this test")
    assert not result.passed
    assert "does not meet maximum length" in result.message.lower()


def test_edge_cases(rule):
    """Test handling of edge cases."""
    edge_cases = {
        "empty": "",
        "whitespace": "   \n\t   ",
        "special_chars": "!@#$%^&*()",
        "unicode": "Hello 世界",
        "newlines": "Line 1\nLine 2\nLine 3",
    }

    for case_name, text in edge_cases.items():
        result = rule.validate(text)
        assert isinstance(result, RuleResult)

        if case_name in ["empty", "whitespace"]:
            assert not result.passed
            assert "does not meet minimum length" in result.message.lower()
            assert result.metadata["length"] < rule.min_length


def test_error_handling(rule):
    """Test error handling for invalid inputs."""
    # Test None input
    with pytest.raises(ValueError, match="Output cannot be None"):
        rule.validate(None)

    # Test non-string input
    with pytest.raises(ValueError, match="Output must be a string"):
        rule.validate(123)


def test_metadata(rule):
    """Test metadata in validation results."""
    text = "This is a test text"
    result = rule.validate(text)

    assert "length" in result.metadata
    assert isinstance(result.metadata["length"], int)
    assert result.metadata["length"] == len(text)
