"""Tests for the FormatRule."""

import pytest
from typing import Dict, Any

from sifaka.rules.format import FormatRule
from sifaka.rules.base import RuleResult


class TestFormatRule(FormatRule):
    """Test implementation of FormatRule."""

    def _validate_impl(self, output: str) -> RuleResult:
        """Implement validation logic."""
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        try:
            if self.required_format == "markdown":
                passed = self._validate_markdown(output)
            elif self.required_format == "json":
                passed = self._validate_json(output)
            else:  # plain_text
                passed = bool(output.strip())  # Any non-empty string is valid plain text

            message = f"Output {'is' if passed else 'is not'} valid {self.required_format}"

            return RuleResult(
                passed=passed,
                message=message,
                metadata={"format": self.required_format, "output_length": len(output)},
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during format validation: {str(e)}",
                metadata={"error": str(e), "format": self.required_format},
            )


@pytest.fixture
def rule():
    """Create a TestFormatRule instance."""
    return TestFormatRule(
        name="test_format", description="Test format rule", config={"required_format": "plain_text"}
    )


def test_initialization():
    """Test rule initialization with different parameters."""
    # Test default initialization (plain_text)
    rule = TestFormatRule(name="test", description="test", config={"required_format": "plain_text"})
    assert rule.name == "test"
    assert rule.required_format == "plain_text"

    # Test markdown initialization
    rule = TestFormatRule(name="test", description="test", config={"required_format": "markdown"})
    assert rule.required_format == "markdown"

    # Test JSON initialization
    rule = TestFormatRule(name="test", description="test", config={"required_format": "json"})
    assert rule.required_format == "json"


def test_initialization_validation():
    """Test initialization with invalid parameters."""
    # Test invalid format type
    with pytest.raises(ValueError):
        TestFormatRule(
            name="test", description="test", config={"required_format": "invalid_format"}
        )


def test_plain_text_validation():
    """Test validation of plain text format."""
    rule = TestFormatRule(name="test", description="test", config={"required_format": "plain_text"})

    # Test valid plain text
    result = rule.validate("This is a valid plain text.")
    assert isinstance(result, RuleResult)
    assert result.passed
    assert result.metadata["format"] == "plain_text"
    assert result.metadata["output_length"] == len("This is a valid plain text.")

    # Test empty text
    result = rule.validate("")
    assert not result.passed

    # Test whitespace-only text
    result = rule.validate("   \n\t   ")
    assert not result.passed


def test_markdown_validation():
    """Test validation of markdown format."""
    rule = TestFormatRule(name="test", description="test", config={"required_format": "markdown"})

    # Test valid markdown with various elements
    markdown_samples = [
        "# Heading",
        "**Bold text**",
        "_Italic text_",
        "`code block`",
        "> blockquote",
        "- list item",
        "1. numbered item",
        "[link](url)",
        "# Mixed\n- List\n`code`",
    ]

    for sample in markdown_samples:
        result = rule.validate(sample)
        assert result.passed
        assert result.metadata["format"] == "markdown"

    # Test invalid markdown (plain text)
    result = rule.validate("Just plain text without any markdown elements")
    assert not result.passed


def test_json_validation():
    """Test validation of JSON format."""
    rule = TestFormatRule(name="test", description="test", config={"required_format": "json"})

    # Test valid JSON samples
    json_samples = [
        '{"key": "value"}',
        '{"number": 42}',
        '{"nested": {"key": "value"}}',
        '{"array": [1, 2, 3]}',
        "[]",
        "{}",
        "null",
        '"string"',
        "42",
        "true",
    ]

    for sample in json_samples:
        result = rule.validate(sample)
        assert result.passed
        assert result.metadata["format"] == "json"

    # Test invalid JSON
    invalid_json_samples = [
        '{key: "value"}',  # Missing quotes around key
        '{"key": value}',  # Missing quotes around value
        '{"key": "value",}',  # Trailing comma
        '{"unclosed": "string}',  # Unclosed string
        "{",  # Incomplete object
        "[1, 2,]",  # Trailing comma in array
    ]

    for sample in invalid_json_samples:
        result = rule.validate(sample)
        assert not result.passed


def test_edge_cases():
    """Test handling of edge cases for each format."""
    formats = ["plain_text", "markdown", "json"]

    for format_type in formats:
        rule = TestFormatRule(
            name="test", description="test", config={"required_format": format_type}
        )

        edge_cases = {
            "empty": "",
            "whitespace": "   \n\t   ",
            "special_chars": "!@#$%^&*()",
            "unicode": "Hello 世界",
            "newlines": "Line 1\nLine 2\nLine 3",
            "very_long": "x" * 1000,
        }

        for text in edge_cases.values():
            result = rule.validate(text)
            assert isinstance(result, RuleResult)
            assert "format" in result.metadata
            assert "output_length" in result.metadata
            assert result.metadata["format"] == format_type
            assert result.metadata["output_length"] == len(text)


def test_error_handling(rule):
    """Test error handling for invalid inputs."""
    # Test None input
    with pytest.raises(ValueError):
        rule.validate(None)

    # Test non-string input
    with pytest.raises(ValueError):
        rule.validate(123)


def test_metadata(rule):
    """Test metadata in validation results."""
    result = rule.validate("This is a test text")
    assert "format" in result.metadata
    assert "output_length" in result.metadata
    assert isinstance(result.metadata["format"], str)
    assert isinstance(result.metadata["output_length"], int)
    assert result.metadata["format"] == rule.required_format
    assert result.metadata["output_length"] == len("This is a test text")


def test_consistent_results():
    """Test consistency of format validation."""
    formats = ["plain_text", "markdown", "json"]
    test_texts = {
        "plain_text": "This is a test message.",
        "markdown": "# Heading\n- List item",
        "json": '{"key": "value"}',
    }

    for format_type in formats:
        rule = TestFormatRule(
            name="test", description="test", config={"required_format": format_type}
        )

        text = test_texts[format_type]
        # Run validation multiple times
        results = [rule.validate(text) for _ in range(3)]

        # All results should be consistent
        first_result = results[0]
        for result in results[1:]:
            assert result.passed == first_result.passed
            assert result.metadata["format"] == first_result.metadata["format"]
            assert result.metadata["output_length"] == first_result.metadata["output_length"]
