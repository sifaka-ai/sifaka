"""
Tests for the format validation rules.

This module tests the format validation rules in the sifaka.rules.formatting.format package.
"""

import pytest

from sifaka.rules.formatting.format import create_format_rule
from sifaka.rules.formatting.format.markdown import create_markdown_rule
from sifaka.rules.formatting.format.json import create_json_rule
from sifaka.rules.formatting.format.plain_text import create_plain_text_rule


def test_markdown_rule_initialization():
    """Test that a markdown rule can be initialized."""
    rule = create_markdown_rule(
        required_elements=["#", "*", "`"],
        min_elements=2,
        name="test_markdown_rule",
        description="Test markdown rule",
    )

    assert rule.name == "test_markdown_rule"
    assert rule.description == "Test markdown rule"
    assert rule.validator.config.required_elements == ["#", "*", "`"]
    assert rule.validator.config.min_elements == 2


def test_markdown_rule_validate_success():
    """Test that a markdown rule validates correctly."""
    rule = create_markdown_rule(
        required_elements=["#", "*", "`"],
        min_elements=2,
        name="markdown_rule",
    )

    result = rule.validate("# Heading\n\n* List item")

    assert result.passed
    assert "Found 2 markdown elements" in result.message
    assert len(result.issues) == 0
    assert result.score > 0.9


def test_markdown_rule_validate_failure():
    """Test that a markdown rule fails validation correctly."""
    rule = create_markdown_rule(
        required_elements=["#", "*", "`"],
        min_elements=2,
        name="markdown_rule",
    )

    result = rule.validate("Plain text without markdown")

    assert not result.passed
    assert "Insufficient markdown elements" in result.message
    assert len(result.issues) > 0
    assert result.score < 0.5


def test_json_rule_initialization():
    """Test that a JSON rule can be initialized."""
    rule = create_json_rule(
        strict=True,
        allow_empty=False,
        name="test_json_rule",
        description="Test JSON rule",
    )

    assert rule.name == "test_json_rule"
    assert rule.description == "Test JSON rule"
    assert rule.validator.config.strict
    assert not rule.validator.config.allow_empty


def test_json_rule_validate_success():
    """Test that a JSON rule validates correctly."""
    rule = create_json_rule(
        strict=True,
        allow_empty=False,
        name="json_rule",
    )

    result = rule.validate('{"key": "value"}')

    assert result.passed
    assert "Valid JSON format" in result.message
    assert len(result.issues) == 0
    assert result.score > 0.9


def test_json_rule_validate_failure():
    """Test that a JSON rule fails validation correctly."""
    rule = create_json_rule(
        strict=True,
        allow_empty=False,
        name="json_rule",
    )

    result = rule.validate('{"key": value}')  # Missing quotes around value

    assert not result.passed
    assert "Invalid JSON format" in result.message
    assert len(result.issues) > 0
    assert result.score < 0.5


def test_plain_text_rule_initialization():
    """Test that a plain text rule can be initialized."""
    rule = create_plain_text_rule(
        min_length=10,
        max_length=100,
        allow_empty=False,
        name="test_plain_text_rule",
        description="Test plain text rule",
    )

    assert rule.name == "test_plain_text_rule"
    assert rule.description == "Test plain text rule"
    assert rule.validator.config.min_length == 10
    assert rule.validator.config.max_length == 100
    assert not rule.validator.config.allow_empty


def test_plain_text_rule_validate_success():
    """Test that a plain text rule validates correctly."""
    rule = create_plain_text_rule(
        min_length=10,
        max_length=100,
        name="plain_text_rule",
    )

    result = rule.validate("This is a test string that is long enough.")

    assert result.passed
    assert "within allowed range" in result.message.lower() or "meets" in result.message.lower()
    assert len(result.issues) == 0
    assert result.score > 0.9


def test_plain_text_rule_validate_too_short():
    """Test that a plain text rule fails validation for text that is too short."""
    rule = create_plain_text_rule(
        min_length=10,
        max_length=100,
        name="plain_text_rule",
    )

    result = rule.validate("Too short")

    assert not result.passed
    assert "less than minimum" in result.message.lower()
    assert len(result.issues) > 0
    assert result.score <= 0.9


def test_plain_text_rule_validate_too_long():
    """Test that a plain text rule fails validation for text that is too long."""
    rule = create_plain_text_rule(
        min_length=10,
        max_length=20,
        name="plain_text_rule",
    )

    result = rule.validate("This text is too long for the maximum length specified.")

    assert not result.passed
    assert "exceeds maximum" in result.message.lower()
    assert len(result.issues) > 0
    assert result.score <= 0.9


def test_format_rule_markdown():
    """Test that a format rule with markdown format works correctly."""
    rule = create_format_rule(
        required_format="markdown",
        markdown_elements={"#", "*", "`"},
        min_elements=2,
        name="format_rule",
    )

    result = rule.validate("# Heading\n\n* List item")

    assert result.passed
    assert "Found 2 markdown elements" in result.message
    assert len(result.issues) == 0
    assert result.score > 0.9


def test_format_rule_json():
    """Test that a format rule with JSON format works correctly."""
    rule = create_format_rule(
        required_format="json",
        strict=True,
        allow_empty=False,
        name="format_rule",
    )

    result = rule.validate('{"key": "value"}')

    assert result.passed
    assert "Valid JSON format" in result.message
    assert len(result.issues) == 0
    assert result.score > 0.9


def test_format_rule_plain_text():
    """Test that a format rule with plain text format works correctly."""
    rule = create_format_rule(
        required_format="plain_text",
        min_length=10,
        max_length=100,
        name="format_rule",
    )

    result = rule.validate("This is a test string that is long enough.")

    assert result.passed
    assert "meets" in result.message.lower()
    assert len(result.issues) == 0
    assert result.score > 0.9


def test_format_rule_invalid_format():
    """Test that creating a format rule with an invalid format raises an error."""
    with pytest.raises(ValueError):
        create_format_rule(
            required_format="invalid_format",
            name="format_rule",
        )
