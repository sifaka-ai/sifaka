"""Tests for format rules."""

import pytest
import json
from typing import Dict, Any, List, Set, Protocol, runtime_checkable, Final
from dataclasses import dataclass, field

from sifaka.rules.format import (
    FormatRule,
    FormatConfig,
    FormatValidator,
    DefaultFormatValidator,
    create_format_rule,
)
from sifaka.rules.base import RuleResult


@pytest.fixture
def format_config() -> FormatConfig:
    """Create a test format configuration."""
    return FormatConfig(
        required_format="markdown",
        markdown_elements={"headers", "lists", "code_blocks"},
        json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        },
        cache_size=10,
        priority=2,
        cost=1.5,
    )


@pytest.fixture
def format_validator(format_config: FormatConfig) -> FormatValidator:
    """Create a test format validator."""
    return DefaultFormatValidator(format_config)


@pytest.fixture
def format_rule(
    format_validator: FormatValidator,
) -> FormatRule:
    """Create a test format rule."""
    return FormatRule(
        name="Test Format Rule",
        description="Test format validation",
        validator=format_validator,
    )


def test_format_config_validation():
    """Test format configuration validation."""
    # Test valid configuration
    config = FormatConfig(
        required_format="markdown",
        markdown_elements={"headers", "lists"},
    )
    assert config.required_format == "markdown"
    assert "headers" in config.markdown_elements

    # Test invalid configurations
    with pytest.raises(ValueError, match="required_format must be one of"):
        FormatConfig(required_format="invalid")

    with pytest.raises(ValueError, match="markdown_elements must be a set"):
        FormatConfig(markdown_elements=["invalid"])  # type: ignore


def test_markdown_validation(format_rule: FormatRule):
    """Test markdown format validation."""
    # Test valid markdown
    text = """
# Header

- List item 1
- List item 2

```python
def hello():
    print("Hello, world!")
```
"""
    result = format_rule.validate(text)
    assert result.passed
    assert "headers" in result.metadata["found_elements"]
    assert "lists" in result.metadata["found_elements"]
    assert "code_blocks" in result.metadata["found_elements"]

    # Test missing required elements
    text = "Just plain text without any markdown elements."
    result = format_rule.validate(text)
    assert not result.passed
    assert "Missing required markdown elements" in result.message


def test_json_validation(format_rule: FormatRule):
    """Test JSON format validation."""
    # Configure rule for JSON validation
    rule = create_format_rule(
        name="JSON Format Rule",
        description="Test JSON validation",
        config=FormatConfig(
            required_format="json",
            json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name"],
            },
        ),
    )

    # Test valid JSON
    text = '{"name": "John", "age": 30}'
    result = rule.validate(text)
    assert result.passed
    assert result.metadata["parsed_json"] == {"name": "John", "age": 30}

    # Test invalid JSON format
    text = '{"name": "John", age: 30}'  # Missing quotes around age
    result = rule.validate(text)
    assert not result.passed
    assert "Invalid JSON format" in result.message

    # Test JSON schema validation
    text = '{"age": 30}'  # Missing required name field
    result = rule.validate(text)
    assert not result.passed
    assert "Schema validation failed" in result.message


def test_plain_text_validation(format_rule: FormatRule):
    """Test plain text format validation."""
    # Configure rule for plain text validation
    rule = create_format_rule(
        name="Plain Text Rule",
        description="Test plain text validation",
        config=FormatConfig(
            required_format="plain_text",
            min_length=10,
            max_length=100,
        ),
    )

    # Test valid plain text
    text = "This is a valid plain text message."
    result = rule.validate(text)
    assert result.passed

    # Test too short text
    text = "Too short"
    result = rule.validate(text)
    assert not result.passed
    assert "Text length below minimum" in result.message

    # Test too long text
    text = "x" * 101
    result = rule.validate(text)
    assert not result.passed
    assert "Text length exceeds maximum" in result.message


def test_factory_function():
    """Test factory function for creating format rules."""
    # Test rule creation with default config
    rule = create_format_rule(
        name="Test Rule",
        description="Test validation",
    )
    assert rule.name == "Test Rule"
    assert isinstance(rule.validator, DefaultFormatValidator)

    # Test rule creation with custom config
    rule = create_format_rule(
        name="Custom Rule",
        description="Custom validation",
        config=FormatConfig(required_format="json"),
    )
    assert rule.validator.config.required_format == "json"


def test_edge_cases():
    """Test edge cases and error handling."""
    rule = create_format_rule(
        name="Test Rule",
        description="Test validation",
    )

    # Test empty text
    result = rule.validate("")
    assert not result.passed
    assert "Empty text" in result.message

    # Test invalid input type
    with pytest.raises(ValueError, match="Text must be a string"):
        rule.validate(123)  # type: ignore

    # Test None input
    with pytest.raises(ValueError, match="Text must be a string"):
        rule.validate(None)  # type: ignore


def test_consistent_results():
    """Test that validation results are consistent."""
    rule = create_format_rule(
        name="Test Rule",
        description="Test validation",
        config=FormatConfig(required_format="markdown"),
    )
    text = """
# Header

- List item
"""
    # Multiple validations should yield the same result
    result1 = rule.validate(text)
    result2 = rule.validate(text)
    assert result1.passed == result2.passed
    assert result1.message == result2.message
    assert result1.metadata == result2.metadata
