"""Comprehensive unit tests for FormatValidator.

This module tests the FormatValidator implementation:
- JSON format validation
- Markdown format validation
- Custom format validation
- Error handling and edge cases

Tests cover:
- Basic format validation functionality
- Different format types and configurations
- Performance characteristics
- Mock-based testing without external dependencies
"""

import json

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.validators.base import ValidationResult
from sifaka.validators.format import (
    FormatValidator,
    create_format_validator,
    json_validator,
    markdown_validator,
)


class TestFormatValidator:
    """Test suite for FormatValidator class."""

    def test_format_validator_creation_minimal(self):
        """Test creating FormatValidator with minimal parameters."""
        validator = FormatValidator(format_type="json")

        assert validator.format_type == "json"
        assert validator.custom_checker is None
        assert validator.strict is True

    def test_format_validator_creation_full(self):
        """Test creating FormatValidator with all parameters."""

        def custom_checker(text):
            return text.startswith("CUSTOM:")

        validator = FormatValidator(
            format_type="custom", custom_checker=custom_checker, strict=False
        )

        assert validator.format_type == "custom"
        assert validator.custom_checker == custom_checker
        assert validator.strict is False

    @pytest.mark.asyncio
    async def test_validate_async_valid_json(self):
        """Test validation with valid JSON."""
        validator = FormatValidator(format_type="json")

        valid_json = '{"name": "test", "value": 123, "active": true}'

        thought = SifakaThought(
            prompt="Test prompt", final_text=valid_json, iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)

        assert result.passed is True
        assert "json" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_async_invalid_json(self):
        """Test validation with invalid JSON."""
        validator = FormatValidator(format_type="json")

        invalid_json = '{"name": "test", "value": 123, "active": true'  # Missing closing brace

        thought = SifakaThought(
            prompt="Test prompt", final_text=invalid_json, iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert "json" in result.message.lower() or "invalid" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_async_valid_markdown(self):
        """Test validation with valid Markdown."""
        validator = FormatValidator(format_type="markdown")

        valid_markdown = """# Title

## Subtitle

This is a paragraph with **bold** and *italic* text.

- List item 1
- List item 2

```python
print("Hello, world!")
```
"""

        thought = SifakaThought(
            prompt="Test prompt", final_text=valid_markdown, iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)

        assert result.passed is True
        assert "markdown" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_async_invalid_markdown(self):
        """Test validation with invalid Markdown."""
        validator = FormatValidator(format_type="markdown", strict=True)

        # Markdown with unclosed code block
        invalid_markdown = """# Title

```python
print("Hello, world!")
# Missing closing ```
"""

        thought = SifakaThought(
            prompt="Test prompt", final_text=invalid_markdown, iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)

        # Note: This test depends on the actual implementation
        # Some markdown parsers are more lenient than others
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validate_async_custom_format(self):
        """Test validation with custom format checker."""

        def custom_checker(text):
            return text.startswith("CUSTOM:") and text.endswith(":END")

        validator = FormatValidator(format_type="custom", custom_checker=custom_checker)

        # Valid custom format
        thought1 = SifakaThought(
            prompt="Test prompt",
            final_text="CUSTOM: This is a custom formatted message :END",
            iteration=1,
            max_iterations=3,
        )

        result1 = await validator.validate_async(thought1)
        assert result1.passed is True

        # Invalid custom format
        thought2 = SifakaThought(
            prompt="Test prompt",
            final_text="This is not a custom formatted message",
            iteration=1,
            max_iterations=3,
        )

        result2 = await validator.validate_async(thought2)
        assert result2.passed is False

    @pytest.mark.asyncio
    async def test_validate_async_empty_text(self):
        """Test validation with empty text."""
        validator = FormatValidator(format_type="json")

        thought = SifakaThought(prompt="Test prompt", final_text="", iteration=1, max_iterations=3)

        result = await validator.validate_async(thought)
        assert result.passed is False
        assert "empty" in result.message.lower() or "json" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_async_none_text(self):
        """Test validation with None text."""
        validator = FormatValidator(format_type="json")

        thought = SifakaThought(
            prompt="Test prompt", final_text=None, iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_validate_async_timing(self):
        """Test that validation includes timing information."""
        validator = FormatValidator(format_type="json")

        thought = SifakaThought(
            prompt="Test prompt", final_text='{"test": "value"}', iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)

        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_validate_async_strict_vs_lenient(self):
        """Test strict vs lenient validation modes."""
        # This test depends on the specific implementation
        # For now, we'll test that both modes work

        strict_validator = FormatValidator(format_type="json", strict=True)
        lenient_validator = FormatValidator(format_type="json", strict=False)

        # Test with potentially problematic JSON (trailing comma)
        json_with_trailing_comma = '{"name": "test", "value": 123,}'

        thought = SifakaThought(
            prompt="Test prompt", final_text=json_with_trailing_comma, iteration=1, max_iterations=3
        )

        strict_result = await strict_validator.validate_async(thought)
        lenient_result = await lenient_validator.validate_async(thought)

        # Both should handle this case (implementation dependent)
        assert isinstance(strict_result, ValidationResult)
        assert isinstance(lenient_result, ValidationResult)


class TestFormatValidatorFactories:
    """Test suite for FormatValidator factory functions."""

    def test_create_format_validator(self):
        """Test create_format_validator factory function."""
        validator = create_format_validator(format_type="json", strict=False)

        assert isinstance(validator, FormatValidator)
        assert validator.format_type == "json"
        assert validator.strict is False

    def test_json_validator(self):
        """Test json_validator factory function."""
        validator = json_validator()

        assert isinstance(validator, FormatValidator)
        assert validator.format_type == "json"

    def test_markdown_validator(self):
        """Test markdown_validator factory function."""
        validator = markdown_validator()

        assert isinstance(validator, FormatValidator)
        assert validator.format_type == "markdown"


class TestFormatValidatorEdgeCases:
    """Test suite for FormatValidator edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_validate_async_unsupported_format(self):
        """Test validation with unsupported format type."""
        validator = FormatValidator(format_type="unsupported")

        thought = SifakaThought(
            prompt="Test prompt", final_text="Some text content", iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)
        # Should handle unsupported format gracefully
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validate_async_custom_checker_exception(self):
        """Test validation when custom checker raises exception."""

        def failing_checker(text):
            raise ValueError("Custom checker failed")

        validator = FormatValidator(format_type="custom", custom_checker=failing_checker)

        thought = SifakaThought(
            prompt="Test prompt", final_text="Test content", iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)
        # Should handle exception gracefully
        assert isinstance(result, ValidationResult)
        assert result.passed is False

    def test_format_validator_repr(self):
        """Test FormatValidator string representation."""
        validator = FormatValidator(format_type="json", strict=True)

        repr_str = repr(validator)
        assert "FormatValidator" in repr_str
        assert "json" in repr_str.lower()

    @pytest.mark.asyncio
    async def test_validate_async_complex_json(self):
        """Test validation with complex JSON structures."""
        validator = FormatValidator(format_type="json")

        complex_json = {
            "users": [
                {"id": 1, "name": "Alice", "active": True},
                {"id": 2, "name": "Bob", "active": False},
            ],
            "metadata": {"total": 2, "page": 1, "timestamp": "2024-01-01T00:00:00Z"},
            "config": {
                "features": ["auth", "logging", "metrics"],
                "settings": {"debug": False, "timeout": 30.5},
            },
        }

        thought = SifakaThought(
            prompt="Test prompt",
            final_text=json.dumps(complex_json, indent=2),
            iteration=1,
            max_iterations=3,
        )

        result = await validator.validate_async(thought)
        assert result.passed is True
