"""Comprehensive unit tests for ContentValidator.

This module tests the ContentValidator implementation:
- Content validation with prohibited and required patterns
- Regex pattern matching and case sensitivity
- Whole word matching functionality
- Error handling and edge cases

Tests cover:
- Basic content validation functionality
- Pattern matching with various configurations
- Performance characteristics
- Mock-based testing without external dependencies
"""


import pytest

from sifaka.core.thought import SifakaThought
from sifaka.validators.base import ValidationResult
from sifaka.validators.content import (
    ContentValidator,
    create_content_validator,
    prohibited_content_validator,
    required_content_validator,
)


class TestContentValidator:
    """Test suite for ContentValidator class."""

    def test_content_validator_creation_minimal(self):
        """Test creating ContentValidator with minimal parameters."""
        validator = ContentValidator(prohibited=["test"])

        assert validator.prohibited == ["test"]
        assert validator.required == []
        assert validator.case_sensitive is False
        assert validator.whole_word is False
        assert validator.regex is False

    def test_content_validator_creation_full(self):
        """Test creating ContentValidator with all parameters."""
        prohibited = ["spam", "scam"]
        required = ["important", "required"]

        validator = ContentValidator(
            prohibited=prohibited,
            required=required,
            case_sensitive=True,
            whole_word=True,
            regex=True,
        )

        assert validator.prohibited == prohibited
        assert validator.required == required
        assert validator.case_sensitive is True
        assert validator.whole_word is True
        assert validator.regex is True

    @pytest.mark.asyncio
    async def test_validate_async_no_violations(self):
        """Test validation with no content violations."""
        validator = ContentValidator(prohibited=["spam", "scam"], required=["important"])

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This is an important message about renewable energy.",
            iteration=1,
            max_iterations=3,
        )

        result = await validator.validate_async(thought)

        assert result.passed is True
        assert "prohibited content" not in result.message.lower()
        assert "required content" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_async_prohibited_content(self):
        """Test validation with prohibited content."""
        validator = ContentValidator(prohibited=["spam", "scam"])

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This is a spam message trying to scam people.",
            iteration=1,
            max_iterations=3,
        )

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert "prohibited content" in result.message.lower() or "spam" in result.message.lower()
        assert "spam" in result.message or "spam" in str(result.issues)
        assert "scam" in result.message or "scam" in str(result.issues)

    @pytest.mark.asyncio
    async def test_validate_async_missing_required_content(self):
        """Test validation with missing required content."""
        validator = ContentValidator(required=["important", "required"])

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This is just a regular message.",
            iteration=1,
            max_iterations=3,
        )

        result = await validator.validate_async(thought)

        assert result.passed is False
        assert "required content" in result.message.lower() or "missing" in result.message.lower()
        assert "important" in result.message or "important" in str(result.issues)
        assert "required" in result.message or "required" in str(result.issues)

    @pytest.mark.asyncio
    async def test_validate_async_case_sensitive(self):
        """Test case-sensitive validation."""
        validator = ContentValidator(prohibited=["SPAM"], case_sensitive=True)

        # Should pass with lowercase
        thought1 = SifakaThought(
            prompt="Test prompt",
            final_text="This message contains spam but not SPAM.",
            iteration=1,
            max_iterations=3,
        )

        result1 = await validator.validate_async(thought1)
        assert result1.passed is True

        # Should fail with uppercase
        thought2 = SifakaThought(
            prompt="Test prompt",
            final_text="This message contains SPAM.",
            iteration=1,
            max_iterations=3,
        )

        result2 = await validator.validate_async(thought2)
        assert result2.passed is False

    @pytest.mark.asyncio
    async def test_validate_async_case_insensitive(self):
        """Test case-insensitive validation."""
        validator = ContentValidator(prohibited=["spam"], case_sensitive=False)

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This message contains SPAM.",
            iteration=1,
            max_iterations=3,
        )

        result = await validator.validate_async(thought)
        assert result.passed is False
        assert "spam" in result.message.lower() or "spam" in str(result.issues).lower()

    @pytest.mark.asyncio
    async def test_validate_async_whole_word_matching(self):
        """Test whole word matching."""
        validator = ContentValidator(prohibited=["cat"], whole_word=True)

        # Should pass - "cat" is part of "category"
        thought1 = SifakaThought(
            prompt="Test prompt",
            final_text="This message is about category classification.",
            iteration=1,
            max_iterations=3,
        )

        result1 = await validator.validate_async(thought1)
        assert result1.passed is True

        # Should fail - "cat" is a whole word
        thought2 = SifakaThought(
            prompt="Test prompt", final_text="I have a cat as a pet.", iteration=1, max_iterations=3
        )

        result2 = await validator.validate_async(thought2)
        assert result2.passed is False

    @pytest.mark.asyncio
    async def test_validate_async_regex_patterns(self):
        """Test regex pattern matching."""
        validator = ContentValidator(
            prohibited=[r"\b\d{3}-\d{2}-\d{4}\b"], regex=True  # SSN pattern
        )

        # Should fail - contains SSN pattern
        thought1 = SifakaThought(
            prompt="Test prompt", final_text="My SSN is 123-45-6789.", iteration=1, max_iterations=3
        )

        result1 = await validator.validate_async(thought1)
        assert result1.passed is False

        # Should pass - no SSN pattern
        thought2 = SifakaThought(
            prompt="Test prompt",
            final_text="My phone is 123-456-7890.",
            iteration=1,
            max_iterations=3,
        )

        result2 = await validator.validate_async(thought2)
        assert result2.passed is True

    @pytest.mark.asyncio
    async def test_validate_async_empty_text(self):
        """Test validation with empty text."""
        validator = ContentValidator(required=["important"])

        thought = SifakaThought(prompt="Test prompt", final_text="", iteration=1, max_iterations=3)

        result = await validator.validate_async(thought)
        assert result.passed is False
        assert "required content" in result.message.lower() or "missing" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_async_timing(self):
        """Test that validation includes timing information."""
        validator = ContentValidator(prohibited=["spam"])

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This is a clean message.",
            iteration=1,
            max_iterations=3,
        )

        result = await validator.validate_async(thought)

        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms >= 0


class TestContentValidatorFactories:
    """Test suite for ContentValidator factory functions."""

    def test_create_content_validator(self):
        """Test create_content_validator factory function."""
        validator = create_content_validator(
            prohibited=["spam"], required=["important"], case_sensitive=True
        )

        assert isinstance(validator, ContentValidator)
        assert validator.prohibited == ["spam"]
        assert validator.required == ["important"]
        assert validator.case_sensitive is True

    def test_prohibited_content_validator(self):
        """Test prohibited_content_validator factory function."""
        validator = prohibited_content_validator(["spam", "scam"])

        assert isinstance(validator, ContentValidator)
        assert validator.prohibited == ["spam", "scam"]
        assert validator.required == []

    def test_required_content_validator(self):
        """Test required_content_validator factory function."""
        validator = required_content_validator(["important", "required"])

        assert isinstance(validator, ContentValidator)
        assert validator.prohibited == []
        assert validator.required == ["important", "required"]


class TestContentValidatorEdgeCases:
    """Test suite for ContentValidator edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_validate_async_invalid_regex(self):
        """Test validation with invalid regex pattern."""
        validator = ContentValidator(prohibited=["[invalid"], regex=True)  # Invalid regex

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="This is a test message.",
            iteration=1,
            max_iterations=3,
        )

        # Should handle invalid regex gracefully
        result = await validator.validate_async(thought)
        # Implementation should either fix the regex or handle the error
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validate_async_none_text(self):
        """Test validation with None text."""
        validator = ContentValidator(required=["important"])

        thought = SifakaThought(
            prompt="Test prompt", final_text=None, iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_validate_async_unicode_content(self):
        """Test validation with unicode content."""
        validator = ContentValidator(
            prohibited=["禁止"],  # Chinese for "prohibited"
            required=["必需"],  # Chinese for "required"
        )

        thought = SifakaThought(
            prompt="Test prompt",
            final_text="这是一个必需的消息。",  # "This is a required message."
            iteration=1,
            max_iterations=3,
        )

        result = await validator.validate_async(thought)
        assert result.passed is True

        # Test with prohibited content
        thought2 = SifakaThought(
            prompt="Test prompt",
            final_text="这是一个禁止的消息。",  # "This is a prohibited message."
            iteration=1,
            max_iterations=3,
        )

        result2 = await validator.validate_async(thought2)
        assert result2.passed is False

    def test_content_validator_repr(self):
        """Test ContentValidator string representation."""
        validator = ContentValidator(prohibited=["spam"], required=["important"])

        repr_str = repr(validator)
        assert "ContentValidator" in repr_str
        assert "prohibited" in repr_str or "spam" in repr_str

    @pytest.mark.asyncio
    async def test_validate_async_large_content(self):
        """Test validation with large content."""
        validator = ContentValidator(prohibited=["spam"], required=["important"])

        # Create large text content
        large_text = "This is an important message. " * 1000 + "No spam here."

        thought = SifakaThought(
            prompt="Test prompt", final_text=large_text, iteration=1, max_iterations=3
        )

        result = await validator.validate_async(thought)
        assert result.passed is True
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_validate_async_multiple_patterns(self):
        """Test validation with multiple prohibited and required patterns."""
        validator = ContentValidator(
            prohibited=["spam", "scam", "phishing", "malware"],
            required=["important", "verified", "secure"],
        )

        # Should pass - has all required, no prohibited
        thought1 = SifakaThought(
            prompt="Test prompt",
            final_text="This important message is verified and secure.",
            iteration=1,
            max_iterations=3,
        )

        result1 = await validator.validate_async(thought1)
        assert result1.passed is True

        # Should fail - missing required content
        thought2 = SifakaThought(
            prompt="Test prompt",
            final_text="This important message is verified.",
            iteration=1,
            max_iterations=3,
        )

        result2 = await validator.validate_async(thought2)
        assert result2.passed is False
        assert "secure" in result2.message or "secure" in str(result2.issues)

    @pytest.mark.asyncio
    async def test_validate_async_complex_regex(self):
        """Test validation with complex regex patterns."""
        validator = ContentValidator(
            prohibited=[
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card pattern
            ],
            regex=True,
        )

        # Should fail - contains email
        thought1 = SifakaThought(
            prompt="Test prompt",
            final_text="Contact me at user@example.com for more info.",
            iteration=1,
            max_iterations=3,
        )

        result1 = await validator.validate_async(thought1)
        assert result1.passed is False

        # Should fail - contains credit card number
        thought2 = SifakaThought(
            prompt="Test prompt",
            final_text="My card number is 1234 5678 9012 3456.",
            iteration=1,
            max_iterations=3,
        )

        result2 = await validator.validate_async(thought2)
        assert result2.passed is False

        # Should pass - no sensitive patterns
        thought3 = SifakaThought(
            prompt="Test prompt",
            final_text="This is a safe message with no sensitive information.",
            iteration=1,
            max_iterations=3,
        )

        result3 = await validator.validate_async(thought3)
        assert result3.passed is True
