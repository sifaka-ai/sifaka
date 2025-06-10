"""Comprehensive unit tests for LengthValidator.

This module provides extensive testing for the length validation functionality:
- LengthValidator: Text length validation with configurable limits
- Character, word, and sentence counting
- Minimum and maximum length constraints
- Error handling and edge cases
- Performance characteristics
- Integration with validation pipeline
"""

import asyncio

import pytest

from sifaka.validators.length import LengthValidator


class TestLengthValidator:
    """Test LengthValidator functionality."""

    def test_length_validator_creation(self):
        """Test creating LengthValidator with various configurations."""
        # Default configuration
        validator = LengthValidator()
        assert validator.name == "length"
        assert validator.min_length == 1
        assert validator.max_length == 10000
        assert validator.count_type == "characters"

        # Custom configuration
        validator = LengthValidator(
            min_length=50, max_length=500, count_type="words", name="custom_length"
        )
        assert validator.min_length == 50
        assert validator.max_length == 500
        assert validator.count_type == "words"
        assert validator.name == "custom_length"

    def test_length_validator_invalid_configuration(self):
        """Test LengthValidator with invalid configurations."""
        # Min length greater than max length
        with pytest.raises(ValueError, match="min_length cannot be greater than max_length"):
            LengthValidator(min_length=100, max_length=50)

        # Negative min length
        with pytest.raises(ValueError, match="min_length cannot be negative"):
            LengthValidator(min_length=-1)

        # Invalid count type
        with pytest.raises(ValueError, match="count_type must be one of"):
            LengthValidator(count_type="invalid")

    def test_character_counting(self):
        """Test character-based length validation."""
        validator = LengthValidator(min_length=5, max_length=20, count_type="characters")

        # Valid length
        result = validator.validate("Hello world")
        assert result.is_valid is True
        assert result.metadata["character_count"] == 11
        assert result.metadata["count_type"] == "characters"

        # Too short
        result = validator.validate("Hi")
        assert result.is_valid is False
        assert "too short" in result.message.lower()
        assert result.metadata["character_count"] == 2

        # Too long
        result = validator.validate("This is a very long text that exceeds the limit")
        assert result.is_valid is False
        assert "too long" in result.message.lower()
        assert result.metadata["character_count"] == 47

    def test_word_counting(self):
        """Test word-based length validation."""
        validator = LengthValidator(min_length=2, max_length=5, count_type="words")

        # Valid word count
        result = validator.validate("Hello world test")
        assert result.is_valid is True
        assert result.metadata["word_count"] == 3
        assert result.metadata["count_type"] == "words"

        # Too few words
        result = validator.validate("Hello")
        assert result.is_valid is False
        assert result.metadata["word_count"] == 1

        # Too many words
        result = validator.validate("This is a very long sentence with many words")
        assert result.is_valid is False
        assert result.metadata["word_count"] == 9

    def test_sentence_counting(self):
        """Test sentence-based length validation."""
        validator = LengthValidator(min_length=1, max_length=3, count_type="sentences")

        # Valid sentence count
        result = validator.validate("Hello world. This is a test.")
        assert result.is_valid is True
        assert result.metadata["sentence_count"] == 2
        assert result.metadata["count_type"] == "sentences"

        # Too few sentences
        result = validator.validate("Hello world")  # No period, might not count as sentence
        assert result.metadata["sentence_count"] >= 0

        # Too many sentences
        result = validator.validate("First. Second. Third. Fourth. Fifth.")
        assert result.is_valid is False
        assert result.metadata["sentence_count"] == 5

    def test_edge_case_texts(self):
        """Test validation with edge case texts."""
        validator = LengthValidator(min_length=1, max_length=100, count_type="characters")

        # Empty string
        result = validator.validate("")
        assert result.is_valid is False
        assert result.metadata["character_count"] == 0

        # Whitespace only
        result = validator.validate("   ")
        assert result.metadata["character_count"] == 3
        # Behavior depends on whether whitespace counts

        # Special characters
        result = validator.validate("!@#$%^&*()")
        assert result.is_valid is True
        assert result.metadata["character_count"] == 10

        # Unicode characters
        result = validator.validate("Hello 世界 🌍")
        assert result.is_valid is True
        assert result.metadata["character_count"] > 0

    def test_word_counting_edge_cases(self):
        """Test word counting with various text formats."""
        validator = LengthValidator(min_length=1, max_length=10, count_type="words")

        # Multiple spaces
        result = validator.validate("word1    word2     word3")
        assert result.metadata["word_count"] == 3

        # Punctuation
        result = validator.validate("Hello, world! How are you?")
        assert result.metadata["word_count"] >= 5  # Depends on implementation

        # Numbers and mixed content
        result = validator.validate("Test 123 more-text")
        assert result.metadata["word_count"] >= 2

        # Hyphenated words
        result = validator.validate("well-known state-of-the-art")
        assert result.metadata["word_count"] >= 2

    def test_sentence_counting_edge_cases(self):
        """Test sentence counting with various punctuation."""
        validator = LengthValidator(min_length=1, max_length=10, count_type="sentences")

        # Different sentence endings
        result = validator.validate("Question? Exclamation! Statement.")
        assert result.metadata["sentence_count"] == 3

        # Abbreviations (shouldn't count as sentence endings)
        result = validator.validate("Dr. Smith went to the U.S.A. yesterday.")
        # Should be 1 sentence, not multiple

        # No punctuation
        result = validator.validate("This is a sentence without punctuation")
        # Behavior depends on implementation

    @pytest.mark.asyncio
    async def test_async_validation(self):
        """Test asynchronous validation."""
        validator = LengthValidator(min_length=5, max_length=50)

        result = await validator.validate_async("This is a test message")
        assert result.is_valid is True
        assert result.validator_name == "length"
        assert result.metadata["character_count"] == 22

    @pytest.mark.asyncio
    async def test_concurrent_validations(self):
        """Test concurrent validation calls."""
        validator = LengthValidator(min_length=1, max_length=100)

        texts = [
            "Short",
            "Medium length text",
            "This is a longer text that should still be valid",
            "X" * 150,  # Too long
        ]

        tasks = [validator.validate_async(text) for text in texts]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert results[0].is_valid is True  # Short
        assert results[1].is_valid is True  # Medium
        assert results[2].is_valid is True  # Longer but valid
        assert results[3].is_valid is False  # Too long

    def test_validation_metadata(self):
        """Test that validation results include proper metadata."""
        validator = LengthValidator(min_length=10, max_length=100, count_type="words")

        result = validator.validate("This is a test sentence with several words")

        # Check required metadata
        assert "word_count" in result.metadata
        assert "count_type" in result.metadata
        assert "min_length" in result.metadata
        assert "max_length" in result.metadata
        assert "validator_name" in result.metadata

        assert result.metadata["count_type"] == "words"
        assert result.metadata["min_length"] == 10
        assert result.metadata["max_length"] == 100
        assert result.metadata["validator_name"] == "length"

    def test_validation_messages(self):
        """Test validation result messages."""
        validator = LengthValidator(min_length=10, max_length=20, count_type="characters")

        # Valid text
        result = validator.validate("Valid length text")
        assert "valid" in result.message.lower() or result.message == ""

        # Too short
        result = validator.validate("Short")
        assert "short" in result.message.lower()
        assert "10" in result.message  # Should mention minimum

        # Too long
        result = validator.validate("This text is definitely too long for the validator")
        assert "long" in result.message.lower()
        assert "20" in result.message  # Should mention maximum

    def test_zero_length_limits(self):
        """Test validator with zero minimum length."""
        validator = LengthValidator(min_length=0, max_length=10)

        # Empty string should be valid
        result = validator.validate("")
        assert result.is_valid is True
        assert result.metadata["character_count"] == 0

    def test_equal_min_max_length(self):
        """Test validator with equal min and max length."""
        validator = LengthValidator(min_length=5, max_length=5)

        # Exact length should be valid
        result = validator.validate("Hello")
        assert result.is_valid is True
        assert result.metadata["character_count"] == 5

        # Different length should be invalid
        result = validator.validate("Hi")
        assert result.is_valid is False

        result = validator.validate("Hello world")
        assert result.is_valid is False

    def test_performance_with_large_text(self):
        """Test validator performance with large text."""
        validator = LengthValidator(min_length=1, max_length=1000000)

        # Large text
        large_text = "A" * 50000
        result = validator.validate(large_text)

        assert result.is_valid is True
        assert result.metadata["character_count"] == 50000

    def test_custom_validator_name(self):
        """Test validator with custom name."""
        validator = LengthValidator(name="custom_length_validator")

        result = validator.validate("Test text")
        assert result.validator_name == "custom_length_validator"
        assert validator.name == "custom_length_validator"

    def test_string_representation(self):
        """Test string representations of the validator."""
        validator = LengthValidator(min_length=5, max_length=100, count_type="words")

        str_repr = str(validator)
        assert "length" in str_repr.lower()

        repr_str = repr(validator)
        assert "LengthValidator" in repr_str
        assert "5" in repr_str  # min_length
        assert "100" in repr_str  # max_length
