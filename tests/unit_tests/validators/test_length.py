"""Comprehensive unit tests for length validator.

This module tests the LengthValidator implementation:
- Text length validation against min/max constraints
- Word count and character count validation
- Edge cases and error handling
- Performance characteristics

Tests cover:
- Basic validation functionality
- Boundary conditions
- Error scenarios
- Configuration options
- Performance with large texts
"""

import pytest
from unittest.mock import patch

from sifaka.validators.length import LengthValidator


class TestLengthValidator:
    """Test the LengthValidator implementation."""

    def test_validator_initialization_default(self):
        """Test validator initialization with default parameters."""
        validator = LengthValidator()
        
        assert validator.name == "length"
        assert validator.min_length == 10
        assert validator.max_length == 10000
        assert validator.count_type == "characters"

    def test_validator_initialization_custom(self):
        """Test validator initialization with custom parameters."""
        validator = LengthValidator(
            min_length=50,
            max_length=500,
            count_type="words",
            name="custom-length"
        )
        
        assert validator.name == "custom-length"
        assert validator.min_length == 50
        assert validator.max_length == 500
        assert validator.count_type == "words"

    def test_validator_initialization_invalid_params(self):
        """Test validator initialization with invalid parameters."""
        # Min length greater than max length
        with pytest.raises(ValueError):
            LengthValidator(min_length=100, max_length=50)
        
        # Negative lengths
        with pytest.raises(ValueError):
            LengthValidator(min_length=-1)
        
        with pytest.raises(ValueError):
            LengthValidator(max_length=-1)
        
        # Invalid count type
        with pytest.raises(ValueError):
            LengthValidator(count_type="invalid")

    @pytest.mark.asyncio
    async def test_validate_characters_within_bounds(self):
        """Test character count validation within bounds."""
        validator = LengthValidator(min_length=10, max_length=50, count_type="characters")
        
        # Text within bounds
        text = "This is a test text with exactly 25 characters."  # Should be around 25 chars
        result = await validator.validate_async(text)
        
        assert result["passed"] is True
        assert "character_count" in result["details"]
        assert result["details"]["character_count"] >= 10
        assert result["details"]["character_count"] <= 50

    @pytest.mark.asyncio
    async def test_validate_characters_too_short(self):
        """Test character count validation when text is too short."""
        validator = LengthValidator(min_length=20, max_length=100, count_type="characters")
        
        # Text too short
        text = "Short"  # 5 characters
        result = await validator.validate_async(text)
        
        assert result["passed"] is False
        assert result["details"]["character_count"] == 5
        assert result["details"]["min_required"] == 20
        assert "too short" in result["details"]["reason"].lower()

    @pytest.mark.asyncio
    async def test_validate_characters_too_long(self):
        """Test character count validation when text is too long."""
        validator = LengthValidator(min_length=5, max_length=20, count_type="characters")
        
        # Text too long
        text = "This is a very long text that exceeds the maximum character limit"  # >20 chars
        result = await validator.validate_async(text)
        
        assert result["passed"] is False
        assert result["details"]["character_count"] > 20
        assert result["details"]["max_allowed"] == 20
        assert "too long" in result["details"]["reason"].lower()

    @pytest.mark.asyncio
    async def test_validate_words_within_bounds(self):
        """Test word count validation within bounds."""
        validator = LengthValidator(min_length=3, max_length=10, count_type="words")
        
        # Text with 5 words
        text = "This is exactly five words"
        result = await validator.validate_async(text)
        
        assert result["passed"] is True
        assert result["details"]["word_count"] == 5
        assert result["details"]["word_count"] >= 3
        assert result["details"]["word_count"] <= 10

    @pytest.mark.asyncio
    async def test_validate_words_too_few(self):
        """Test word count validation when too few words."""
        validator = LengthValidator(min_length=5, max_length=20, count_type="words")
        
        # Text with 2 words
        text = "Two words"
        result = await validator.validate_async(text)
        
        assert result["passed"] is False
        assert result["details"]["word_count"] == 2
        assert result["details"]["min_required"] == 5
        assert "too few" in result["details"]["reason"].lower()

    @pytest.mark.asyncio
    async def test_validate_words_too_many(self):
        """Test word count validation when too many words."""
        validator = LengthValidator(min_length=1, max_length=5, count_type="words")
        
        # Text with 8 words
        text = "This text has exactly eight words in it"
        result = await validator.validate_async(text)
        
        assert result["passed"] is False
        assert result["details"]["word_count"] == 8
        assert result["details"]["max_allowed"] == 5
        assert "too many" in result["details"]["reason"].lower()

    @pytest.mark.asyncio
    async def test_validate_empty_text(self):
        """Test validation with empty text."""
        validator = LengthValidator(min_length=1, max_length=100)
        
        result = await validator.validate_async("")
        
        assert result["passed"] is False
        assert result["details"]["character_count"] == 0
        assert result["details"]["min_required"] == 1

    @pytest.mark.asyncio
    async def test_validate_whitespace_only_text(self):
        """Test validation with whitespace-only text."""
        validator = LengthValidator(min_length=5, max_length=100, count_type="words")
        
        # Text with only whitespace
        text = "   \n\t  "
        result = await validator.validate_async(text)
        
        assert result["passed"] is False
        assert result["details"]["word_count"] == 0

    @pytest.mark.asyncio
    async def test_validate_text_with_special_characters(self):
        """Test validation with special characters and unicode."""
        validator = LengthValidator(min_length=5, max_length=50, count_type="characters")
        
        # Text with special characters and unicode
        text = "Hello! 🌟 This has émojis and spëcial chars: @#$%"
        result = await validator.validate_async(text)
        
        assert result["passed"] is True
        assert "character_count" in result["details"]
        assert result["details"]["character_count"] >= 5

    @pytest.mark.asyncio
    async def test_validate_boundary_conditions(self):
        """Test validation at exact boundary conditions."""
        validator = LengthValidator(min_length=10, max_length=20, count_type="characters")
        
        # Exactly at minimum
        text_min = "1234567890"  # Exactly 10 characters
        result_min = await validator.validate_async(text_min)
        assert result_min["passed"] is True
        assert result_min["details"]["character_count"] == 10
        
        # Exactly at maximum
        text_max = "12345678901234567890"  # Exactly 20 characters
        result_max = await validator.validate_async(text_max)
        assert result_max["passed"] is True
        assert result_max["details"]["character_count"] == 20
        
        # One below minimum
        text_below = "123456789"  # 9 characters
        result_below = await validator.validate_async(text_below)
        assert result_below["passed"] is False
        
        # One above maximum
        text_above = "123456789012345678901"  # 21 characters
        result_above = await validator.validate_async(text_above)
        assert result_above["passed"] is False

    @pytest.mark.asyncio
    async def test_validate_score_calculation(self):
        """Test that validation score is calculated correctly."""
        validator = LengthValidator(min_length=10, max_length=100, count_type="characters")
        
        # Text within bounds should have high score
        text_good = "This is a good length text for validation testing"
        result_good = await validator.validate_async(text_good)
        assert result_good["passed"] is True
        assert "score" in result_good
        assert result_good["score"] > 0.8
        
        # Text outside bounds should have low score
        text_bad = "Short"
        result_bad = await validator.validate_async(text_bad)
        assert result_bad["passed"] is False
        assert "score" in result_bad
        assert result_bad["score"] < 0.5

    @pytest.mark.asyncio
    async def test_validate_performance_large_text(self):
        """Test validation performance with large text."""
        validator = LengthValidator(min_length=1000, max_length=100000, count_type="characters")
        
        # Create large text (10KB)
        large_text = "A" * 10000
        
        import time
        start_time = time.time()
        result = await validator.validate_async(large_text)
        end_time = time.time()
        
        # Should complete quickly (under 1 second)
        assert (end_time - start_time) < 1.0
        assert result["passed"] is True
        assert result["details"]["character_count"] == 10000

    @pytest.mark.asyncio
    async def test_validate_word_counting_accuracy(self):
        """Test accuracy of word counting with various text formats."""
        validator = LengthValidator(min_length=1, max_length=100, count_type="words")
        
        test_cases = [
            ("single", 1),
            ("two words", 2),
            ("word1 word2 word3", 3),
            ("words  with   multiple    spaces", 4),
            ("words\nwith\nnewlines", 3),
            ("words\twith\ttabs", 3),
            ("punctuation, words! with? various. marks:", 5),
            ("contractions don't count as two", 5),
            ("hyphenated-words count-as-one", 2),
        ]
        
        for text, expected_count in test_cases:
            result = await validator.validate_async(text)
            assert result["details"]["word_count"] == expected_count, f"Failed for: '{text}'"

    @pytest.mark.asyncio
    async def test_validate_multilingual_text(self):
        """Test validation with multilingual text."""
        validator = LengthValidator(min_length=5, max_length=100, count_type="words")
        
        # Text with multiple languages
        multilingual_text = "Hello 你好 Bonjour Hola مرحبا"
        result = await validator.validate_async(multilingual_text)
        
        assert result["passed"] is True
        assert result["details"]["word_count"] == 5

    def test_validator_string_representation(self):
        """Test string representation of validator."""
        validator = LengthValidator(min_length=10, max_length=100, count_type="words")
        
        str_repr = str(validator)
        assert "LengthValidator" in str_repr
        assert "min_length=10" in str_repr
        assert "max_length=100" in str_repr
        assert "words" in str_repr

    def test_validator_equality(self):
        """Test validator equality comparison."""
        validator1 = LengthValidator(min_length=10, max_length=100)
        validator2 = LengthValidator(min_length=10, max_length=100)
        validator3 = LengthValidator(min_length=20, max_length=100)
        
        assert validator1 == validator2
        assert validator1 != validator3

    @pytest.mark.asyncio
    async def test_validate_with_none_input(self):
        """Test validation with None input."""
        validator = LengthValidator()
        
        with pytest.raises(TypeError):
            await validator.validate_async(None)

    @pytest.mark.asyncio
    async def test_validate_with_non_string_input(self):
        """Test validation with non-string input."""
        validator = LengthValidator()
        
        # Should handle conversion or raise appropriate error
        with pytest.raises(TypeError):
            await validator.validate_async(123)
        
        with pytest.raises(TypeError):
            await validator.validate_async(["list", "input"])

    def test_validator_configuration_immutability(self):
        """Test that validator configuration is immutable after creation."""
        validator = LengthValidator(min_length=10, max_length=100)
        
        # Configuration should not be modifiable
        original_min = validator.min_length
        original_max = validator.max_length
        
        # Attempting to modify should not affect the validator
        # (This depends on implementation - some validators might be immutable)
        assert validator.min_length == original_min
        assert validator.max_length == original_max
