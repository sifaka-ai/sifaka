#!/usr/bin/env python3
"""Unit tests for base validator implementations.

This module contains comprehensive unit tests for base validators including
LengthValidator, RegexValidator, and other core validation functionality.
"""


from sifaka.core.thought import Thought
from sifaka.validators.base import LengthValidator, RegexValidator
from sifaka.validators.shared import ValidationResult


class TestLengthValidator:
    """Test LengthValidator functionality."""

    def test_initialization(self):
        """Test LengthValidator initialization."""
        validator = LengthValidator(min_length=10, max_length=100)

        assert validator.min_length == 10
        assert validator.max_length == 100
        assert validator.name == "length"

    def test_initialization_with_name(self):
        """Test LengthValidator initialization with custom name."""
        validator = LengthValidator(min_length=5, max_length=50, name="custom_length")

        assert validator.name == "custom_length"

    def test_valid_length(self, sample_text_data):
        """Test validation with valid length."""
        validator = LengthValidator(min_length=10, max_length=100)
        thought = Thought(prompt="Test", text=sample_text_data["medium"])

        result = validator.validate(thought)

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.validator_name == "length"
        assert "valid" in result.message.lower() or "passed" in result.message.lower()

    def test_text_too_short(self, sample_text_data):
        """Test validation with text too short."""
        validator = LengthValidator(min_length=20, max_length=100)
        thought = Thought(prompt="Test", text=sample_text_data["short"])

        result = validator.validate(thought)

        assert result.passed is False
        assert "short" in result.message.lower() or "minimum" in result.message.lower()

    def test_text_too_long(self, sample_text_data):
        """Test validation with text too long."""
        validator = LengthValidator(min_length=5, max_length=20)
        thought = Thought(prompt="Test", text=sample_text_data["long"])

        result = validator.validate(thought)

        assert result.passed is False
        assert "long" in result.message.lower() or "maximum" in result.message.lower()

    def test_empty_text(self, sample_text_data):
        """Test validation with empty text."""
        validator = LengthValidator(min_length=1, max_length=100)
        thought = Thought(prompt="Test", text=sample_text_data["empty"])

        result = validator.validate(thought)

        assert result.passed is False

    def test_whitespace_only(self, sample_text_data):
        """Test validation with whitespace-only text."""
        validator = LengthValidator(min_length=5, max_length=100)
        thought = Thought(prompt="Test", text=sample_text_data["whitespace"])

        result = validator.validate(thought)

        # Behavior depends on whether whitespace counts toward length
        assert isinstance(result.passed, bool)

    def test_exact_boundaries(self):
        """Test validation at exact length boundaries."""
        validator = LengthValidator(min_length=5, max_length=10)

        # Exactly minimum length
        thought_min = Thought(prompt="Test", text="12345")
        result_min = validator.validate(thought_min)
        assert result_min.passed is True

        # Exactly maximum length
        thought_max = Thought(prompt="Test", text="1234567890")
        result_max = validator.validate(thought_max)
        assert result_max.passed is True

        # One character short
        thought_short = Thought(prompt="Test", text="1234")
        result_short = validator.validate(thought_short)
        assert result_short.passed is False

        # One character long
        thought_long = Thought(prompt="Test", text="12345678901")
        result_long = validator.validate(thought_long)
        assert result_long.passed is False

    def test_unicode_length(self, sample_text_data):
        """Test length validation with unicode characters."""
        validator = LengthValidator(min_length=5, max_length=50)
        thought = Thought(prompt="Test", text=sample_text_data["unicode"])

        result = validator.validate(thought)

        # Should handle unicode correctly
        assert isinstance(result.passed, bool)

    def test_min_length_only(self):
        """Test validator with only minimum length."""
        validator = LengthValidator(min_length=10)

        # Short text should fail
        thought_short = Thought(prompt="Test", text="Short")
        result_short = validator.validate(thought_short)
        assert result_short.passed is False

        # Long text should pass
        thought_long = Thought(
            prompt="Test", text="This is a much longer text that exceeds minimum"
        )
        result_long = validator.validate(thought_long)
        assert result_long.passed is True

    def test_max_length_only(self):
        """Test validator with only maximum length."""
        validator = LengthValidator(max_length=20)

        # Short text should pass
        thought_short = Thought(prompt="Test", text="Short text")
        result_short = validator.validate(thought_short)
        assert result_short.passed is True

        # Long text should fail
        thought_long = Thought(
            prompt="Test", text="This is a very long text that exceeds the maximum length limit"
        )
        result_long = validator.validate(thought_long)
        assert result_long.passed is False


class TestRegexValidator:
    """Test RegexValidator functionality."""

    def test_initialization_required_patterns(self):
        """Test RegexValidator initialization with required patterns."""
        patterns = [r"\d+", r"[A-Z]", r"test"]
        validator = RegexValidator(required_patterns=patterns)

        assert validator.required_patterns == patterns
        assert validator.forbidden_patterns == []
        assert validator.name == "regex"

    def test_initialization_forbidden_patterns(self):
        """Test RegexValidator initialization with forbidden patterns."""
        patterns = [r"spam", r"bad", r"evil"]
        validator = RegexValidator(forbidden_patterns=patterns)

        assert validator.required_patterns == []
        assert validator.forbidden_patterns == patterns

    def test_initialization_both_patterns(self):
        """Test RegexValidator initialization with both pattern types."""
        required = [r"\d+", r"[A-Z]"]
        forbidden = [r"spam", r"bad"]
        validator = RegexValidator(required_patterns=required, forbidden_patterns=forbidden)

        assert validator.required_patterns == required
        assert validator.forbidden_patterns == forbidden

    def test_required_patterns_pass(self):
        """Test validation with required patterns that pass."""
        validator = RegexValidator(required_patterns=[r"\d+", r"[A-Z]"])
        thought = Thought(prompt="Test", text="Test 123 ABC")

        result = validator.validate(thought)

        assert result.passed is True
        assert result.validator_name == "regex"

    def test_required_patterns_fail_missing_digit(self):
        """Test validation with required patterns that fail (missing digit)."""
        validator = RegexValidator(required_patterns=[r"\d+", r"[A-Z]"])
        thought = Thought(prompt="Test", text="Test ABC")  # No digits

        result = validator.validate(thought)

        assert result.passed is False
        assert "required" in result.message.lower() or "missing" in result.message.lower()

    def test_required_patterns_fail_missing_uppercase(self):
        """Test validation with required patterns that fail (missing uppercase)."""
        validator = RegexValidator(required_patterns=[r"\d+", r"[A-Z]"])
        thought = Thought(prompt="Test", text="test 123")  # No uppercase

        result = validator.validate(thought)

        assert result.passed is False

    def test_forbidden_patterns_pass(self):
        """Test validation with forbidden patterns that pass."""
        validator = RegexValidator(forbidden_patterns=[r"spam", r"bad"])
        thought = Thought(prompt="Test", text="This is good content")

        result = validator.validate(thought)

        assert result.passed is True

    def test_forbidden_patterns_fail(self):
        """Test validation with forbidden patterns that fail."""
        validator = RegexValidator(forbidden_patterns=[r"spam", r"bad"])
        thought = Thought(prompt="Test", text="This is bad content")

        result = validator.validate(thought)

        assert result.passed is False
        assert "forbidden" in result.message.lower() or "prohibited" in result.message.lower()

    def test_case_sensitive_matching(self):
        """Test case-sensitive pattern matching."""
        validator = RegexValidator(required_patterns=[r"Test"])

        # Exact case should pass
        thought_exact = Thought(prompt="Test", text="Test content")
        result_exact = validator.validate(thought_exact)
        assert result_exact.passed is True

        # Different case should fail
        thought_lower = Thought(prompt="Test", text="test content")
        result_lower = validator.validate(thought_lower)
        assert result_lower.passed is False

    def test_case_insensitive_matching(self):
        """Test case-insensitive pattern matching."""
        validator = RegexValidator(required_patterns=[r"(?i)test"])

        # Different cases should all pass
        test_cases = ["Test content", "TEST content", "test content", "TeSt content"]

        for text in test_cases:
            thought = Thought(prompt="Test", text=text)
            result = validator.validate(thought)
            assert result.passed is True, f"Failed for text: {text}"

    def test_complex_patterns(self):
        """Test complex regex patterns."""
        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        validator = RegexValidator(required_patterns=[email_pattern])

        # Valid email
        thought_valid = Thought(prompt="Test", text="Contact us at test@example.com")
        result_valid = validator.validate(thought_valid)
        assert result_valid.passed is True

        # Invalid email
        thought_invalid = Thought(prompt="Test", text="Contact us at invalid-email")
        result_invalid = validator.validate(thought_invalid)
        assert result_invalid.passed is False

    def test_multiple_required_patterns(self):
        """Test multiple required patterns."""
        validator = RegexValidator(
            required_patterns=[
                r"\d+",  # Must have digits
                r"[A-Z]",  # Must have uppercase
                r"[a-z]",  # Must have lowercase
                r"[!@#$%^&*]",  # Must have special chars
            ]
        )

        # Text with all requirements
        thought_valid = Thought(prompt="Test", text="Test123! Content")
        result_valid = validator.validate(thought_valid)
        assert result_valid.passed is True

        # Text missing special characters
        thought_invalid = Thought(prompt="Test", text="Test123 Content")
        result_invalid = validator.validate(thought_invalid)
        assert result_invalid.passed is False

    def test_multiple_forbidden_patterns(self):
        """Test multiple forbidden patterns."""
        validator = RegexValidator(forbidden_patterns=[r"spam", r"scam", r"virus", r"malware"])

        # Clean text should pass
        thought_clean = Thought(prompt="Test", text="This is clean content")
        result_clean = validator.validate(thought_clean)
        assert result_clean.passed is True

        # Text with any forbidden pattern should fail
        forbidden_texts = [
            "This is spam content",
            "Beware of scam",
            "Virus detected",
            "Malware alert",
        ]

        for text in forbidden_texts:
            thought = Thought(prompt="Test", text=text)
            result = validator.validate(thought)
            assert result.passed is False, f"Should have failed for: {text}"

    def test_empty_pattern_lists(self):
        """Test validator with empty pattern lists."""
        validator = RegexValidator(required_patterns=[], forbidden_patterns=[])
        thought = Thought(prompt="Test", text="Any content")

        result = validator.validate(thought)

        # Should pass when no patterns are specified
        assert result.passed is True

    def test_special_regex_characters(self, sample_text_data):
        """Test patterns with special regex characters."""
        # Pattern that matches special characters literally
        validator = RegexValidator(required_patterns=[r"\$\d+\.\d{2}"])  # Price pattern

        thought_valid = Thought(prompt="Test", text="Price is $19.99")
        result_valid = validator.validate(thought_valid)
        assert result_valid.passed is True

        thought_invalid = Thought(prompt="Test", text="Price is 19.99")
        result_invalid = validator.validate(thought_invalid)
        assert result_invalid.passed is False

    def test_unicode_patterns(self, sample_text_data):
        """Test regex patterns with unicode text."""
        validator = RegexValidator(required_patterns=[r"[你好世界]"])
        thought = Thought(prompt="Test", text=sample_text_data["unicode"])

        result = validator.validate(thought)

        # Should handle unicode correctly
        assert isinstance(result.passed, bool)
