"""
Tests for the validators module.

This module contains tests for the validators in the Sifaka framework.
"""

from typing import Any, Optional

import pytest

from sifaka.results import ValidationResult
from sifaka.validators import json_format, length, prohibited_content
from sifaka.validators.base import BaseValidator, safe_validate


class TestBaseValidator:
    """Tests for the BaseValidator class."""

    def test_init_with_defaults(self) -> None:
        """Test initializing a BaseValidator with default parameters."""

        class TestValidator(BaseValidator):
            def _validate(self, text: str) -> ValidationResult:
                return ValidationResult(passed=True, message="Test passed")

        validator = TestValidator()
        assert validator.name == "TestValidator"
        assert validator._options == {}

    def test_init_with_name(self) -> None:
        """Test initializing a BaseValidator with a custom name."""

        class TestValidator(BaseValidator):
            def _validate(self, text: str) -> ValidationResult:
                return ValidationResult(passed=True, message="Test passed")

        validator = TestValidator(name="CustomName")
        assert validator.name == "CustomName"

    def test_init_with_options(self) -> None:
        """Test initializing a BaseValidator with options."""

        class TestValidator(BaseValidator):
            def _validate(self, text: str) -> ValidationResult:
                return ValidationResult(passed=True, message="Test passed")

        validator = TestValidator(option1="value1", option2="value2")
        assert validator._options == {"option1": "value1", "option2": "value2"}

    def test_configure(self) -> None:
        """Test configuring a BaseValidator with new options."""

        class TestValidator(BaseValidator):
            def _validate(self, text: str) -> ValidationResult:
                return ValidationResult(passed=True, message="Test passed")

        validator = TestValidator(option1="value1")
        validator.configure(option2="value2", option3="value3")
        assert validator._options == {
            "option1": "value1",
            "option2": "value2",
            "option3": "value3",
        }

    def test_validate_empty_text(self) -> None:
        """Test validating empty text."""

        class TestValidator(BaseValidator):
            def _validate(self, text: str) -> ValidationResult:
                return ValidationResult(passed=True, message="Test passed")

        validator = TestValidator()
        result = validator.validate("")
        assert result.passed is False
        assert "empty" in result.message.lower()

    def test_validate_calls_validate_method(self) -> None:
        """Test that validate calls the _validate method."""

        class TestValidator(BaseValidator):
            def __init__(self, name: Optional[str] = None, **options: Any):
                super().__init__(name=name, **options)
                self.validate_called = False

            def _validate(self, text: str) -> ValidationResult:
                self.validate_called = True
                return ValidationResult(passed=True, message="Test passed")

        validator = TestValidator()
        validator.validate("Test text")
        assert validator.validate_called is True

    def test_validate_handles_exceptions(self) -> None:
        """Test that validate handles exceptions from _validate."""

        class TestValidator(BaseValidator):
            def _validate(self, text: str) -> ValidationResult:
                raise ValueError("Test error")

        validator = TestValidator()
        result = validator.validate("Test text")
        assert result.passed is False
        assert "error" in result.message.lower()
        assert "test error" in result.message.lower()


class TestSafeValidate:
    """Tests for the safe_validate function."""

    def test_safe_validate_with_passing_validator(self, mock_validator) -> None:
        """Test safe_validate with a validator that passes."""
        result = safe_validate(mock_validator, "Test text")
        assert result.passed is True
        assert len(mock_validator.validate_calls) == 1
        assert mock_validator.validate_calls[0] == "Test text"

    @pytest.mark.parametrize("mock_validator", [False], indirect=True)
    def test_safe_validate_with_failing_validator(self, mock_validator) -> None:
        """Test safe_validate with a validator that fails."""
        result = safe_validate(mock_validator, "Test text")
        assert result.passed is False
        assert len(mock_validator.validate_calls) == 1
        assert mock_validator.validate_calls[0] == "Test text"

    def test_safe_validate_with_exception(self) -> None:
        """Test safe_validate with a validator that raises an exception."""

        class ExceptionValidator:
            @property
            def name(self) -> str:
                return "ExceptionValidator"

            def validate(self, text: str) -> ValidationResult:
                raise ValueError("Test error")

        validator = ExceptionValidator()
        result = safe_validate(validator, "Test text")
        assert result.passed is False
        assert "error" in result.message.lower()
        assert "test error" in result.message.lower()


class TestLengthValidator:
    """Tests for the length validator."""

    def test_length_validator_with_min_words(self) -> None:
        """Test the length validator with minimum word count."""
        validator = length(min_words=5)

        # Test with text that meets the minimum word count
        result = validator.validate("This is a test with five words.")
        assert result.passed is True

        # Test with text that doesn't meet the minimum word count
        result = validator.validate("Too few words.")
        assert result.passed is False
        assert "minimum" in result.message.lower()
        assert "words" in result.message.lower()

    def test_length_validator_with_max_words(self) -> None:
        """Test the length validator with maximum word count."""
        validator = length(max_words=5)

        # Test with text that meets the maximum word count
        result = validator.validate("This is five words only.")
        assert result.passed is True

        # Test with text that exceeds the maximum word count
        result = validator.validate(
            "This text has more than five words and should fail validation."
        )
        assert result.passed is False
        assert "maximum" in result.message.lower()
        assert "words" in result.message.lower()

    def test_length_validator_with_min_and_max_words(self) -> None:
        """Test the length validator with minimum and maximum word count."""
        validator = length(min_words=3, max_words=5)

        # Test with text that meets both constraints
        result = validator.validate("This is four words.")
        assert result.passed is True

        # Test with text that doesn't meet the minimum word count
        result = validator.validate("Too few.")
        assert result.passed is False
        assert "minimum" in result.message.lower()

        # Test with text that exceeds the maximum word count
        result = validator.validate(
            "This text has more than five words and should fail validation."
        )
        assert result.passed is False
        assert "maximum" in result.message.lower()

    def test_length_validator_with_min_chars(self) -> None:
        """Test the length validator with minimum character count."""
        validator = length(min_chars=20)

        # Test with text that meets the minimum character count
        result = validator.validate("This text has more than twenty characters.")
        assert result.passed is True

        # Test with text that doesn't meet the minimum character count
        result = validator.validate("Too short.")
        assert result.passed is False
        assert "minimum" in result.message.lower()
        assert "characters" in result.message.lower()

    def test_length_validator_with_max_chars(self) -> None:
        """Test the length validator with maximum character count."""
        validator = length(max_chars=20)

        # Test with text that meets the maximum character count
        result = validator.validate("Short enough text.")
        assert result.passed is True

        # Test with text that exceeds the maximum character count
        result = validator.validate(
            "This text has more than twenty characters and should fail validation."
        )
        assert result.passed is False
        assert "maximum" in result.message.lower()
        assert "characters" in result.message.lower()


class TestProhibitedContentValidator:
    """Tests for the prohibited_content validator."""

    def test_prohibited_content_validator_with_single_term(self) -> None:
        """Test the prohibited_content validator with a single prohibited term."""
        validator = prohibited_content(prohibited=["prohibited"])

        # Test with text that doesn't contain the prohibited term
        result = validator.validate("This text is allowed.")
        assert result.passed is True

        # Test with text that contains the prohibited term
        result = validator.validate("This text contains a prohibited term.")
        assert result.passed is False
        assert "prohibited" in result.message.lower()

    def test_prohibited_content_validator_with_multiple_terms(self) -> None:
        """Test the prohibited_content validator with multiple prohibited terms."""
        validator = prohibited_content(prohibited=["term1", "term2", "term3"])

        # Test with text that doesn't contain any prohibited terms
        result = validator.validate("This text is allowed.")
        assert result.passed is True

        # Test with text that contains one of the prohibited terms
        result = validator.validate("This text contains term1 which is prohibited.")
        assert result.passed is False
        assert "term1" in result.message.lower()

        # Test with text that contains multiple prohibited terms
        result = validator.validate("This text contains term1 and term3 which are prohibited.")
        assert result.passed is False
        assert "term1" in result.message.lower()
        assert "term3" in result.message.lower()

    def test_prohibited_content_validator_case_sensitivity(self) -> None:
        """Test the prohibited_content validator with case sensitivity."""
        validator = prohibited_content(prohibited=["prohibited"], case_sensitive=True)

        # Test with text that contains the prohibited term in the same case
        result = validator.validate("This text contains a prohibited term.")
        assert result.passed is False

        # Test with text that contains the prohibited term in a different case
        result = validator.validate("This text contains a PROHIBITED term.")
        assert result.passed is True  # Should pass because case_sensitive=True

        # Test with case_sensitive=False
        validator = prohibited_content(prohibited=["prohibited"], case_sensitive=False)
        result = validator.validate("This text contains a PROHIBITED term.")
        assert result.passed is False  # Should fail because case_sensitive=False


class TestJsonFormatValidator:
    """Tests for the json_format validator."""

    def test_json_format_validator_with_valid_json(self) -> None:
        """Test the json_format validator with valid JSON."""
        validator = json_format()

        # Test with valid JSON
        result = validator.validate('{"name": "John", "age": 30}')
        assert result.passed is True

        # Test with invalid JSON
        result = validator.validate('{"name": "John", "age": }')
        assert result.passed is False
        assert "json" in result.message.lower()

    def test_json_format_validator_with_schema(self) -> None:
        """Test the json_format validator with a schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
            "required": ["name", "age"],
        }
        validator = json_format(schema=schema)

        # Test with JSON that matches the schema
        result = validator.validate('{"name": "John", "age": 30}')
        assert result.passed is True

        # Test with JSON that doesn't match the schema (missing required field)
        result = validator.validate('{"name": "John"}')
        assert result.passed is False
        assert "schema" in result.message.lower()

        # Test with JSON that doesn't match the schema (wrong type)
        result = validator.validate('{"name": "John", "age": "thirty"}')
        assert result.passed is False
        assert "schema" in result.message.lower()

    def test_json_format_validator_with_array(self) -> None:
        """Test the json_format validator with a JSON array."""
        validator = json_format()

        # Test with valid JSON array
        result = validator.validate("[1, 2, 3, 4, 5]")
        assert result.passed is True

        # Test with invalid JSON array
        result = validator.validate("[1, 2, 3, 4, ]")
        assert result.passed is False
        assert "json" in result.message.lower()
