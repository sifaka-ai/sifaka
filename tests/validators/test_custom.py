"""
Tests for the CustomValidator.
"""

import pytest
import re
from sifaka.validators import CustomValidator, create_regex_validator, create_threshold_validator
from sifaka.types import ValidationResult


def test_initialization():
    """Test CustomValidator initializes with correct parameters."""

    # Simple validation function
    def validate_func(text):
        return len(text) > 10

    validator = CustomValidator(
        validation_func=validate_func,
        name="Length Checker",
        description="Checks if text is longer than 10 characters",
        failure_message="Text is too short",
        success_message="Text has sufficient length",
        failure_score=0.0,
        success_score=1.0,
    )

    assert validator.validation_func is validate_func
    assert validator.name == "Length Checker"
    assert validator.description == "Checks if text is longer than 10 characters"
    assert validator.failure_message == "Text is too short"
    assert validator.success_message == "Text has sufficient length"
    assert validator.failure_score == 0.0
    assert validator.success_score == 1.0
    assert validator.default_issues == ["Text is too short"]
    assert "Check input and try again" in validator.default_suggestions


def test_boolean_validation():
    """Test CustomValidator with a validation function that returns a boolean."""

    # Simple validation function that returns a boolean
    def validate_length(text):
        return len(text) > 10

    validator = CustomValidator(
        validation_func=validate_length,
        name="Length Checker",
        failure_message="Text is too short",
        success_message="Text has sufficient length",
    )

    # Test passing case
    result1 = validator.validate("This text is longer than 10 characters")
    assert result1.passed is True
    assert result1.message == "Text has sufficient length"
    assert result1.score == 1.0
    assert len(result1.issues) == 0
    assert len(result1.suggestions) == 0
    assert result1.metadata["validator_name"] == "Length Checker"

    # Test failing case
    result2 = validator.validate("Too short")
    assert result2.passed is False
    assert result2.message == "Text is too short"
    assert result2.score == 0.0
    assert "Text is too short" in result2.issues
    assert "Check input and try again" in result2.suggestions


def test_tuple_validation():
    """Test CustomValidator with a validation function that returns a tuple."""

    # Validation function that returns a tuple with detailed information
    def validate_with_details(text):
        is_valid = len(text) > 10
        details = {
            "message": f"Text length: {len(text)} characters",
            "score": min(1.0, len(text) / 20.0),  # Score based on length, max at 20 chars
            "issues": [] if is_valid else [f"Text has only {len(text)} characters, need > 10"],
            "suggestions": [] if is_valid else ["Add more content"],
            "text_length": len(text),
            "custom_field": "Extra information",
        }
        return is_valid, details

    validator = CustomValidator(
        validation_func=validate_with_details,
        name="Detailed Length Checker",
    )

    # Test passing case
    result1 = validator.validate("This text is longer than 10 characters")
    assert result1.passed is True
    assert "Text length:" in result1.message
    assert 0.0 < result1.score <= 1.0
    assert len(result1.issues) == 0
    assert len(result1.suggestions) == 0
    assert result1.metadata["text_length"] > 10
    assert result1.metadata["custom_field"] == "Extra information"

    # Test failing case
    result2 = validator.validate("Too short")
    assert result2.passed is False
    assert "Text length:" in result2.message
    assert 0.0 <= result2.score < 1.0
    assert "only" in result2.issues[0]
    assert "Add more content" in result2.suggestions
    assert result2.metadata["text_length"] < 10


def test_empty_text():
    """Test handling of empty text."""
    validator = CustomValidator(
        validation_func=lambda text: True,  # Always pass
        name="Test Validator",
    )

    result = validator.validate("")

    assert result.passed is False
    assert "Empty text" in result.message
    assert result.score == 0.0
    assert "Text is empty" in result.issues[0]
    assert result.metadata["validator_name"] == "Test Validator"


def test_validation_function_error():
    """Test handling of errors in the validation function."""

    # Validation function that raises an exception
    def buggy_validator(text):
        # This will raise a ZeroDivisionError
        return 10 / 0

    validator = CustomValidator(
        validation_func=buggy_validator,
        name="Buggy Validator",
    )

    result = validator.validate("Some text")

    assert result.passed is False
    assert "Error in custom validation" in result.message
    assert result.score == 0.0
    assert "Validation function error" in result.issues[0]
    assert "ZeroDivisionError" in result.metadata["error_type"]


def test_invalid_result_type():
    """Test handling of invalid return type from validation function."""

    # Validation function that returns an invalid type
    def invalid_validator(text):
        return "not a boolean or tuple"

    validator = CustomValidator(
        validation_func=invalid_validator,
        name="Invalid Validator",
    )

    result = validator.validate("Some text")

    assert result.passed is False
    assert "Error in custom validation" in result.message
    assert "Invalid validation function result" in result.metadata["error"]


def test_create_regex_validator():
    """Test the create_regex_validator factory function."""
    # Create a validator that requires text to contain an email address
    validator = create_regex_validator(
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        mode="match",
        name="Email Validator",
        failure_message="Text must contain a valid email address",
    )

    # Test text with email
    result1 = validator.validate("Contact us at info@example.com for details.")
    assert result1.passed is True
    assert result1.score == 1.0

    # Test text without email
    result2 = validator.validate("Contact us for details.")
    assert result2.passed is False
    assert result2.score == 0.0
    assert "doesn't match the required pattern" in result2.issues[0]

    # Test forbid mode
    forbid_validator = create_regex_validator(
        pattern=r"bad|awful|terrible",
        mode="not_match",
        name="Negative Words Validator",
        failure_message="Text contains negative words",
    )

    # Test text with forbidden words
    result3 = forbid_validator.validate("That was a bad experience.")
    assert result3.passed is False
    assert "matches forbidden pattern" in result3.issues[0]

    # Test text without forbidden words
    result4 = forbid_validator.validate("That was a good experience.")
    assert result4.passed is True


def test_create_threshold_validator():
    """Test the create_threshold_validator factory function."""

    # Function to count words in text
    def count_words(text):
        return len(text.split())

    # Create a validator that requires between 5 and 20 words
    validator = create_threshold_validator(
        extraction_func=count_words,
        min_threshold=5,
        max_threshold=20,
        name="Word Count Validator",
        description="Validates word count is between 5 and 20",
        value_name="word count",
    )

    # Test text with too few words
    result1 = validator.validate("Too few words.")
    assert result1.passed is False
    assert "word count" in result1.message.lower()
    assert "below minimum threshold" in result1.issues[0]
    assert "Increase word count" in result1.suggestions[0]

    # Test text with too many words
    too_many_words = "This text has too many words. " * 5  # 25 words
    result2 = validator.validate(too_many_words)
    assert result2.passed is False
    assert "exceeds maximum threshold" in result2.issues[0]
    assert "Decrease word count" in result2.suggestions[0]

    # Test text with acceptable word count
    result3 = validator.validate("This text has an acceptable number of words for the validator.")
    assert result3.passed is True

    # Test threshold validator with only min_threshold
    min_validator = create_threshold_validator(
        extraction_func=count_words, min_threshold=10, value_name="word count"
    )

    result4 = min_validator.validate("This sentence has exactly ten words in it.")
    assert result4.passed is True

    # Test threshold validator with extraction function error
    def buggy_extractor(text):
        return int(text)  # Will fail if text is not a number

    error_validator = create_threshold_validator(
        extraction_func=buggy_extractor, min_threshold=10, value_name="numeric value"
    )

    result5 = error_validator.validate("Not a number")
    assert result5.passed is False
    assert "Error extracting numeric value" in result5.message
    assert "Could not extract numeric value from text" in result5.issues[0]


def test_threshold_validator_configuration():
    """Test validation of threshold validator configuration."""
    # Test no thresholds provided
    with pytest.raises(ValueError) as excinfo:
        create_threshold_validator(
            extraction_func=lambda x: len(x),
        )
    assert "At least one of min_threshold or max_threshold must be specified" in str(excinfo.value)
