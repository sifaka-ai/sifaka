"""
Tests for the length validator.
"""

import pytest
from core.thought import Thought
from validators.length_validator import LengthValidator, create_length_validator


def test_length_validator_initialization():
    """Test that a length validator can be initialized with various constraints."""
    # Test with min_words
    validator = LengthValidator(min_words=10)
    assert validator.name == "LengthValidator"
    assert validator.min_words == 10
    assert validator.max_words is None
    assert validator.min_chars is None
    assert validator.max_chars is None

    # Test with max_words
    validator = LengthValidator(max_words=100)
    assert validator.max_words == 100

    # Test with min_chars
    validator = LengthValidator(min_chars=50)
    assert validator.min_chars == 50

    # Test with max_chars
    validator = LengthValidator(max_chars=500)
    assert validator.max_chars == 500

    # Test with custom name
    validator = LengthValidator(min_words=10, name="CustomLengthValidator")
    assert validator.name == "CustomLengthValidator"

    # Test with multiple constraints
    validator = LengthValidator(min_words=10, max_words=100, min_chars=50, max_chars=500)
    assert validator.min_words == 10
    assert validator.max_words == 100
    assert validator.min_chars == 50
    assert validator.max_chars == 500


def test_length_validator_initialization_errors():
    """Test that initializing a length validator with invalid constraints raises errors."""
    # Test with no constraints
    with pytest.raises(ValueError):
        LengthValidator()

    # Test with min_words > max_words
    with pytest.raises(ValueError):
        LengthValidator(min_words=100, max_words=50)

    # Test with min_chars > max_chars
    with pytest.raises(ValueError):
        LengthValidator(min_chars=500, max_chars=250)


def test_length_validator_validate_empty_text():
    """Test that validating empty text fails."""
    validator = LengthValidator(min_words=10)
    thought = Thought(prompt="Test prompt")
    thought.text = ""

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "LengthValidator"
    assert "Empty text" in thought.validation_results[0].message


def test_length_validator_validate_min_words():
    """Test that validating text with too few words fails."""
    validator = LengthValidator(min_words=10)
    thought = Thought(prompt="Test prompt")
    thought.text = "This is only five words total."

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "LengthValidator"
    assert "too short" in thought.validation_results[0].message
    assert "minimum 10 words" in thought.validation_results[0].message


def test_length_validator_validate_max_words():
    """Test that validating text with too many words fails."""
    validator = LengthValidator(max_words=5)
    thought = Thought(prompt="Test prompt")
    thought.text = "This is more than five words in total."

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "LengthValidator"
    assert "too long" in thought.validation_results[0].message
    assert "maximum 5 words" in thought.validation_results[0].message


def test_length_validator_validate_min_chars():
    """Test that validating text with too few characters fails."""
    validator = LengthValidator(min_chars=50)
    thought = Thought(prompt="Test prompt")
    thought.text = "This text is too short."

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "LengthValidator"
    assert "too short" in thought.validation_results[0].message
    assert "minimum 50 characters" in thought.validation_results[0].message


def test_length_validator_validate_max_chars():
    """Test that validating text with too many characters fails."""
    validator = LengthValidator(max_chars=20)
    thought = Thought(prompt="Test prompt")
    thought.text = "This text is too long for the maximum character limit."

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "LengthValidator"
    assert "too long" in thought.validation_results[0].message
    assert "maximum 20 characters" in thought.validation_results[0].message


def test_length_validator_validate_success():
    """Test that validating text that meets all constraints passes."""
    validator = LengthValidator(min_words=5, max_words=10, min_chars=20, max_chars=100)
    thought = Thought(prompt="Test prompt")
    thought.text = "This text meets all the length requirements."

    result = validator.validate(thought)

    assert result is True
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is True
    assert thought.validation_results[0].validator_name == "LengthValidator"
    assert "meets length requirements" in thought.validation_results[0].message


def test_create_length_validator():
    """Test that the create_length_validator function creates a LengthValidator."""
    validator = create_length_validator(min_words=10, max_words=100)

    assert isinstance(validator, LengthValidator)
    assert validator.min_words == 10
    assert validator.max_words == 100
