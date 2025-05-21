"""
Tests for the base validator.
"""

from unittest.mock import MagicMock, patch

import pytest

from sifaka.core.thought import Thought
from sifaka.validators.base_validator import BaseValidator, safe_validate


class MockValidator(BaseValidator):
    """Mock validator for testing BaseValidator functionality."""

    def _validate(self, text):
        """Mock implementation of _validate."""
        if text == "valid":
            return {
                "passed": True,
                "message": "Text is valid",
                "details": {"reason": "text_is_valid"},
            }
        else:
            return {
                "passed": False,
                "message": "Text is invalid",
                "details": {"reason": "text_is_invalid"},
                "score": 0.5,
            }


def test_base_validator_initialization():
    """Test that a base validator can be initialized with various options."""
    # Test with default name
    validator = MockValidator()
    assert validator.name == "MockValidator"

    # Test with custom name
    validator = MockValidator(name="CustomValidator")
    assert validator.name == "CustomValidator"

    # Test with options
    validator = MockValidator(option1="value1", option2="value2")
    assert validator._options["option1"] == "value1"
    assert validator._options["option2"] == "value2"


def test_base_validator_validate_empty_text():
    """Test that validating empty text fails."""
    validator = MockValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = ""

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "MockValidator"
    assert "Empty text" in thought.validation_results[0].message
    assert thought.validation_results[0].details["error_type"] == "EmptyText"


def test_base_validator_validate_valid_text():
    """Test that validating valid text passes."""
    validator = MockValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = "valid"

    result = validator.validate(thought)

    assert result is True
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is True
    assert thought.validation_results[0].validator_name == "MockValidator"
    assert thought.validation_results[0].message == "Text is valid"
    assert thought.validation_results[0].details["reason"] == "text_is_valid"
    assert thought.validation_results[0].score == 1.0  # Default score for passing validation
    assert "processing_time_ms" in thought.validation_results[0].details


def test_base_validator_validate_invalid_text():
    """Test that validating invalid text fails."""
    validator = MockValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = "invalid"

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "MockValidator"
    assert thought.validation_results[0].message == "Text is invalid"
    assert thought.validation_results[0].details["reason"] == "text_is_invalid"
    assert thought.validation_results[0].score == 0.5  # Custom score for failing validation
    assert "processing_time_ms" in thought.validation_results[0].details


def test_base_validator_validate_error():
    """Test that validating text handles errors gracefully."""
    validator = MockValidator()
    validator._validate = MagicMock(side_effect=RuntimeError("Test error"))

    thought = Thought(prompt="Test prompt")
    thought.text = "test"

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "MockValidator"
    assert "Validation error" in thought.validation_results[0].message
    assert thought.validation_results[0].details["error_type"] == "RuntimeError"
    assert thought.validation_results[0].details["error_message"] == "Test error"
    assert thought.validation_results[0].score == 0.0  # Error score


def test_safe_validate():
    """Test that safe_validate wraps validation with error handling."""
    validator = MockValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = "valid"

    result = safe_validate(validator, thought)

    assert result is True
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is True


def test_safe_validate_error():
    """Test that safe_validate handles errors gracefully."""
    validator = MockValidator()
    validator.validate = MagicMock(side_effect=RuntimeError("Test error"))

    thought = Thought(prompt="Test prompt")
    thought.text = "test"

    result = safe_validate(validator, thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert "Validation error" in thought.validation_results[0].message
    assert thought.validation_results[0].details["error_type"] == "RuntimeError"
    assert thought.validation_results[0].details["error_message"] == "Test error"
