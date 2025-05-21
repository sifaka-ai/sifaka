"""
Tests for the profanity validator.
"""

from unittest.mock import MagicMock, patch

import pytest

from sifaka.core.thought import Thought
from sifaka.validators.profanity_validator import ProfanityValidator, create_profanity_validator


# Mock the ProfanityClassifier to avoid actual profanity detection during tests
@pytest.fixture
def mock_profanity_classifier():
    """Create a mock ProfanityClassifier for testing."""
    with patch("sifaka.validators.profanity_validator.ProfanityClassifier") as mock_cls:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        # Configure the mock classify method
        mock_result = MagicMock()
        mock_instance.classify.return_value = mock_result

        # Default to clean text
        mock_result.label = "clean"
        mock_result.confidence = 0.9
        mock_result.metadata = {
            "profane_words": [],
            "profane_word_count": 0,
        }

        yield mock_instance


def test_profanity_validator_initialization():
    """Test that a profanity validator can be initialized with various options."""
    # Test with default options
    validator = ProfanityValidator()
    assert validator.name == "ProfanityValidator"
    assert validator.threshold == 0.5

    # Test with custom threshold
    validator = ProfanityValidator(threshold=0.7)
    assert validator.threshold == 0.7

    # Test with custom words
    validator = ProfanityValidator(custom_words=["custom", "words"])
    assert validator.classifier is not None

    # Test with custom name
    validator = ProfanityValidator(name="CustomProfanityValidator")
    assert validator.name == "CustomProfanityValidator"


def test_profanity_validator_validate_empty_text(mock_profanity_classifier):
    """Test that validating empty text fails."""
    validator = ProfanityValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = ""

    result = validator.validate(thought)

    assert result is False  # Empty text should fail validation
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "ProfanityValidator"
    assert "Empty text" in thought.validation_results[0].message


def test_profanity_validator_validate_clean_text(mock_profanity_classifier):
    """Test that validating clean text passes."""
    # Configure mock to return clean text
    mock_profanity_classifier.classify.return_value.label = "clean"
    mock_profanity_classifier.classify.return_value.confidence = 0.9

    validator = ProfanityValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = "This is clean text."

    result = validator.validate(thought)

    assert result is True
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is True
    assert thought.validation_results[0].validator_name == "ProfanityValidator"
    assert "contains no profanity" in thought.validation_results[0].message


def test_profanity_validator_validate_profane_text(mock_profanity_classifier):
    """Test that validating profane text fails."""
    # Configure mock to return profane text
    mock_profanity_classifier.classify.return_value.label = "profane"
    mock_profanity_classifier.classify.return_value.confidence = 0.8
    mock_profanity_classifier.classify.return_value.metadata = {
        "profane_words": ["badword1", "badword2"],
        "profane_word_count": 2,
    }

    validator = ProfanityValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = "This text contains profanity."

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "ProfanityValidator"
    assert "contains profanity" in thought.validation_results[0].message

    # Check that details contain profane words
    details = thought.validation_results[0].details
    assert "profane_words" in details
    assert "badword1" in details["profane_words"]
    assert "badword2" in details["profane_words"]
    assert details["profane_word_count"] == 2


def test_profanity_validator_threshold(mock_profanity_classifier):
    """Test that the threshold option works correctly."""
    # Configure mock to return profane text with confidence below default threshold
    mock_profanity_classifier.classify.return_value.label = "profane"
    mock_profanity_classifier.classify.return_value.confidence = (
        0.4  # Below default threshold of 0.5
    )

    validator = ProfanityValidator()
    thought = Thought(prompt="Test prompt")
    thought.text = "This text contains mild profanity."

    result = validator.validate(thought)
    assert result is True  # Should pass because confidence is below threshold

    # Now set a lower threshold
    validator = ProfanityValidator(threshold=0.3)
    result = validator.validate(thought)
    assert result is False  # Should fail because confidence is above new threshold


def test_create_profanity_validator():
    """Test that the create_profanity_validator function creates a ProfanityValidator."""
    validator = create_profanity_validator(
        custom_words=["custom", "words"], threshold=0.7, name="CustomValidator"
    )

    assert isinstance(validator, ProfanityValidator)
    assert validator.threshold == 0.7
    assert validator.name == "CustomValidator"
