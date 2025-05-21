"""
Tests for the content validator.
"""

import pytest

from sifaka.core.thought import Thought
from sifaka.validators.content_validator import (
    ContentValidator,
    create_content_validator,
    prohibited_content,
)


def test_content_validator_initialization():
    """Test that a content validator can be initialized with various options."""
    # Test with basic prohibited list
    validator = ContentValidator(prohibited=["bad", "words"])
    assert validator.name == "ContentValidator"
    assert validator.prohibited == ["bad", "words"]
    assert validator.case_sensitive is False
    assert validator.whole_word is False
    assert validator.regex is False

    # Test with case_sensitive=True
    validator = ContentValidator(prohibited=["bad", "words"], case_sensitive=True)
    assert validator.case_sensitive is True

    # Test with whole_word=True
    validator = ContentValidator(prohibited=["bad", "words"], whole_word=True)
    assert validator.whole_word is True

    # Test with regex=True
    validator = ContentValidator(prohibited=["bad", "words"], regex=True)
    assert validator.regex is True

    # Test with custom name
    validator = ContentValidator(prohibited=["bad", "words"], name="CustomContentValidator")
    assert validator.name == "CustomContentValidator"


def test_content_validator_initialization_errors():
    """Test that initializing a content validator with invalid options raises errors."""
    # Test with empty prohibited list
    with pytest.raises(ValueError):
        ContentValidator(prohibited=[])

    # Test with invalid regex pattern when regex=True
    with pytest.raises(ValueError):
        ContentValidator(prohibited=["[invalid regex"], regex=True)


def test_content_validator_validate_empty_text():
    """Test that validating empty text passes."""
    validator = ContentValidator(prohibited=["bad", "words"])
    thought = Thought(prompt="Test prompt")
    thought.text = ""

    result = validator.validate(thought)

    assert result is False  # Empty text should fail validation
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "ContentValidator"
    assert "Empty text" in thought.validation_results[0].message


def test_content_validator_validate_no_prohibited_content():
    """Test that validating text without prohibited content passes."""
    validator = ContentValidator(prohibited=["bad", "words"])
    thought = Thought(prompt="Test prompt")
    thought.text = "This text is clean and appropriate."

    result = validator.validate(thought)

    assert result is True
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is True
    assert thought.validation_results[0].validator_name == "ContentValidator"
    assert "no prohibited content" in thought.validation_results[0].message


def test_content_validator_validate_with_prohibited_content():
    """Test that validating text with prohibited content fails."""
    validator = ContentValidator(prohibited=["bad", "inappropriate"])
    thought = Thought(prompt="Test prompt")
    thought.text = "This text contains bad words that are inappropriate."

    result = validator.validate(thought)

    assert result is False
    assert len(thought.validation_results) == 1
    assert thought.validation_results[0].passed is False
    assert thought.validation_results[0].validator_name == "ContentValidator"
    assert "contains prohibited content" in thought.validation_results[0].message
    assert "bad" in thought.validation_results[0].message
    assert "inappropriate" in thought.validation_results[0].message


def test_content_validator_case_sensitive():
    """Test that case_sensitive option works correctly."""
    # Case-insensitive (default)
    validator = ContentValidator(prohibited=["Bad"])
    thought = Thought(prompt="Test prompt")
    thought.text = "This text contains bad words."

    result = validator.validate(thought)
    assert result is False

    # Case-sensitive
    validator = ContentValidator(prohibited=["Bad"], case_sensitive=True)
    thought = Thought(prompt="Test prompt")
    thought.text = "This text contains bad words."

    result = validator.validate(thought)
    assert result is True  # Should pass because "bad" != "Bad" with case_sensitive=True

    thought.text = "This text contains Bad words."
    result = validator.validate(thought)
    assert result is False  # Should fail because "Bad" == "Bad" with case_sensitive=True


def test_content_validator_whole_word():
    """Test that whole_word option works correctly."""
    # Without whole_word (default)
    validator = ContentValidator(prohibited=["bad"])
    thought = Thought(prompt="Test prompt")
    thought.text = "This text contains badger."

    result = validator.validate(thought)
    assert result is False  # Should fail because "bad" is in "badger"

    # With whole_word=True
    validator = ContentValidator(prohibited=["bad"], whole_word=True)
    thought = Thought(prompt="Test prompt")
    thought.text = "This text contains badger."

    result = validator.validate(thought)
    assert result is True  # Should pass because "bad" is not a whole word in "badger"

    thought.text = "This text contains bad words."
    result = validator.validate(thought)
    assert result is False  # Should fail because "bad" is a whole word


def test_content_validator_regex():
    """Test that regex option works correctly."""
    # With regex=True
    validator = ContentValidator(prohibited=[r"\b\d{3}-\d{2}-\d{4}\b"], regex=True)
    thought = Thought(prompt="Test prompt")
    thought.text = "My SSN is 123-45-6789."

    result = validator.validate(thought)
    assert result is False  # Should fail because text contains a pattern matching SSN

    thought.text = "My phone number is 555-123-4567."
    result = validator.validate(thought)
    assert result is True  # Should pass because text doesn't contain SSN pattern


def test_create_content_validator():
    """Test that the create_content_validator function creates a ContentValidator."""
    validator = create_content_validator(
        prohibited=["bad", "words"],
        case_sensitive=True,
        whole_word=True,
        regex=False,
        name="CustomValidator",
    )

    assert isinstance(validator, ContentValidator)
    assert validator.prohibited == ["bad", "words"]
    assert validator.case_sensitive is True
    assert validator.whole_word is True
    assert validator.regex is False
    assert validator.name == "CustomValidator"


def test_prohibited_content():
    """Test that the prohibited_content function creates a ContentValidator."""
    validator = prohibited_content(
        prohibited=["bad", "words"],
        case_sensitive=True,
        whole_word=True,
        regex=False,
        name="CustomValidator",
    )

    assert isinstance(validator, ContentValidator)
    assert validator.prohibited == ["bad", "words"]
    assert validator.case_sensitive is True
    assert validator.whole_word is True
    assert validator.regex is False
    assert validator.name == "CustomValidator"
