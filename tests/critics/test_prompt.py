"""Tests for the PromptCritic class."""

import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from sifaka.critics.prompt import PromptCritic


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.generate.return_value = {
        "score": 0.85,
        "feedback": "The prompt is clear and well-structured",
        "issues": ["Could be more concise"],
        "suggestions": ["Consider removing redundant words"],
    }
    return model


def test_prompt_critic_initialization(mock_model):
    """Test that a PromptCritic can be initialized with valid parameters."""
    critic = PromptCritic(
        name="test_prompt_critic",
        description="A test prompt critic",
        model=mock_model,
        min_confidence=0.7,
        config={"key": "value"},
    )

    assert critic.name == "test_prompt_critic"
    assert critic.description == "A test prompt critic"
    assert critic.model == mock_model
    assert critic.min_confidence == 0.7
    assert critic.config == {"key": "value"}


def test_prompt_critic_initialization_defaults(mock_model):
    """Test that a PromptCritic can be initialized with default parameters."""
    critic = PromptCritic(
        name="test_prompt_critic", description="A test prompt critic", model=mock_model
    )

    assert critic.name == "test_prompt_critic"
    assert critic.description == "A test prompt critic"
    assert critic.model == mock_model
    assert critic.min_confidence == 0.7  # Updated to match actual default
    assert critic.config == {}  # default value


def test_prompt_critic_missing_model():
    """Test that initializing a PromptCritic without a model raises ValidationError."""
    with pytest.raises(ValidationError):
        PromptCritic(name="test_prompt_critic", description="A test prompt critic")


def test_prompt_critic_critique(mock_model):
    """Test that PromptCritic's critique method works as expected."""
    critic = PromptCritic(
        name="test_prompt_critic", description="A test prompt critic", model=mock_model
    )

    result = critic.critique("test prompt")
    assert isinstance(result, dict)
    assert "score" in result
    assert "feedback" in result
    assert "issues" in result
    assert "suggestions" in result
    assert isinstance(result["score"], float)
    assert isinstance(result["feedback"], str)
    assert isinstance(result["issues"], list)
    assert isinstance(result["suggestions"], list)


def test_prompt_critic_validate(mock_model):
    """Test that PromptCritic's validate method works as expected."""
    critic = PromptCritic(
        name="test_prompt_critic",
        description="A test prompt critic",
        model=mock_model,
        min_confidence=0.8,
    )

    # Should pass validation (mock returns 0.85)
    assert critic.validate("good prompt") is True

    # Should fail validation
    mock_model.generate.return_value = {
        "score": 0.7,
        "feedback": "Not good enough",
        "issues": ["Issue"],
        "suggestions": ["Suggestion"],
    }
    assert critic.validate("bad prompt") is False


def test_prompt_critic_validate_invalid_response(mock_model):
    """Test that PromptCritic's validate method handles invalid model responses."""
    critic = PromptCritic(
        name="test_prompt_critic", description="A test prompt critic", model=mock_model
    )

    # Test with missing required keys
    mock_model.generate.return_value = {"score": 0.8}
    with pytest.raises(KeyError) as exc_info:
        critic.validate("test prompt")
    assert "missing required keys" in str(exc_info.value)

    # Test with invalid score type
    mock_model.generate.return_value = {
        "score": "not a number",
        "feedback": "Test feedback",
        "issues": [],
        "suggestions": [],
    }
    with pytest.raises(TypeError) as exc_info:
        critic.validate("test prompt")
    assert "Score must be a number" in str(exc_info.value)

    # Test with invalid response type
    mock_model.generate.return_value = "not a dict"
    with pytest.raises(TypeError) as exc_info:
        critic.validate("test prompt")
    assert "Model response must be a dictionary" in str(exc_info.value)

    # Test with empty prompt
    with pytest.raises(ValueError) as exc_info:
        critic.validate("")
    assert "Prompt cannot be empty" in str(exc_info.value)

    # Test with non-string prompt
    with pytest.raises(TypeError) as exc_info:
        critic.validate(123)
    assert "Prompt must be a string" in str(exc_info.value)
