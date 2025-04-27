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
    # Configure mock to return a properly formatted string response
    mock_model.generate.return_value = """
    SCORE: 0.85
    FEEDBACK: Good prompt with clear instructions
    ISSUES:
    - Could be more specific
    - Missing context
    SUGGESTIONS:
    - Add more specific requirements
    - Include relevant context
    """

    critic = PromptCritic(
        name="test_prompt_critic", description="A test prompt critic", model=mock_model
    )

    result = critic.critique("test prompt")
    assert isinstance(result, dict)
    assert "score" in result
    assert "feedback" in result
    assert "issues" in result
    assert "suggestions" in result


def test_prompt_critic_validate(mock_model):
    """Test that PromptCritic's validate method works as expected."""
    # Configure mock to return a properly formatted string response
    mock_model.generate.return_value = """
    SCORE: 0.85
    FEEDBACK: Good prompt with clear instructions
    ISSUES:
    - Minor clarity issues
    SUGGESTIONS:
    - Add more detail
    """

    critic = PromptCritic(
        name="test_prompt_critic",
        description="A test prompt critic",
        model=mock_model,
        min_confidence=0.8,
    )

    assert critic.validate("good prompt") is True


def test_prompt_critic_validate_invalid_response(mock_model):
    """Test that PromptCritic's validate method handles invalid model responses."""
    # Configure mock to return a properly formatted string response
    mock_model.generate.return_value = """
    SCORE: invalid
    FEEDBACK: Invalid score format
    ISSUES:
    - Score not numeric
    SUGGESTIONS:
    - Fix score format
    """

    critic = PromptCritic(
        name="test_prompt_critic", description="A test prompt critic", model=mock_model
    )

    with pytest.raises(ValueError) as exc_info:
        critic.validate("test prompt")
    assert "Invalid score format" in str(exc_info.value)
