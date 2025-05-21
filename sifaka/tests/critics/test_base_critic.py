"""
Tests for the base critic.
"""

from unittest.mock import MagicMock, patch

import pytest

from sifaka.core.thought import Thought
from sifaka.critics.base_critic import BaseCritic


class MockCritic(BaseCritic):
    """Mock critic for testing BaseCritic functionality."""

    def _critique(self, thought):
        """Mock implementation of _critique."""
        return {
            "feedback": "This is mock feedback.",
            "suggestions": ["Suggestion 1", "Suggestion 2"],
            "details": {"mock_detail": "value"},
        }


def test_base_critic_initialization():
    """Test that a base critic can be initialized with various options."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Test with required parameters
    critic = MockCritic(model=mock_model)
    assert critic.name == "MockCritic"
    assert critic.model == mock_model
    assert critic.temperature == 0.7
    assert critic.system_prompt is None

    # Test with optional parameters
    critic = MockCritic(
        model=mock_model,
        system_prompt="Test system prompt",
        temperature=0.5,
        name="CustomCritic",
        option1="value1",
    )
    assert critic.name == "CustomCritic"
    assert critic.model == mock_model
    assert critic.system_prompt == "Test system prompt"
    assert critic.temperature == 0.5
    assert critic.options["option1"] == "value1"


def test_base_critic_initialization_errors():
    """Test that initializing a base critic with invalid options raises errors."""
    # Test with no model
    with pytest.raises(ValueError):
        MockCritic(model=None)


def test_base_critic_critique_empty_text():
    """Test that critiquing empty text returns appropriate feedback."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic
    critic = MockCritic(model=mock_model)

    # Create a thought with empty text
    thought = Thought(prompt="Test prompt")
    thought.text = ""

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert "Empty text cannot be critiqued" in feedback
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "MockCritic"
    assert "Empty text cannot be critiqued" in thought.critic_feedback[0].feedback
    assert thought.critic_feedback[0].suggestions == []
    assert "EmptyText" in thought.critic_feedback[0].details["error_type"]


def test_base_critic_critique():
    """Test that critiquing text calls _critique and adds feedback to the thought."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic
    critic = MockCritic(model=mock_model)

    # Create a thought with text
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert feedback == "This is mock feedback."
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "MockCritic"
    assert thought.critic_feedback[0].feedback == "This is mock feedback."
    assert thought.critic_feedback[0].suggestions == ["Suggestion 1", "Suggestion 2"]
    assert thought.critic_feedback[0].details["mock_detail"] == "value"
    assert "processing_time_ms" in thought.critic_feedback[0].details


def test_base_critic_critique_error():
    """Test that critiquing text handles errors gracefully."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic with a _critique method that raises an exception
    critic = MockCritic(model=mock_model)
    critic._critique = MagicMock(side_effect=RuntimeError("Test error"))

    # Create a thought with text
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert "Error critiquing text" in feedback
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "MockCritic"
    assert "Error critiquing text" in thought.critic_feedback[0].feedback
    assert thought.critic_feedback[0].suggestions == []
    assert thought.critic_feedback[0].details["error_type"] == "RuntimeError"
    assert thought.critic_feedback[0].details["error_message"] == "Test error"


def test_base_critic_generate_with_model():
    """Test that _generate_with_model calls the model's generate method."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"
    mock_model.generate.return_value = "Generated text"

    # Create a critic
    critic = MockCritic(model=mock_model)

    # Call _generate_with_model
    result = critic._generate_with_model("Test prompt")

    # Check that the model's generate method was called
    mock_model.generate.assert_called_once()
    assert result == "Generated text"


def test_base_critic_generate_with_model_error():
    """Test that _generate_with_model handles errors gracefully."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"
    mock_model.generate.side_effect = RuntimeError("Test error")

    # Create a critic
    critic = MockCritic(model=mock_model)

    # Call _generate_with_model
    with pytest.raises(RuntimeError):
        critic._generate_with_model("Test prompt")
