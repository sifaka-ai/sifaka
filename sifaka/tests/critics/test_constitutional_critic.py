"""
Tests for the constitutional critic.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sifaka.core.thought import Thought
from sifaka.critics.constitutional_critic import (
    DEFAULT_PRINCIPLES,
    ConstitutionalCritic,
    create_constitutional_critic,
)


def test_constitutional_critic_initialization():
    """Test that a constitutional critic can be initialized with various options."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Test with default parameters
    critic = ConstitutionalCritic(model=mock_model)
    assert critic.name == "constitutional"
    assert critic.model == mock_model
    assert critic.principles == DEFAULT_PRINCIPLES
    assert critic.temperature == 0.7

    # Test with custom principles
    custom_principles = ["Principle 1", "Principle 2"]
    critic = ConstitutionalCritic(
        model=mock_model,
        principles=custom_principles,
    )
    assert critic.principles == custom_principles

    # Test with custom system prompt
    critic = ConstitutionalCritic(
        model=mock_model,
        system_prompt="Custom system prompt",
    )
    assert critic.system_prompt == "Custom system prompt"

    # Test with custom temperature
    critic = ConstitutionalCritic(
        model=mock_model,
        temperature=0.5,
    )
    assert critic.temperature == 0.5

    # Test with custom name
    critic = ConstitutionalCritic(
        model=mock_model,
        name="CustomConstitutionalCritic",
    )
    assert critic.name == "CustomConstitutionalCritic"


def test_constitutional_critic_critique():
    """Test that critiquing text calls _critique and adds feedback to the thought."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a mock response with valid JSON
    mock_response = json.dumps(
        {
            "needs_improvement": True,
            "message": "The text violates principles.",
            "violations": ["Violation 1", "Violation 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
        }
    )

    # Create a critic with a mock _generate_with_model method
    critic = ConstitutionalCritic(model=mock_model)
    critic._generate_with_model = MagicMock(return_value=mock_response)

    # Create a thought with text
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert "The text violates principles" in feedback
    assert "Violation 1" in feedback
    assert "Violation 2" in feedback
    assert "Suggestion 1" in feedback
    assert "Suggestion 2" in feedback

    # Check the thought's critic_feedback
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "constitutional"
    assert "The text violates principles" in thought.critic_feedback[0].feedback
    assert thought.critic_feedback[0].suggestions == ["Suggestion 1", "Suggestion 2"]
    assert thought.critic_feedback[0].details["needs_improvement"] is True
    assert thought.critic_feedback[0].details["violations"] == ["Violation 1", "Violation 2"]
    assert thought.critic_feedback[0].details["principles"] == DEFAULT_PRINCIPLES


def test_constitutional_critic_critique_no_violations():
    """Test that critiquing text with no violations works correctly."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a mock response with valid JSON indicating no violations
    mock_response = json.dumps(
        {
            "needs_improvement": False,
            "message": "The text adheres to all principles.",
            "violations": [],
            "suggestions": [],
        }
    )

    # Create a critic with a mock _generate_with_model method
    critic = ConstitutionalCritic(model=mock_model)
    critic._generate_with_model = MagicMock(return_value=mock_response)

    # Create a thought with text
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert "The text adheres to all principles" in feedback

    # Check the thought's critic_feedback
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "constitutional"
    assert "The text adheres to all principles" in thought.critic_feedback[0].feedback
    assert thought.critic_feedback[0].suggestions == []
    assert thought.critic_feedback[0].details["needs_improvement"] is False
    assert thought.critic_feedback[0].details["violations"] == []


def test_constitutional_critic_critique_invalid_json():
    """Test that critiquing text handles invalid JSON responses gracefully."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a mock response with invalid JSON
    mock_response = "This is not valid JSON"

    # Create a critic with a mock _generate_with_model method
    critic = ConstitutionalCritic(model=mock_model)
    critic._generate_with_model = MagicMock(return_value=mock_response)

    # Create a thought with text
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert "Failed to parse critique response" in feedback

    # Check the thought's critic_feedback
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "constitutional"
    assert "Failed to parse critique response" in thought.critic_feedback[0].feedback
    assert "Try again with a different prompt or model" in thought.critic_feedback[0].suggestions
    assert thought.critic_feedback[0].details["needs_improvement"] is True
    # The implementation may not include raw_response in the details
    assert "message" in thought.critic_feedback[0].details
    assert "violations" in thought.critic_feedback[0].details


def test_constitutional_critic_critique_partial_json():
    """Test that critiquing text handles partial JSON responses correctly."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a mock response with JSON embedded in text
    mock_response = (
        "Here's my evaluation:\n\n"
        '{"needs_improvement": true, "message": "The text violates principles.", '
        '"violations": ["Violation 1"], "suggestions": ["Suggestion 1"]}\n\n'
        "I hope this helps!"
    )

    # Create a critic with a mock _generate_with_model method
    critic = ConstitutionalCritic(model=mock_model)
    critic._generate_with_model = MagicMock(return_value=mock_response)

    # Create a thought with text
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert "The text violates principles" in feedback
    assert "Violation 1" in feedback
    assert "Suggestion 1" in feedback

    # Check the thought's critic_feedback
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "constitutional"
    assert thought.critic_feedback[0].suggestions == ["Suggestion 1"]
    assert thought.critic_feedback[0].details["violations"] == ["Violation 1"]


def test_create_constitutional_critic():
    """Test that create_constitutional_critic creates a ConstitutionalCritic."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic using the factory function
    custom_principles = ["Principle 1", "Principle 2"]
    critic = create_constitutional_critic(
        model=mock_model,
        principles=custom_principles,
        system_prompt="Custom system prompt",
        temperature=0.5,
        name="CustomCritic",
    )

    assert isinstance(critic, ConstitutionalCritic)
    assert critic.model == mock_model
    assert critic.principles == custom_principles
    assert critic.system_prompt == "Custom system prompt"
    assert critic.temperature == 0.5
    assert critic.name == "CustomCritic"
