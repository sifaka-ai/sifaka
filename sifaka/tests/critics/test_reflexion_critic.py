"""
Tests for the reflexion critic.
"""

from unittest.mock import MagicMock, patch

import pytest

from sifaka.core.thought import Thought
from sifaka.critics.reflexion_critic import ReflexionCritic


def test_reflexion_critic_initialization():
    """Test that a reflexion critic can be initialized with various options."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Test with required parameters
    critic = ReflexionCritic(model=mock_model)
    assert critic.name == "reflexion"
    assert critic.model == mock_model

    # Test with custom prompt template
    custom_template = "Custom template: {text}\nIssues: {validation_issues}"
    critic = ReflexionCritic(
        model=mock_model,
        reflection_prompt_template=custom_template,
    )
    assert critic.reflection_prompt_template == custom_template

    # Test with custom name
    critic = ReflexionCritic(
        model=mock_model,
        name="CustomReflexionCritic",
    )
    assert critic.name == "CustomReflexionCritic"


def test_reflexion_critic_critique():
    """Test that critiquing text calls _critique and adds feedback to the thought."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"
    mock_model.generate.return_value = (
        "Here's my reflection:\n\n"
        "The text is good, but could be improved.\n\n"
        "Suggestions:\n"
        "- Add more details\n"
        "- Fix grammar issues\n"
    )

    # Create a critic
    critic = ReflexionCritic(model=mock_model)

    # Create a thought with text and validation results
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."
    thought.add_validation_result(
        validator_name="TestValidator",
        passed=False,
        message="Test validation failed",
    )

    # Critique the thought
    feedback = critic.critique(thought)

    # Check the feedback
    assert "Here's my reflection" in feedback
    assert len(thought.critic_feedback) == 1
    assert thought.critic_feedback[0].critic_name == "reflexion"
    assert "The text is good, but could be improved" in thought.critic_feedback[0].feedback
    assert len(thought.critic_feedback[0].suggestions) == 2
    assert "Add more details" in thought.critic_feedback[0].suggestions
    assert "Fix grammar issues" in thought.critic_feedback[0].suggestions


def test_reflexion_critic_critique_with_validation_issues():
    """Test that validation issues are included in the reflection prompt."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic with a mock _generate_with_model method
    critic = ReflexionCritic(model=mock_model)
    critic._generate_with_model = MagicMock(return_value="Mock reflection")

    # Create a thought with text and validation results
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."
    thought.add_validation_result(
        validator_name="TestValidator1",
        passed=False,
        message="Test validation 1 failed",
    )
    thought.add_validation_result(
        validator_name="TestValidator2",
        passed=False,
        message="Test validation 2 failed",
    )

    # Critique the thought
    critic.critique(thought)

    # Check that _generate_with_model was called with the correct prompt
    prompt = critic._generate_with_model.call_args[0][0]
    assert "This is test text." in prompt
    assert "TestValidator1: Test validation 1 failed" in prompt
    assert "TestValidator2: Test validation 2 failed" in prompt


def test_reflexion_critic_critique_without_validation_issues():
    """Test that a default message is used when there are no validation issues."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic with a mock _generate_with_model method
    critic = ReflexionCritic(model=mock_model)
    critic._generate_with_model = MagicMock(return_value="Mock reflection")

    # Create a thought with text and no validation results
    thought = Thought(prompt="Test prompt")
    thought.text = "This is test text."

    # Critique the thought
    critic.critique(thought)

    # Check that _generate_with_model was called with the correct prompt
    prompt = critic._generate_with_model.call_args[0][0]
    assert "This is test text." in prompt
    assert "No specific validation issues identified" in prompt


def test_reflexion_critic_extract_suggestions():
    """Test that _extract_suggestions correctly extracts suggestions from reflection."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic
    critic = ReflexionCritic(model=mock_model)

    # Test with bullet points
    reflection = (
        "Here's my reflection:\n\n"
        "The text is good, but could be improved.\n\n"
        "Suggestions:\n"
        "- Add more details\n"
        "- Fix grammar issues\n"
    )
    suggestions = critic._extract_suggestions(reflection)
    assert len(suggestions) == 2
    assert "Add more details" in suggestions
    assert "Fix grammar issues" in suggestions

    # Test with asterisks
    reflection = (
        "Here's my reflection:\n\n"
        "The text is good, but could be improved.\n\n"
        "Recommendations:\n"
        "* Add more details\n"
        "* Fix grammar issues\n"
    )
    suggestions = critic._extract_suggestions(reflection)
    assert len(suggestions) == 2
    assert "Add more details" in suggestions
    assert "Fix grammar issues" in suggestions

    # Test with no bullet points but suggestion keywords
    reflection = (
        "Here's my reflection:\n\n"
        "The text is good, but could be improved.\n\n"
        "You should add more details.\n"
        "Consider fixing grammar issues.\n"
    )
    suggestions = critic._extract_suggestions(reflection)
    # The implementation now extracts all sentences with suggestion keywords
    assert len(suggestions) >= 2
    assert "You should add more details." in suggestions
    assert "Consider fixing grammar issues." in suggestions


def test_create_reflexion_critic():
    """Test that create_reflexion_critic creates a ReflexionCritic."""
    # Create a mock model
    mock_model = MagicMock()
    mock_model.name = "mock_model"

    # Create a critic using the factory function
    critic = ReflexionCritic(
        model=mock_model,
        reflection_prompt_template="Custom template",
        name="CustomCritic",
    )

    assert isinstance(critic, ReflexionCritic)
    assert critic.model == mock_model
    assert critic.reflection_prompt_template == "Custom template"
    assert critic.name == "CustomCritic"
