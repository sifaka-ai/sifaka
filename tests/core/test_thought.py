"""
Tests for the Thought class.
"""

import pytest

from sifaka.core.thought import Thought, ValidationResult, CriticFeedback, RetrievedContext


def test_thought_initialization():
    """Test that a Thought can be initialized with a prompt."""
    thought = Thought(prompt="Test prompt")
    assert thought.prompt == "Test prompt"
    assert thought.text == ""
    assert thought.retrieved_context == []
    assert thought.validation_results == []
    assert thought.validation_passed is False
    assert thought.critic_feedback == []
    assert thought.metadata == {}
    assert thought.history == []


def test_add_retrieved_context():
    """Test that retrieved context can be added to a Thought."""
    thought = Thought(prompt="Test prompt")
    thought.add_retrieved_context(
        source="test_source",
        content="test_content",
        metadata={"key": "value"},
        relevance_score=0.8,
    )

    assert len(thought.retrieved_context) == 1
    context = thought.retrieved_context[0]
    assert context.source == "test_source"
    assert context.content == "test_content"
    assert context.metadata == {"key": "value"}
    assert context.relevance_score == 0.8


def test_add_validation_result():
    """Test that validation results can be added to a Thought."""
    thought = Thought(prompt="Test prompt")
    thought.add_validation_result(
        validator_name="test_validator",
        passed=True,
        score=0.9,
        details={"key": "value"},
        message="Test message",
    )

    assert len(thought.validation_results) == 1
    result = thought.validation_results[0]
    assert result.validator_name == "test_validator"
    assert result.passed is True
    assert result.score == 0.9
    assert result.details == {"key": "value"}
    assert result.message == "Test message"


def test_add_critic_feedback():
    """Test that critic feedback can be added to a Thought."""
    thought = Thought(prompt="Test prompt")
    thought.add_critic_feedback(
        critic_name="test_critic",
        feedback="test_feedback",
        suggestions=["suggestion1", "suggestion2"],
        details={"key": "value"},
    )

    assert len(thought.critic_feedback) == 1
    feedback = thought.critic_feedback[0]
    assert feedback.critic_name == "test_critic"
    assert feedback.feedback == "test_feedback"
    assert feedback.suggestions == ["suggestion1", "suggestion2"]
    assert feedback.details == {"key": "value"}


def test_update_validation_status():
    """Test that validation status is updated correctly."""
    thought = Thought(prompt="Test prompt")

    # Add a passing validation result
    thought.add_validation_result(
        validator_name="test_validator1",
        passed=True,
    )

    # Update validation status
    assert thought.update_validation_status() is True
    assert thought.validation_passed is True

    # Add a failing validation result
    thought.add_validation_result(
        validator_name="test_validator2",
        passed=False,
    )

    # Update validation status
    assert thought.update_validation_status() is False
    assert thought.validation_passed is False


def test_create_new_version():
    """Test that a new version of a Thought can be created."""
    thought = Thought(prompt="Test prompt", text="Original text")
    thought.add_validation_result(
        validator_name="test_validator",
        passed=False,
    )
    thought.add_critic_feedback(
        critic_name="test_critic",
        feedback="test_feedback",
    )

    # Create new version
    new_thought = thought.create_new_version("New text")

    # Check that the new version has the same prompt but new text
    assert new_thought.prompt == "Test prompt"
    assert new_thought.text == "New text"

    # Check that validation results and critic feedback are reset
    assert new_thought.validation_results == []
    assert new_thought.critic_feedback == []

    # Check that history contains the original thought
    assert len(new_thought.history) == 1
    assert new_thought.history[0].prompt == "Test prompt"
    assert new_thought.history[0].text == "Original text"
    assert len(new_thought.history[0].validation_results) == 1
    assert len(new_thought.history[0].critic_feedback) == 1


def test_to_dict_and_from_dict():
    """Test that a Thought can be converted to and from a dictionary."""
    thought = Thought(prompt="Test prompt", text="Test text")
    thought.add_retrieved_context(
        source="test_source",
        content="test_content",
    )
    thought.add_validation_result(
        validator_name="test_validator",
        passed=True,
    )
    thought.add_critic_feedback(
        critic_name="test_critic",
        feedback="test_feedback",
    )

    # Convert to dictionary
    data = thought.to_dict()

    # Convert back to Thought
    new_thought = Thought.from_dict(data)

    # Check that the new Thought is equivalent to the original
    assert new_thought.prompt == thought.prompt
    assert new_thought.text == thought.text
    assert len(new_thought.retrieved_context) == len(thought.retrieved_context)
    assert len(new_thought.validation_results) == len(thought.validation_results)
    assert len(new_thought.critic_feedback) == len(thought.critic_feedback)

    # Check specific fields
    assert new_thought.retrieved_context[0].source == "test_source"
    assert new_thought.retrieved_context[0].content == "test_content"
    assert new_thought.validation_results[0].validator_name == "test_validator"
    assert new_thought.validation_results[0].passed is True
    assert new_thought.critic_feedback[0].critic_name == "test_critic"
    assert new_thought.critic_feedback[0].feedback == "test_feedback"
