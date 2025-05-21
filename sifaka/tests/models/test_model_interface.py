"""
Tests for the model interface.
"""

from typing import Any, Dict, Protocol

import pytest
from core.interfaces import Model
from core.thought import Thought
from models.mock_model import MockModel


class ModelImplementation(Protocol):
    """Protocol for model implementations to test."""

    def __call__(self, **kwargs: Any) -> Model:
        """Create a model instance."""
        ...


@pytest.fixture
def thought() -> Thought:
    """Create a thought for testing."""
    return Thought(prompt="Test prompt")


@pytest.fixture
def thought_with_context() -> Thought:
    """Create a thought with context for testing."""
    thought = Thought(prompt="Test prompt with context")
    thought.add_retrieved_context(
        source="test_source",
        content="test_context",
        metadata={"key": "value"},
        relevance_score=0.8,
    )
    return thought


@pytest.fixture
def thought_with_validation() -> Thought:
    """Create a thought with validation results for testing."""
    thought = Thought(prompt="Test prompt with validation")
    thought.add_validation_result(
        validator_name="test_validator",
        passed=False,
        score=0.5,
        details={"key": "value"},
        message="Test validation message",
    )
    return thought


@pytest.fixture
def thought_with_feedback() -> Thought:
    """Create a thought with critic feedback for testing."""
    thought = Thought(prompt="Test prompt with feedback")
    thought.add_critic_feedback(
        critic_name="test_critic",
        feedback="Test feedback",
        suggestions=["Suggestion 1", "Suggestion 2"],
        details={"key": "value"},
    )
    return thought


@pytest.fixture
def thought_with_history() -> Thought:
    """Create a thought with history for testing."""
    original_thought = Thought(prompt="Original prompt", text="Original text")
    new_thought = original_thought.create_new_version("New text")
    return new_thought


@pytest.fixture
def mock_model_implementation() -> ModelImplementation:
    """Create a mock model implementation for testing."""

    def create_model(**kwargs: Any) -> Model:
        return MockModel(**kwargs)

    return create_model


def test_model_interface_basic(
    mock_model_implementation: ModelImplementation, thought: Thought
) -> None:
    """Test that a model implements the basic interface."""
    model = mock_model_implementation()

    # Test name property
    assert hasattr(model, "name")
    assert isinstance(model.name, str)

    # Test generate method
    assert hasattr(model, "generate")
    result = model.generate(thought)
    assert isinstance(result, str)
    assert "Test prompt" in result


def test_model_with_context(
    mock_model_implementation: ModelImplementation, thought_with_context: Thought
) -> None:
    """Test that a model can handle thoughts with context."""
    model = mock_model_implementation()
    result = model.generate(thought_with_context)
    assert isinstance(result, str)
    assert "Test prompt with context" in result
    assert "Context" in result
    assert "test_source" in result


def test_model_with_validation(
    mock_model_implementation: ModelImplementation, thought_with_validation: Thought
) -> None:
    """Test that a model can handle thoughts with validation results."""
    model = mock_model_implementation()
    result = model.generate(thought_with_validation)
    assert isinstance(result, str)
    assert "Test prompt with validation" in result
    assert "Validation" in result
    assert "test_validator" in result


def test_model_with_feedback(
    mock_model_implementation: ModelImplementation, thought_with_feedback: Thought
) -> None:
    """Test that a model can handle thoughts with critic feedback."""
    model = mock_model_implementation()
    result = model.generate(thought_with_feedback)
    assert isinstance(result, str)
    assert "Test prompt with feedback" in result
    assert "Feedback" in result
    assert "test_critic" in result


def test_model_with_history(
    mock_model_implementation: ModelImplementation, thought_with_history: Thought
) -> None:
    """Test that a model can handle thoughts with history."""
    model = mock_model_implementation()
    result = model.generate(thought_with_history)
    assert isinstance(result, str)
    assert "Original prompt" in result


def test_model_with_custom_response_template(
    mock_model_implementation: ModelImplementation, thought: Thought
) -> None:
    """Test that a model can be configured with a custom response template."""
    model = mock_model_implementation(response_template="Custom response: {prompt}")
    result = model.generate(thought)
    assert isinstance(result, str)
    assert "Custom response: Test prompt" in result
