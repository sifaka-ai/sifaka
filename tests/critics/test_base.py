"""Tests for the base Critic class."""

import pytest
from pydantic import ValidationError, Field
from abc import ABC, abstractmethod

from sifaka.critics.base import Critic


class MockCritic(Critic):
    """A mock implementation of Critic for testing."""

    def validate(self, prompt: str) -> bool:
        return True

    def critique(self, prompt: str) -> dict:
        return {
            "score": 0.9,
            "feedback": "Test feedback",
            "issues": ["Test issue"],
            "suggestions": ["Test suggestion"],
        }


def test_critic_initialization():
    """Test that a Critic can be initialized with valid parameters."""
    critic = MockCritic(
        name="test_critic", description="A test critic", min_confidence=0.7, config={"key": "value"}
    )

    assert critic.name == "test_critic"
    assert critic.description == "A test critic"
    assert critic.min_confidence == 0.7
    assert critic.config == {"key": "value"}


def test_critic_initialization_defaults():
    """Test that a Critic can be initialized with default parameters."""
    critic = MockCritic(name="test_critic", description="A test critic")

    assert critic.name == "test_critic"
    assert critic.description == "A test critic"
    assert critic.min_confidence == 0.7  # Updated to match actual default
    assert critic.config == {}  # default value


def test_critic_invalid_min_confidence():
    """Test that initializing a Critic with invalid min_confidence raises ValidationError."""
    with pytest.raises(ValidationError):
        MockCritic(
            name="test_critic",
            description="A test critic",
            min_confidence=2.0,  # Invalid value > 1.0
        )

    with pytest.raises(ValidationError):
        MockCritic(
            name="test_critic",
            description="A test critic",
            min_confidence=-0.1,  # Invalid value < 0.0
        )


def test_critic_abstract_methods():
    """Test that instantiating base Critic class raises TypeError."""

    class InvalidCritic(Critic):
        pass

    with pytest.raises(TypeError):
        InvalidCritic(name="test_critic", description="A test critic")


def test_mock_critic_validate():
    """Test that MockCritic's validate method works as expected."""
    critic = MockCritic(name="test_critic", description="A test critic")

    result = critic.validate("test prompt")
    assert isinstance(result, bool)
    assert result is True


def test_mock_critic_critique():
    """Test that MockCritic's critique method works as expected."""
    critic = MockCritic(name="test_critic", description="A test critic")

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
