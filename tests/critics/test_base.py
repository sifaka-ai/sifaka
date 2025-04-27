"""Tests for base critic functionality."""

import pytest
from typing import Dict, Any
from pydantic import ValidationError
from sifaka.critics.base import Critic
from sifaka.critics.protocols import TextValidator, TextCritic, TextImprover


class MockValidator(TextValidator):
    """Mock validator for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.validate_called = 0

    def validate(self, text: str) -> bool:
        self.validate_called += 1
        if self.should_fail:
            raise ValueError("Validation failed")
        return len(text) > 10

    async def validate(self, text: str) -> bool:
        self.validate_called += 1
        if self.should_fail:
            raise ValueError("Validation failed")
        return len(text) > 10


class MockTextCritic(TextCritic):
    """Mock critic for testing."""

    def __init__(self, response: Dict[str, Any] = None):
        self.critique_called = 0
        self.response = response or {
            "score": 0.8,
            "feedback": "Mock feedback",
            "issues": ["Mock issue"],
            "suggestions": ["Mock suggestion"],
        }

    def critique(self, text: str) -> dict:
        self.critique_called += 1
        return self.response

    async def critique(self, text: str) -> dict:
        self.critique_called += 1
        return self.response


class MockImprover(TextImprover):
    """Mock improver for testing."""

    def __init__(self, should_fail: bool = False):
        self.improve_called = 0
        self.should_fail = should_fail

    def improve(self, text: str, feedback: str) -> str:
        self.improve_called += 1
        if self.should_fail:
            raise ValueError("Improvement failed")
        return f"Improved: {text}"

    async def improve(self, text: str, feedback: str) -> str:
        self.improve_called += 1
        if self.should_fail:
            raise ValueError("Improvement failed")
        return f"Improved: {text}"


class TestCritic(Critic):
    """Test implementation of Critic."""

    def __init__(self, **data):
        super().__init__(**data)
        self.validator = MockValidator()
        self.critic = MockTextCritic()
        self.improver = MockImprover()


# Basic initialization tests
def test_critic_initialization():
    """Test that a Critic can be initialized with valid parameters."""
    critic = TestCritic(
        name="test_critic", description="A test critic", min_confidence=0.7, config={"key": "value"}
    )
    assert critic.name == "test_critic"
    assert critic.description == "A test critic"
    assert critic.min_confidence == 0.7
    assert critic.config == {"key": "value"}


def test_critic_initialization_defaults():
    """Test that a Critic can be initialized with default parameters."""
    critic = TestCritic(name="test_critic", description="A test critic")
    assert critic.name == "test_critic"
    assert critic.description == "A test critic"
    assert critic.min_confidence == 0.7  # Default value
    assert critic.config == {}  # Default empty dict


# Validation tests
def test_critic_invalid_min_confidence():
    """Test that initializing a Critic with invalid min_confidence raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TestCritic(
            name="test_critic",
            description="A test critic",
            min_confidence=2.0,  # Invalid value > 1.0
        )
    assert "Input should be less than or equal to 1.0" in str(exc_info.value)


def test_critic_invalid_name():
    """Test that initializing a Critic with invalid name raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TestCritic(name="", description="A test critic")  # Invalid empty name
    assert "String should have at least 1 character" in str(exc_info.value)


def test_critic_invalid_description():
    """Test that initializing a Critic with invalid description raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TestCritic(name="test_critic", description="")  # Invalid empty description
    assert "String should have at least 1 character" in str(exc_info.value)


# Config validation tests
@pytest.mark.parametrize(
    "invalid_config",
    [
        "not a dict",
        123,
        ["list", "not", "allowed"],
        {"key": complex(1, 2)},  # Invalid value type
        {"key": object()},  # Invalid value type
    ],
)
def test_critic_invalid_config(invalid_config):
    """Test that initializing a Critic with invalid config raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TestCritic(name="test_critic", description="A test critic", config=invalid_config)
    assert "Config must be a dictionary with valid types" in str(exc_info.value)


# Component validation tests
def test_critic_validate():
    """Test that Critic's validate method works as expected."""
    critic = TestCritic(name="test_critic", description="A test critic")

    # Test basic validation
    assert critic.validate("long enough text") is True
    assert critic.validate("short") is False
    assert critic.validator.validate_called == 2

    # Test validation with failing validator
    critic.validator = MockValidator(should_fail=True)
    with pytest.raises(ValueError, match="Validation failed"):
        critic.validate("test text")


def test_critic_critique():
    """Test that Critic's critique method works as expected."""
    critic = TestCritic(name="test_critic", description="A test critic")

    # Test basic critique
    result = critic.critique("test text")
    assert isinstance(result, dict)
    assert result["score"] == 0.8
    assert result["feedback"] == "Mock feedback"
    assert result["issues"] == ["Mock issue"]
    assert result["suggestions"] == ["Mock suggestion"]
    assert critic.critic.critique_called == 1

    # Test critique with custom response
    custom_response = {
        "score": 0.9,
        "feedback": "Custom feedback",
        "issues": [],
        "suggestions": ["Good as is"],
    }
    critic.critic = MockTextCritic(response=custom_response)
    result = critic.critique("perfect text")
    assert result == custom_response


def test_critic_improve():
    """Test that Critic's improve method works as expected."""
    critic = TestCritic(name="test_critic", description="A test critic")

    # Test basic improvement
    improved = critic.improve("test text", "needs improvement")
    assert improved == "Improved: test text"
    assert critic.improver.improve_called == 1

    # Test improvement failure
    critic.improver = MockImprover(should_fail=True)
    with pytest.raises(ValueError, match="Improvement failed"):
        critic.improve("test text", "needs improvement")


# Async tests
@pytest.mark.asyncio
async def test_critic_async_methods():
    """Test async versions of Critic methods."""
    critic = TestCritic(name="test_critic", description="A test critic")

    # Test async validate
    assert await critic.avalidate("long enough text") is True
    assert await critic.avalidate("short") is False
    assert critic.validator.validate_called == 2

    # Test async critique
    result = await critic.acritique("test text")
    assert isinstance(result, dict)
    assert result["score"] == 0.8
    assert critic.critic.critique_called == 1

    # Test async improve
    improved = await critic.aimprove("test text", "needs improvement")
    assert improved == "Improved: test text"
    assert critic.improver.improve_called == 1


@pytest.mark.asyncio
async def test_critic_async_failures():
    """Test async methods with failures."""
    critic = TestCritic(name="test_critic", description="A test critic")

    # Test async validate failure
    critic.validator = MockValidator(should_fail=True)
    with pytest.raises(ValueError, match="Validation failed"):
        await critic.avalidate("test text")

    # Test async improve failure
    critic.improver = MockImprover(should_fail=True)
    with pytest.raises(ValueError, match="Improvement failed"):
        await critic.aimprove("test text", "needs improvement")


# Context manager tests
def test_critic_context_manager():
    """Test that Critic works as a context manager."""
    with TestCritic(name="test_critic", description="A test critic") as critic:
        assert critic.name == "test_critic"
        result = critic.critique("test text")
        assert isinstance(result, dict)
        assert critic.critic.critique_called == 1


# String representation tests
def test_critic_str_representation():
    """Test the string representation of a Critic."""
    critic = TestCritic(name="test_critic", description="A test critic")
    assert str(critic) == "TestCritic(name='test_critic')"


def test_critic_repr_representation():
    """Test the repr representation of a Critic."""
    critic = TestCritic(name="test_critic", description="A test critic")
    assert repr(critic) == (
        "TestCritic("
        "name='test_critic', "
        "description='A test critic', "
        "min_confidence=0.7, "
        "config={})"
    )


# Missing components tests
def test_critic_missing_components():
    """Test that Critic raises appropriate errors when components are missing."""
    critic = Critic(name="test_critic", description="A test critic")

    with pytest.raises(RuntimeError, match="No validator configured"):
        critic.validate("test")

    with pytest.raises(RuntimeError, match="No critic configured"):
        critic.critique("test")

    with pytest.raises(RuntimeError, match="No improver configured"):
        critic.improve("test", "feedback")
