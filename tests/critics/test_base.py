"""Tests for the base Critic class."""

import pytest
from pydantic import ValidationError, Field
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from sifaka.critics.base import Critic


class MockCritic(Critic):
    """A mock critic for testing."""

    def __init__(
        self,
        name: str = "mock_critic",
        description: str = "A mock critic",
        min_confidence: float = 0.7,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name, description=description, min_confidence=min_confidence, config=config or {}
        )

    def validate(self, output: str) -> bool:
        """Mock validation that validates input and returns True."""
        if not isinstance(output, str):
            raise TypeError("Output must be a string")
        if not output.strip():
            raise ValueError("Output cannot be empty")
        return True

    def critique(self, output: str) -> Dict[str, Any]:
        """Mock critique that validates input and returns a fixed response."""
        if not isinstance(output, str):
            raise TypeError("Output must be a string")
        if not output.strip():
            raise ValueError("Output cannot be empty")
        return {
            "score": 0.9,
            "feedback": "Mock feedback",
            "issues": ["Mock issue"],
            "suggestions": ["Mock suggestion"],
        }

    def improve(self, output: str, violations: Optional[List[Dict[str, Any]]] = None) -> str:
        """Mock improvement that returns an improved version of the input."""
        if not isinstance(output, str):
            raise TypeError("Output must be a string")
        if not output.strip():
            raise ValueError("Output cannot be empty")
        if not isinstance(violations, list):
            raise TypeError("Violations must be a list")

        return "Improved test output"


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


def test_mock_critic_validate_invalid_input():
    """Test that MockCritic's validate method handles invalid input correctly."""
    critic = MockCritic(name="test_critic", description="A test critic")

    with pytest.raises(TypeError):
        critic.validate(123)  # Non-string input

    with pytest.raises(ValueError):
        critic.validate("")  # Empty string


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


def test_mock_critic_critique_invalid_input():
    """Test that MockCritic's critique method handles invalid input correctly."""
    critic = MockCritic(name="test_critic", description="A test critic")

    with pytest.raises(TypeError):
        critic.critique(123)  # Non-string input

    with pytest.raises(ValueError):
        critic.critique("")  # Empty string


def test_mock_critic_improve():
    """Test that MockCritic's improve method works as expected."""
    critic = MockCritic(name="test_critic", description="A test critic")

    result = critic.improve("test output", [{"type": "test", "message": "test violation"}])
    assert isinstance(result, str)
    assert result == "Improved test output"


def test_mock_critic_improve_invalid_input():
    """Test that MockCritic's improve method handles invalid input correctly."""
    critic = MockCritic(name="test_critic", description="A test critic")

    with pytest.raises(TypeError):
        critic.improve(123, [])  # Non-string output

    with pytest.raises(TypeError):
        critic.improve("test", "not a list")  # Non-list violations

    with pytest.raises(ValueError):
        critic.improve("", [])  # Empty string output


def test_critic_config_immutability():
    """Test that the critic's config cannot be modified after initialization."""
    initial_config = {"key": "value"}
    critic = MockCritic(name="test_critic", description="A test critic", config=initial_config)

    # Attempt to modify the original config
    initial_config["key"] = "new_value"

    # Check that the critic's config remains unchanged
    assert critic.config["key"] == "value"


def test_critic_name_validation():
    """Test validation of critic names."""
    # Test with invalid names
    invalid_names = ["", " ", "\t", "\n"]
    for name in invalid_names:
        with pytest.raises(ValidationError):
            MockCritic(name=name, description="A test critic")


def test_critic_description_validation():
    """Test validation of critic descriptions."""
    # Test with invalid descriptions
    invalid_descriptions = ["", " ", "\t", "\n"]
    for description in invalid_descriptions:
        with pytest.raises(ValidationError):
            MockCritic(name="test_critic", description=description)


def test_critic_str_representation():
    """Test the string representation of critics."""
    critic = MockCritic(name="test_critic", description="A test critic")
    assert str(critic) == "MockCritic(name='test_critic')"
    assert (
        repr(critic)
        == "MockCritic(name='test_critic', description='A test critic', min_confidence=0.7, config={})"
    )


def test_critic_config_type_validation():
    """Test validation of critic config types."""
    # Test with invalid config types
    invalid_configs = ["not a dict", 123, ["list", "not", "dict"], None]
    for config in invalid_configs:
        with pytest.raises(ValidationError):
            MockCritic(name="test_critic", description="A test critic", config=config)


def test_critic_min_confidence_type_validation():
    """Test validation of min_confidence types."""
    # Test with invalid min_confidence types
    invalid_confidences = ["0.5", None, [0.5], {"confidence": 0.5}]  # string instead of float
    for confidence in invalid_confidences:
        with pytest.raises(ValidationError):
            MockCritic(name="test_critic", description="A test critic", min_confidence=confidence)


def test_critic_improve_empty_violations():
    """Test improve method with empty violations list."""
    critic = MockCritic(name="test_critic", description="A test critic")
    result = critic.improve("test output", [])
    assert result == "Improved test output"


def test_critic_improve_complex_violations():
    """Test improve method with complex violation structures."""
    critic = MockCritic(name="test_critic", description="A test critic")
    violations = [
        {"type": "error", "message": "Error message", "severity": "high"},
        {"type": "warning", "message": "Warning message", "details": {"line": 10}},
        {"type": "info", "message": "Info message", "metadata": {"source": "test"}},
    ]
    result = critic.improve("test output", violations)
    assert result == "Improved test output"


@pytest.mark.parametrize(
    "min_confidence",
    [
        0.0,  # minimum valid value
        0.5,  # middle value
        1.0,  # maximum valid value
        0.7,  # default value
        0.999,  # near maximum
        0.001,  # near minimum
    ],
)
def test_critic_valid_min_confidence_values(min_confidence):
    """Test various valid min_confidence values."""
    critic = MockCritic(
        name="test_critic", description="A test critic", min_confidence=min_confidence
    )
    assert critic.min_confidence == min_confidence


def test_critic_config_deep_copy():
    """Test that nested config structures are properly deep copied."""
    complex_config = {
        "nested": {"deep": {"value": "original"}},
        "list": ["item1", "item2"],
        "simple": "value",
    }

    critic = MockCritic(name="test_critic", description="A test critic", config=complex_config)

    # Modify the original config deeply
    complex_config["nested"]["deep"]["value"] = "modified"
    complex_config["list"].append("item3")
    complex_config["simple"] = "new_value"

    # Check that the critic's config remains unchanged
    assert critic.config["nested"]["deep"]["value"] == "original"
    assert critic.config["list"] == ["item1", "item2"]
    assert critic.config["simple"] == "value"
