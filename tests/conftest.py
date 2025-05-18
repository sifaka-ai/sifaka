"""
Pytest configuration for Sifaka tests.

This module contains fixtures and configuration for testing the Sifaka framework.
"""

import os
import pytest
from typing import Dict, Any, Optional

from sifaka.models.base import Model
from sifaka.results import ValidationResult, ImprovementResult


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_name: str = "mock-model", **options: Any):
        """Initialize the mock model."""
        self.model_name = model_name
        self.options = options
        self.generate_calls = []
        self.count_tokens_calls = []
        self.response = "This is a mock response."

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt."""
        self.generate_calls.append((prompt, options))
        return self.response

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        self.count_tokens_calls.append(text)
        return len(text.split())

    def configure(self, **options: Any) -> None:
        """Configure the model with new options."""
        self.options.update(options)

    def set_response(self, response: str) -> None:
        """Set the response for the mock model."""
        self.response = response


class MockValidator:
    """Mock validator for testing."""

    def __init__(self, name: str = "MockValidator", should_pass: bool = True):
        """Initialize the mock validator."""
        self._name = name
        self.should_pass = should_pass
        self.validate_calls = []

    @property
    def name(self) -> str:
        """Get the name of the validator."""
        return self._name

    def validate(self, text: str) -> ValidationResult:
        """Validate text against specific criteria."""
        self.validate_calls.append(text)
        if self.should_pass:
            return ValidationResult(
                passed=True,
                message="Validation passed",
                score=1.0
            )
        else:
            return ValidationResult(
                passed=False,
                message="Validation failed",
                score=0.0,
                issues=["Mock issue"],
                suggestions=["Mock suggestion"]
            )


class MockCritic:
    """Mock critic for testing."""

    def __init__(self, name: str = "MockCritic", should_improve: bool = True):
        """Initialize the mock critic."""
        self._name = name
        self.should_improve = should_improve
        self.improve_calls = []
        self.improved_text = "This is improved text."

    @property
    def name(self) -> str:
        """Get the name of the critic."""
        return self._name

    def improve(self, text: str) -> tuple[str, ImprovementResult]:
        """Improve text using this critic."""
        self.improve_calls.append(text)
        if self.should_improve:
            return self.improved_text, ImprovementResult(
                _original_text=text,
                _improved_text=self.improved_text,
                _changes_made=True,
                message="Text improved",
                processing_time_ms=100.0
            )
        else:
            return text, ImprovementResult(
                _original_text=text,
                _improved_text=text,
                _changes_made=False,
                message="No improvements needed",
                processing_time_ms=100.0
            )

    def set_improved_text(self, text: str) -> None:
        """Set the improved text for the mock critic."""
        self.improved_text = text


@pytest.fixture
def mock_model() -> MockModel:
    """Fixture for a mock model."""
    return MockModel()


@pytest.fixture
def mock_validator(request) -> MockValidator:
    """Fixture for a mock validator.
    
    Args:
        request: The pytest request object, which can be used to pass parameters.
            Use `@pytest.mark.parametrize("mock_validator", [True, False], indirect=True)`
            to control whether the validator should pass or fail.
    """
    should_pass = True
    if hasattr(request, "param"):
        should_pass = request.param
    return MockValidator(should_pass=should_pass)


@pytest.fixture
def mock_critic(request) -> MockCritic:
    """Fixture for a mock critic.
    
    Args:
        request: The pytest request object, which can be used to pass parameters.
            Use `@pytest.mark.parametrize("mock_critic", [True, False], indirect=True)`
            to control whether the critic should improve the text or not.
    """
    should_improve = True
    if hasattr(request, "param"):
        should_improve = request.param
    return MockCritic(should_improve=should_improve)


@pytest.fixture
def env_vars() -> Dict[str, str]:
    """Fixture for environment variables."""
    return {
        "OPENAI_API_KEY": "test-openai-api-key",
        "ANTHROPIC_API_KEY": "test-anthropic-api-key",
        "GOOGLE_API_KEY": "test-google-api-key",
    }


@pytest.fixture
def set_env_vars(env_vars: Dict[str, str]) -> None:
    """Fixture to set environment variables for testing."""
    original_vars = {}
    for key, value in env_vars.items():
        if key in os.environ:
            original_vars[key] = os.environ[key]
        os.environ[key] = value
    
    yield
    
    # Restore original environment variables
    for key in env_vars:
        if key in original_vars:
            os.environ[key] = original_vars[key]
        else:
            del os.environ[key]
