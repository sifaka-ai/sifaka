"""
Pytest fixtures for critic tests.
"""

import pytest
from sifaka.utils.config.critics import PromptCriticConfig
from tests.utils.mock_provider import MockProvider


@pytest.fixture
def mock_provider():
    """Fixture for a mock provider."""
    provider = MockProvider()
    return provider


@pytest.fixture
def validation_response_provider():
    """Fixture for a mock provider that returns validation responses."""
    provider = MockProvider(
        responses={
            "any_prompt": '{"valid": true, "score": 0.9, "feedback": "Good text."}'
        }
    )
    return provider


@pytest.fixture
def invalid_response_provider():
    """Fixture for a mock provider that returns invalid responses."""
    provider = MockProvider(
        responses={
            "any_prompt": '{"valid": false, "score": 0.3, "feedback": "Bad text."}'
        }
    )
    return provider


@pytest.fixture
def critique_response_provider():
    """Fixture for a mock provider that returns critique responses."""
    provider = MockProvider(
        responses={
            "any_prompt": '{"score": 0.8, "feedback": "Good text, but could be improved."}'
        }
    )
    return provider


@pytest.fixture
def improvement_response_provider():
    """Fixture for a mock provider that returns improvement responses."""
    provider = MockProvider(
        responses={
            "any_prompt": "This is an improved test text."
        }
    )
    return provider


@pytest.fixture
def prompt_critic_config():
    """Fixture for a prompt critic configuration."""
    config = PromptCriticConfig(
        system_prompt="You are a helpful critic.",
        temperature=0.7,
        max_tokens=100,
    )
    return config
