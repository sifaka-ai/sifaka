"""
Pytest fixtures for model tests.
"""

import pytest
from sifaka.utils.config.models import ModelConfig


@pytest.fixture
def model_config():
    """Fixture for a model configuration."""
    config = ModelConfig(
        model="test-model",
        temperature=0.7,
        max_tokens=100,
    )
    return config


@pytest.fixture
def test_prompt():
    """Fixture for a test prompt."""
    return "This is a test prompt."


@pytest.fixture
def test_response():
    """Fixture for a test response."""
    return "This is a test response."


@pytest.fixture
def mock_responses():
    """Fixture for mock responses."""
    return {
        "Test prompt 1": "Test response 1",
        "Test prompt 2": "Test response 2",
    }
