"""
Tests for the MockProvider.
"""

import pytest
from tests.utils.mock_provider import MockProvider
from sifaka.utils.config.models import ModelConfig


def test_mock_provider_initialization():
    """Test that the MockProvider can be initialized."""
    provider = MockProvider(
        model_name="test-model",
        name="test_mock_provider",
        description="Test mock provider",
        config=ModelConfig(
            temperature=0.7,
            max_tokens=100,
            api_key="mock-api-key",
        ),
    )

    assert provider is not None
    assert provider.name == "test_mock_provider"
    assert provider.description == "Test mock provider"
    assert provider.model_name == "test-model"
    assert provider.config.temperature == 0.7
    assert provider.config.max_tokens == 100


def test_mock_provider_generate():
    """Test that the MockProvider generates text correctly."""
    provider = MockProvider(
        responses={"Test prompt": "Test response"},
        default_response="Default response",
    )

    # Test with a prompt that has a specific response
    response = provider.generate("Test prompt")
    assert response == "Test response"

    # Test with a prompt that uses the default response
    response = provider.generate("Unknown prompt")
    assert response == "Default response"


def test_mock_provider_count_tokens():
    """Test that the MockProvider counts tokens correctly."""
    provider = MockProvider(token_count=42)

    count = provider.count_tokens("Test text")
    assert count == 42

    # Check that statistics are updated
    stats = provider.get_statistics()
    assert stats["token_count_calls"] == 1


def test_mock_provider_tracking():
    """Test that the MockProvider tracks calls correctly."""
    provider = MockProvider()

    # Make some calls
    provider.generate("Prompt 1")
    provider.generate("Prompt 2", temperature=0.8)
    provider.generate("Prompt 3", max_tokens=50)

    # Check that calls were tracked
    calls = provider.get_calls()
    assert len(calls) == 3
    assert calls[0]["prompt"] == "Prompt 1"
    assert calls[1]["prompt"] == "Prompt 2"
    assert calls[1]["kwargs"]["temperature"] == 0.8
    assert calls[2]["prompt"] == "Prompt 3"
    assert calls[2]["kwargs"]["max_tokens"] == 50

    # Check that statistics are updated
    stats = provider.get_statistics()
    assert stats["generation_count"] == 3

    # Reset calls and check that they're cleared
    provider.reset_calls()
    assert len(provider.get_calls()) == 0


def test_mock_provider_set_response():
    """Test that the MockProvider can set responses."""
    provider = MockProvider(default_response="Default")

    # Set a specific response
    provider.set_response("Specific prompt", "Specific response")
    response = provider.generate("Specific prompt")
    assert response == "Specific response"

    # Set the default response
    provider.set_default_response("New default")
    response = provider.generate("Unknown prompt")
    assert response == "New default"


def test_mock_provider_state():
    """Test that the MockProvider manages state correctly."""
    provider = MockProvider(
        model_name="test-model", name="test_provider", description="Test description"
    )

    # Check state
    state = provider.get_state()
    assert state["initialized"] is True
    assert state["model_name"] == "test-model"
    assert state["name"] == "test_provider"
    assert state["description"] == "Test description"

    # Update config
    new_config = ModelConfig(temperature=0.9, max_tokens=200, api_key="new-api-key")
    provider.update_config(new_config)

    # Check that config was updated
    assert provider.config.temperature == 0.9
    assert provider.config.max_tokens == 200
    assert provider.config.api_key == "new-api-key"
