"""
Tests for the model factory.
"""

import os
from unittest.mock import patch

import pytest
from core.interfaces import Model
from models.factory import (
    ModelConfigurationError,
    ModelNotFoundError,
    create_anthropic_model,
    create_mock_model,
    create_model,
    create_openai_model,
)


def test_create_model_with_provider_and_model_name() -> None:
    """Test creating a model with separate provider and model name."""
    model = create_model("mock", "test-model")
    assert isinstance(model, Model)
    assert "mock" in model.name
    assert "test-model" in model.name


def test_create_model_with_combined_format() -> None:
    """Test creating a model with combined provider:model format."""
    model = create_model("mock:test-model")
    assert isinstance(model, Model)
    assert "mock" in model.name
    assert "test-model" in model.name


def test_create_model_with_options() -> None:
    """Test creating a model with additional options."""
    model = create_model("mock", "test-model", temperature=0.5, max_tokens=100)
    assert isinstance(model, Model)
    assert "mock" in model.name
    assert "test-model" in model.name


def test_create_model_with_unknown_provider() -> None:
    """Test that creating a model with an unknown provider raises an error."""
    with pytest.raises(ModelNotFoundError):
        create_model("unknown-provider", "test-model")


@patch("models.factory.create_openai_model")
def test_create_model_calls_provider_factory(mock_create_openai_model) -> None:
    """Test that create_model calls the appropriate provider factory."""
    mock_create_openai_model.return_value = create_mock_model("test-model")

    create_model("openai", "gpt-4", temperature=0.7)

    mock_create_openai_model.assert_called_once_with("gpt-4", temperature=0.7)


@patch("models.factory.create_anthropic_model")
def test_create_model_calls_anthropic_factory(mock_create_anthropic_model) -> None:
    """Test that create_model calls the Anthropic factory."""
    mock_create_anthropic_model.return_value = create_mock_model("test-model")

    create_model("anthropic", "claude-3", temperature=0.7)

    mock_create_anthropic_model.assert_called_once_with("claude-3", temperature=0.7)


@patch("models.factory.create_gemini_model")
def test_create_model_calls_gemini_factory(mock_create_gemini_model) -> None:
    """Test that create_model calls the Gemini factory."""
    mock_create_gemini_model.return_value = create_mock_model("test-model")

    create_model("gemini", "gemini-pro", temperature=0.7)

    mock_create_gemini_model.assert_called_once_with("gemini-pro", temperature=0.7)


@patch("models.openai_model.OpenAIModel")
def test_create_openai_model(mock_openai_model) -> None:
    """Test creating an OpenAI model."""
    mock_model = create_mock_model("gpt-4")
    mock_openai_model.return_value = mock_model

    # Test with explicit API key
    create_openai_model("gpt-4", api_key="test-api-key", temperature=0.7)
    mock_openai_model.assert_called_with(
        model_name="gpt-4",
        api_key="test-api-key",
        temperature=0.7,
    )

    # Reset mock
    mock_openai_model.reset_mock()

    # Test with environment variable
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}):
        create_openai_model("gpt-4", temperature=0.7)
        mock_openai_model.assert_called_with(
            model_name="gpt-4",
            api_key="env-api-key",
            temperature=0.7,
        )


@patch("models.anthropic_model.AnthropicModel")
def test_create_anthropic_model(mock_anthropic_model) -> None:
    """Test creating an Anthropic model."""
    mock_model = create_mock_model("claude-3")
    mock_anthropic_model.return_value = mock_model

    # Test with explicit API key
    create_anthropic_model("claude-3", api_key="test-api-key", temperature=0.7)
    mock_anthropic_model.assert_called_with(
        model_name="claude-3",
        api_key="test-api-key",
        temperature=0.7,
    )

    # Reset mock
    mock_anthropic_model.reset_mock()

    # Test with environment variable
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-api-key"}):
        create_anthropic_model("claude-3", temperature=0.7)
        mock_anthropic_model.assert_called_with(
            model_name="claude-3",
            api_key="env-api-key",
            temperature=0.7,
        )


def test_create_mock_model() -> None:
    """Test creating a mock model."""
    model = create_mock_model("test-model", response_template="Custom: {prompt}")
    assert isinstance(model, Model)
    assert "mock" in model.name
    assert "test-model" in model.name
