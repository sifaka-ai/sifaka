"""
Tests for the Anthropic model implementation.

This module contains tests for the Anthropic model implementation in Sifaka.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from sifaka.errors import ConfigurationError, ModelAPIError, ModelError


# Mock the anthropic module for testing
class MockAnthropicModule:
    """Mock for the anthropic module."""

    class Anthropic:
        """Mock for the Anthropic class."""

        def __init__(self, api_key=None):
            self.api_key = api_key
            # Create a mock messages object with mock methods
            self.messages = MagicMock()
            # Set up the create method as a MagicMock
            self.messages.create = MagicMock()
            # Set up the count_tokens method as a MagicMock
            self.messages.count_tokens = MagicMock()

    class APIError(Exception):
        """Mock for the APIError class."""

    class RateLimitError(Exception):
        """Mock for the RateLimitError class."""

    class APIConnectionError(Exception):
        """Mock for the APIConnectionError class."""


# Patch the anthropic module
@pytest.fixture
def mock_anthropic():
    """Fixture to mock the anthropic module."""
    mock_module = MockAnthropicModule()
    with patch.dict("sys.modules", {"anthropic": mock_module}):
        # Set the ANTHROPIC_AVAILABLE flag to True
        with patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True):
            # Also patch the Anthropic class directly
            with patch("sifaka.models.anthropic.Anthropic", mock_module.Anthropic):
                yield mock_module


class TestAnthropicModel:
    """Tests for the AnthropicModel class."""

    def test_init_without_anthropic_package(self):
        """Test initialization when the anthropic package is not available."""
        # Import here to avoid early import errors
        from sifaka.models.anthropic import AnthropicModel

        # Patch ANTHROPIC_AVAILABLE to False
        with patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", False):
            with pytest.raises(ConfigurationError) as excinfo:
                AnthropicModel("claude-2", api_key="test-key")

            assert "Anthropic package not installed" in str(excinfo.value)
            assert "pip install anthropic" in str(excinfo.value)

    def test_init_without_api_key(self, mock_anthropic):
        """Test initialization without an API key."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Remove any existing API key from the environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModelError) as excinfo:
                AnthropicModel("claude-2")

            assert "API key not provided" in str(excinfo.value)
            assert "ANTHROPIC_API_KEY environment variable" in str(excinfo.value)

    def test_init_with_api_key_param(self, mock_anthropic):
        """Test initialization with an API key parameter."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        model = AnthropicModel("claude-2", api_key="test-key")

        assert model.model_name == "claude-2"
        assert model.api_key == "test-key"
        assert isinstance(model.client, mock_anthropic.Anthropic)
        assert model.client.api_key == "test-key"

    def test_init_with_api_key_env(self, mock_anthropic):
        """Test initialization with an API key from the environment."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Set the API key in the environment
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-test-key"}):
            model = AnthropicModel("claude-2")

            assert model.model_name == "claude-2"
            assert model.api_key == "env-test-key"
            assert isinstance(model.client, mock_anthropic.Anthropic)
            assert model.client.api_key == "env-test-key"

    def test_init_with_options(self, mock_anthropic):
        """Test initialization with additional options."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        model = AnthropicModel("claude-2", api_key="test-key", temperature=0.5, max_tokens=100)

        assert model.model_name == "claude-2"
        assert model.api_key == "test-key"
        assert model.options == {"temperature": 0.5, "max_tokens": 100}

    def test_generate_success(self, mock_anthropic):
        """Test successful text generation."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Create a mock response
        mock_content = MagicMock()
        mock_content.text = "Generated text response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        # Set up the mock client
        model = AnthropicModel("claude-2", api_key="test-key")
        model.client.messages.create.return_value = mock_response

        # Generate text
        result = model.generate("Test prompt", temperature=0.7)

        # Check the result
        assert result == "Generated text response"

        # Check that the client was called correctly
        # Note: The model now adds a default max_tokens parameter
        call_args = model.client.messages.create.call_args[1]
        assert call_args["model"] == "claude-2"
        assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 1000  # Default value

    def test_generate_with_options_conversion(self, mock_anthropic):
        """Test option conversion during generation."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Create a mock response
        mock_content = MagicMock()
        mock_content.text = "Generated text response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        # Set up the mock client
        model = AnthropicModel("claude-2", api_key="test-key")
        model.client.messages.create.return_value = mock_response

        # Generate text with options that need conversion
        result = model.generate(
            "Test prompt",
            max_tokens=100,  # This is now the correct parameter name
            stop=["END"],  # Should be converted to stop_sequences
        )

        # Check that the client was called with converted options
        call_args = model.client.messages.create.call_args[1]
        assert call_args["model"] == "claude-2"
        assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert call_args["max_tokens"] == 100
        assert call_args["stop_sequences"] == ["END"]

    def test_generate_rate_limit_error(self, mock_anthropic):
        """Test handling of rate limit errors during generation."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Set up the mock client to raise a rate limit error
        model = AnthropicModel("claude-2", api_key="test-key")
        model.client.messages.create.side_effect = mock_anthropic.RateLimitError(
            "Rate limit exceeded"
        )

        # Generate text and check for the correct error
        with pytest.raises(ModelAPIError) as excinfo:
            model.generate("Test prompt")

        error = excinfo.value
        assert "rate limit exceeded" in str(error).lower()
        # The ModelError doesn't have model_name as a direct attribute
        # It's stored in the metadata
        assert "claude-2" in str(error)
        assert error.component == "AnthropicModel"  # The actual implementation sets this
        assert "generation" in str(error)
        # The suggestions are part of the error message
        assert "suggestions" in str(error).lower()

    def test_count_tokens(self, mock_anthropic):
        """Test token counting."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Create a mock response
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_response = MagicMock()
        mock_response.usage = mock_usage

        # Set up the mock client
        model = AnthropicModel("claude-2", api_key="test-key")
        model.client.messages.count_tokens.return_value = mock_response

        # Count tokens
        token_count = model.count_tokens("Test text")

        # Check the result
        assert token_count == 10

        # Check that the client was called correctly
        model.client.messages.count_tokens.assert_called_once_with(
            messages=[{"role": "user", "content": "Test text"}]
        )

    def test_count_tokens_error(self, mock_anthropic):
        """Test error handling during token counting."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Set up the mock client to raise an error
        model = AnthropicModel("claude-2", api_key="test-key")
        model.client.messages.count_tokens.side_effect = Exception("Token counting failed")

        # Count tokens and check for the correct error
        with pytest.raises(ModelError) as excinfo:
            model.count_tokens("Test text")

        error = excinfo.value
        assert "Error counting tokens" in str(error)
        assert error.component == "AnthropicModel"
        assert error.operation == "token_counting"
        assert any("API" in suggestion for suggestion in error.suggestions)

    def test_configure(self, mock_anthropic):
        """Test model configuration."""
        # Import here after mocking
        from sifaka.models.anthropic import AnthropicModel

        # Create a model with initial options
        model = AnthropicModel("claude-2", api_key="test-key", temperature=0.7)

        assert model.options == {"temperature": 0.7}

        # Configure with new options
        model.configure(temperature=0.5, max_tokens=100)

        # Check that the options were updated
        assert model.options == {"temperature": 0.5, "max_tokens": 100}
