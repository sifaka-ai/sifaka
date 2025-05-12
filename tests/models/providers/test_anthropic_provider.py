"""
Tests for the Anthropic provider.

This module contains tests for the Anthropic provider implementation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from sifaka.models.providers.anthropic import AnthropicProvider
from sifaka.utils.config.models import ModelConfig


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.send_prompt.return_value = "Mock response"
    return mock_client


@pytest.fixture
def mock_token_counter():
    """Create a mock token counter."""
    mock_counter = MagicMock()
    mock_counter.count_tokens.return_value = 10
    return mock_counter


class TestAnthropicProvider:
    """Tests for the AnthropicProvider class."""

    def test_init(self):
        """Test initialization with default parameters."""
        with patch("importlib.util.find_spec", return_value=True):
            provider = AnthropicProvider()
            assert provider.name == "Anthropic-claude-3-opus-20240229"
            assert provider._state_manager.get("model_name") == "claude-3-opus-20240229"
            assert provider._state_manager.get("initialized") is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        with patch("importlib.util.find_spec", return_value=True):
            config = ModelConfig(temperature=0.8, max_tokens=500)
            provider = AnthropicProvider(model_name="claude-3-sonnet-20240229", config=config)
            assert provider.name == "Anthropic-claude-3-sonnet-20240229"
            assert provider._state_manager.get("model_name") == "claude-3-sonnet-20240229"
            assert provider._state_manager.get("config").temperature == 0.8
            assert provider._state_manager.get("config").max_tokens == 500

    def test_generate_with_mocks(self, mock_anthropic_client, mock_token_counter):
        """Test generate method with mock client and token counter."""
        with patch("importlib.util.find_spec", return_value=True):
            provider = AnthropicProvider(
                api_client=mock_anthropic_client,
                token_counter=mock_token_counter
            )
            provider._state_manager.update("client", mock_anthropic_client)
            provider._state_manager.update("initialized", True)
            
            result = provider.generate("Test prompt")
            
            assert result == "Mock response"
            mock_anthropic_client.send_prompt.assert_called_once()

    def test_count_tokens_with_mocks(self, mock_anthropic_client, mock_token_counter):
        """Test count_tokens method with mock token counter."""
        with patch("importlib.util.find_spec", return_value=True):
            provider = AnthropicProvider(
                api_client=mock_anthropic_client,
                token_counter=mock_token_counter
            )
            provider._state_manager.update("token_counter", mock_token_counter)
            provider._state_manager.update("initialized", True)
            
            result = provider.count_tokens("Test text")
            
            assert result == 10
            mock_token_counter.count_tokens.assert_called_once_with("Test text")

    def test_invoke_with_mocks(self, mock_anthropic_client, mock_token_counter):
        """Test invoke method with mock client and token counter."""
        with patch("importlib.util.find_spec", return_value=True):
            provider = AnthropicProvider(
                api_client=mock_anthropic_client,
                token_counter=mock_token_counter
            )
            provider._state_manager.update("client", mock_anthropic_client)
            provider._state_manager.update("initialized", True)
            
            result = provider.invoke("Test prompt")
            
            assert result == "Mock response"
            mock_anthropic_client.send_prompt.assert_called_once()
            
            # Check that statistics were updated
            stats = provider._state_manager.get("stats")
            assert stats["generation_count"] == 1

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
    def test_integration_generate(self):
        """Integration test for generate method with real API."""
        provider = AnthropicProvider(model_name="claude-3-haiku-20240307")
        result = provider.generate("Say hello", max_tokens=10)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
    def test_integration_count_tokens(self):
        """Integration test for count_tokens method with real API."""
        provider = AnthropicProvider(model_name="claude-3-haiku-20240307")
        result = provider.count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0
