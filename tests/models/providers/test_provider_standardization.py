"""
Tests for the standardization of model providers.

This module contains tests for the standardization of model providers,
focusing on the common interface and behavior across different providers.
"""

import pytest
from unittest.mock import patch, MagicMock

from sifaka.models.providers.openai import OpenAIProvider
from sifaka.models.providers.anthropic import AnthropicProvider
from sifaka.models.providers.gemini import GeminiProvider
from sifaka.models.providers.mock import MockProvider


class TestProviderStandardization:
    """Tests for the standardization of model providers."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        mock_manager = MagicMock()
        mock_manager.get.return_value = "test-model"
        return mock_manager

    def test_openai_description(self):
        """Test that OpenAIProvider has a description property."""
        with patch.object(OpenAIProvider, "_state_manager", create=True) as mock_state:
            mock_state.get.return_value = "gpt-4"
            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._state_manager = mock_state

            assert hasattr(provider, "description")
            assert isinstance(provider.description, str)
            assert "OpenAI" in provider.description
            assert "gpt-4" in provider.description

    def test_anthropic_description(self):
        """Test that AnthropicProvider has a description property."""
        with patch.object(AnthropicProvider, "_state_manager", create=True) as mock_state:
            mock_state.get.return_value = "claude-3-opus"
            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._state_manager = mock_state

            assert hasattr(provider, "description")
            assert isinstance(provider.description, str)
            assert "Anthropic" in provider.description
            assert "claude-3-opus" in provider.description

    def test_openai_update_config(self):
        """Test that OpenAIProvider has an update_config method."""
        with patch.object(OpenAIProvider, "_state_manager", create=True) as mock_state:
            mock_config = MagicMock()
            # Add with_options and with_params methods to the mock
            mock_config.with_options.return_value = mock_config
            mock_config.with_params.return_value = mock_config
            mock_state.get.return_value = mock_config

            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._state_manager = mock_state

            assert hasattr(provider, "update_config")
            provider.update_config(temperature=0.8, max_tokens=500)

            # Check that state manager was called to update config with the mock config
            # The actual config will be the result of with_options/with_params calls
            mock_state.update.assert_called_once()

    def test_anthropic_update_config(self):
        """Test that AnthropicProvider has an update_config method."""
        with patch.object(AnthropicProvider, "_state_manager", create=True) as mock_state:
            mock_config = MagicMock()
            # Add with_options and with_params methods to the mock
            mock_config.with_options.return_value = mock_config
            mock_config.with_params.return_value = mock_config
            mock_state.get.return_value = mock_config

            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._state_manager = mock_state

            assert hasattr(provider, "update_config")
            provider.update_config(temperature=0.8, max_tokens=500)

            # Check that state manager was called to update config with the mock config
            # The actual config will be the result of with_options/with_params calls
            mock_state.update.assert_called_once()

    def test_consistent_interface_across_providers(self):
        """Test that all providers have a consistent interface."""
        # Check that all provider classes have the required methods
        provider_classes = [OpenAIProvider, AnthropicProvider, GeminiProvider, MockProvider]

        # Check that all providers have the same interface
        for provider_class in provider_classes:
            # Check if the class has the description property
            assert any(
                attr == "description" for attr in dir(provider_class)
            ), f"{provider_class.__name__} missing description"

            # Check if the class has the update_config method
            assert "update_config" in dir(
                provider_class
            ), f"{provider_class.__name__} missing update_config"
