"""
Tests for the MockProvider class.
"""

import pytest
from unittest.mock import MagicMock
from sifaka.models.mock import MockProvider
from sifaka.models.base import APIClient, TokenCounter
from sifaka.models.managers.client import ClientManager
from sifaka.models.managers.token_counter import TokenCounterManager


class MockClientManager(ClientManager):
    """Mock client manager for testing."""

    def _create_default_client(self) -> APIClient:
        """Create a mock API client."""
        mock_client = MagicMock(spec=APIClient)
        return mock_client


class MockTokenCounterManager(TokenCounterManager):
    """Mock token counter manager for testing."""

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a mock token counter."""
        mock_counter = MagicMock(spec=TokenCounter)
        mock_counter.count_tokens.return_value = 10  # Default token count
        return mock_counter


class ConcreteMockProvider(MockProvider):
    """Concrete implementation of MockProvider for testing."""

    def _create_default_client(self) -> ClientManager:
        """Create a default client manager."""
        return MockClientManager(model_name="test-model")

    def _create_default_token_counter(self) -> TokenCounterManager:
        """Create a default token counter manager."""
        return MockTokenCounterManager(model_name="test-model")


class TestMockProvider:
    """Tests for the MockProvider class."""

    def test_initialization(self):
        """Test initialization with valid config."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider"
        }

        provider = ConcreteMockProvider(config)

        # We're just verifying the provider initializes successfully
        assert isinstance(provider, ConcreteMockProvider)

    def test_initialization_with_params(self):
        """Test initialization with params."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider",
            "params": {
                "custom_param": "custom_value"
            }
        }

        provider = ConcreteMockProvider(config)

        # We're just verifying the provider initializes successfully
        assert isinstance(provider, ConcreteMockProvider)

    def test_generate(self):
        """Test generate method."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider"
        }

        provider = ConcreteMockProvider(config)

        # Test with a simple prompt
        prompt = "Hello, world!"
        response = provider.generate(prompt)

        assert isinstance(response, dict)
        assert "text" in response
        assert response["text"] == "Mock response to: Hello, world!"
        assert "usage" in response
        assert response["usage"]["prompt_tokens"] == len(prompt.split())
        assert response["usage"]["completion_tokens"] == 10
        assert response["usage"]["total_tokens"] == len(prompt.split()) + 10

    def test_generate_with_empty_prompt(self):
        """Test generate method with empty prompt."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider"
        }

        provider = ConcreteMockProvider(config)

        # Test with empty prompt
        prompt = ""
        response = provider.generate(prompt)

        assert isinstance(response, dict)
        assert "text" in response
        assert response["text"] == "Mock response to: "
        assert "usage" in response
        assert response["usage"]["prompt_tokens"] == 0  # No words in prompt
        assert response["usage"]["completion_tokens"] == 10
        assert response["usage"]["total_tokens"] == 10