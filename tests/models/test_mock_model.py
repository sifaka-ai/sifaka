"""Test module for sifaka.models.mock."""

import pytest
from typing import Dict, Any, Optional
from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.mock import MockProvider


class MockAPIClient(APIClient):
    """Mock API client for testing."""

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Mock implementation of send_prompt."""
        return f"Mock response to: {prompt}"


class MockTokenCounter(TokenCounter):
    """Mock token counter for testing."""

    def count_tokens(self, text: str) -> int:
        """Mock implementation of count_tokens."""
        return len(text.split())


class ConcreteMockProvider(MockProvider):
    """Concrete implementation of MockProvider for testing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with model name set to mock-model."""
        # Store original config for tests
        self._original_config = config.copy()
        # Validate config before passing to parent
        self.validate_config(config)
        super().__init__(config)

    def _create_default_client(self) -> APIClient:
        """Create a default mock API client."""
        return MockAPIClient()

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default mock token counter."""
        return MockTokenCounter()

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        if not config.get("name"):
            raise ValueError("Name is required")
        if not config.get("description"):
            raise ValueError("Description is required")


class TestMockProvider:
    """Tests for the MockProvider class."""

    def test_initialization(self):
        """Test initialization with valid config."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider",
        }
        provider = ConcreteMockProvider(config)
        # Check that our original config values are accessible on the provider
        assert provider._original_config == config

    def test_invalid_config_missing_name(self):
        """Test initialization with missing name in config."""
        config = {
            "description": "Test mock provider",
        }
        with pytest.raises(ValueError, match="Name is required"):
            ConcreteMockProvider(config)

    def test_invalid_config_missing_description(self):
        """Test initialization with missing description in config."""
        config = {
            "name": "test_mock",
        }
        with pytest.raises(ValueError, match="Description is required"):
            ConcreteMockProvider(config)

    def test_generate(self):
        """Test generate method returns expected response."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider",
        }
        provider = ConcreteMockProvider(config)

        # Test with a simple prompt
        prompt = "Hello, world!"
        response = provider.generate(prompt)

        # Check response format and content
        assert isinstance(response, dict)
        assert "text" in response
        assert "usage" in response
        assert "Mock response to: Hello, world!" == response["text"]

        # Check token counts
        assert response["usage"]["prompt_tokens"] == 2  # "Hello," and "world!"
        assert response["usage"]["completion_tokens"] == 10  # Fixed in the mock implementation
        assert response["usage"]["total_tokens"] == 12  # 2 prompt + 10 completion

    def test_generate_with_additional_kwargs(self):
        """Test generate method ignores additional kwargs."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider",
        }
        provider = ConcreteMockProvider(config)

        # Test with additional kwargs that should be ignored
        prompt = "Test prompt"
        response = provider.generate(
            prompt,
            temperature=0.7,
            max_tokens=100,
            some_parameter="value"
        )

        # Basic response validation
        assert "Mock response to: Test prompt" == response["text"]
        assert response["usage"]["prompt_tokens"] == 2
