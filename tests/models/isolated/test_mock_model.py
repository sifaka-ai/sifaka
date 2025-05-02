"""Test module for sifaka.models.mock."""

import unittest
from unittest.mock import MagicMock
from typing import Dict, Any, Optional

# Create mock classes for base module
class ModelConfig:
    """Mock ModelConfig class."""
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None
    trace_enabled: bool = True

class APIClient:
    """Mock API client interface."""
    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Mock implementation of send_prompt."""
        return f"Mock response to: {prompt}"

class TokenCounter:
    """Mock token counter interface."""
    def count_tokens(self, text: str) -> int:
        """Mock implementation of count_tokens."""
        return len(text.split())

class ModelProvider:
    """Mock model provider base class."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with config."""
        self._config = config

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        pass

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response."""
        return {}


# MockProvider implementation
class MockProvider(ModelProvider):
    """Mock model provider for testing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock provider."""
        super().__init__(config)

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate a mock response."""
        return {
            "text": f"Mock response to: {prompt}",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(prompt.split()) + 10
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration."""
        if not config.get("name"):
            raise ValueError("Name is required")
        if not config.get("description"):
            raise ValueError("Description is required")


# Concrete implementation for testing
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
        return APIClient()

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default mock token counter."""
        return TokenCounter()


class TestMockProvider(unittest.TestCase):
    """Tests for the MockProvider class."""

    def test_initialization(self):
        """Test initialization with valid config."""
        config = {
            "name": "test_mock",
            "description": "Test mock provider",
        }
        provider = ConcreteMockProvider(config)
        # Check that our original config values are accessible on the provider
        self.assertEqual(provider._original_config, config)

    def test_invalid_config_missing_name(self):
        """Test initialization with missing name in config."""
        config = {
            "description": "Test mock provider",
        }
        with self.assertRaises(ValueError):
            ConcreteMockProvider(config)

    def test_invalid_config_missing_description(self):
        """Test initialization with missing description in config."""
        config = {
            "name": "test_mock",
        }
        with self.assertRaises(ValueError):
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
        self.assertIsInstance(response, dict)
        self.assertIn("text", response)
        self.assertIn("usage", response)
        self.assertEqual("Mock response to: Hello, world!", response["text"])

        # Check token counts
        self.assertEqual(2, response["usage"]["prompt_tokens"])  # "Hello," and "world!"
        self.assertEqual(10, response["usage"]["completion_tokens"])  # Fixed in the mock implementation
        self.assertEqual(12, response["usage"]["total_tokens"])  # 2 prompt + 10 completion

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
        self.assertEqual("Mock response to: Test prompt", response["text"])
        self.assertEqual(2, response["usage"]["prompt_tokens"])