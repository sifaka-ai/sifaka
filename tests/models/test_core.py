"""
Tests for the core model provider implementation.
"""

import unittest
from unittest.mock import Mock, patch

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.models.managers.client import ClientManager
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.models.managers.tracing import TracingManager
from sifaka.models.services.generation import GenerationService


class MockAPIClient(APIClient):
    """Mock API client for testing."""
    
    def __init__(self):
        self.send_prompt = Mock(return_value="Generated text")
        
    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Mock implementation of send_prompt."""
        return "Generated text"


class MockTokenCounter(TokenCounter):
    """Mock token counter for testing."""
    
    def __init__(self):
        self.count_tokens = Mock(return_value=10)
        
    def count_tokens(self, text: str) -> int:
        """Mock implementation of count_tokens."""
        return 10


class MockModelProvider(ModelProviderCore):
    """Mock model provider for testing."""
    
    def _create_default_client(self) -> APIClient:
        """Create a default mock API client."""
        return MockAPIClient()
        
    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default mock token counter."""
        return MockTokenCounter()


class TestModelProviderCore(unittest.TestCase):
    """Tests for the ModelProviderCore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(temperature=0.5, max_tokens=100)
        self.provider = MockModelProvider(
            model_name="test-model",
            config=self.config,
        )
        
    def test_initialization(self):
        """Test that the provider initializes correctly."""
        self.assertEqual(self.provider.model_name, "test-model")
        self.assertEqual(self.provider.config, self.config)
        
    def test_count_tokens(self):
        """Test that count_tokens works correctly."""
        count = self.provider.count_tokens("Test text")
        self.assertEqual(count, 10)
        
    def test_generate(self):
        """Test that generate works correctly."""
        text = self.provider.generate("Test prompt")
        self.assertEqual(text, "Generated text")
        
    def test_generate_with_overrides(self):
        """Test that generate works correctly with config overrides."""
        text = self.provider.generate("Test prompt", temperature=0.8, max_tokens=200)
        self.assertEqual(text, "Generated text")


if __name__ == "__main__":
    unittest.main()
