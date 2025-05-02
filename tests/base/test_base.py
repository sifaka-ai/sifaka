"""Tests for base components."""

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Optional
import pytest
from sifaka.models.base import ModelConfig, ModelProvider
from sifaka.rules.base import Rule, RuleConfig, RuleResult, RulePriority

class BaseTestCase(unittest.TestCase):
    """Base test case with common utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_openai = patch('openai.OpenAI').start()
        self.mock_anthropic = patch('anthropic.Anthropic').start()
        self.addCleanup(patch.stopall)

    def create_mock_response(self, content: str, role: str = "assistant") -> Dict[str, Any]:
        """Create a mock LLM response."""
        return {
            "choices": [{
                "message": {
                    "content": content,
                    "role": role
                }
            }]
        }

    def create_mock_model_config(self, **kwargs) -> ModelConfig:
        """Create a mock model configuration."""
        default_config = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key": "test_key",
            "trace_enabled": True
        }
        default_config.update(kwargs)
        return ModelConfig(**default_config)

    def assert_dict_contains(self, actual: Dict[str, Any], expected: Dict[str, Any]):
        """Assert that actual dict contains all expected key-value pairs."""
        for key, value in expected.items():
            self.assertIn(key, actual)
            self.assertEqual(actual[key], value)

    def assert_list_contains(self, actual: List[Any], expected: List[Any]):
        """Assert that actual list contains all expected items."""
        for item in expected:
            self.assertIn(item, actual)

@pytest.fixture
def mock_openai():
    """Fixture for mocking OpenAI client."""
    with patch('openai.OpenAI') as mock:
        yield mock

@pytest.fixture
def mock_anthropic():
    """Fixture for mocking Anthropic client."""
    with patch('anthropic.Anthropic') as mock:
        yield mock

@pytest.fixture
def mock_model_config():
    """Fixture for creating a mock model configuration."""
    return ModelConfig(
        temperature=0.7,
        max_tokens=1000,
        api_key="test_key",
        trace_enabled=True
    )