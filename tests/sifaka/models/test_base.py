"""
Tests for the base model interface.
"""

import pytest
from unittest.mock import Mock

from sifaka.models.base import create_model


class TestModelCreation:
    """Tests for model creation."""
    
    def test_create_model_not_implemented(self):
        """Test that create_model raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_model("openai", "gpt-4")


class MockModel:
    """A mock model implementation for testing."""
    
    def __init__(self, model_name: str, **options):
        self.model_name = model_name
        self.options = options
    
    def generate(self, prompt: str, **options) -> str:
        """Generate text from a prompt."""
        return f"Generated text for prompt: {prompt}"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text.split())


class TestModelInterface:
    """Tests for the Model protocol."""
    
    def test_model_protocol_implementation(self):
        """Test that a class implementing the Model protocol works."""
        model = MockModel("test-model")
        
        # Test generate method
        result = model.generate("Test prompt")
        assert result == "Generated text for prompt: Test prompt"
        
        # Test count_tokens method
        count = model.count_tokens("This is a test")
        assert count == 4
