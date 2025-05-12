"""
Integration tests for model providers.

This module contains integration tests for the OpenAI and Anthropic providers
that connect to the actual APIs using environment variables for API keys.
"""

import os
import pytest
from typing import Optional

from sifaka.models.providers.openai import OpenAIProvider
from sifaka.models.providers.anthropic import AnthropicProvider
from sifaka.utils.config.models import ModelConfig


def get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key from environment variables."""
    return os.environ.get("OPENAI_API_KEY")


def get_anthropic_api_key() -> Optional[str]:
    """Get the Anthropic API key from environment variables."""
    return os.environ.get("ANTHROPIC_API_KEY")


class TestProviderIntegration:
    """Integration tests for model providers."""

    @pytest.mark.skipif(not get_openai_api_key(), reason="OpenAI API key not set")
    def test_openai_generate(self):
        """Test OpenAI provider with real API."""
        # Create provider with API key from environment
        config = ModelConfig(api_key=get_openai_api_key())
        provider = OpenAIProvider(model_name="gpt-3.5-turbo", config=config)
        
        # Generate text
        prompt = "Write a one-sentence poem about coding."
        response = provider.generate(prompt, max_tokens=30)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nOpenAI response: {response}")

    @pytest.mark.skipif(not get_openai_api_key(), reason="OpenAI API key not set")
    def test_openai_count_tokens(self):
        """Test OpenAI token counting with real API."""
        # Create provider with API key from environment
        config = ModelConfig(api_key=get_openai_api_key())
        provider = OpenAIProvider(model_name="gpt-3.5-turbo", config=config)
        
        # Count tokens
        text = "This is a test sentence for counting tokens."
        token_count = provider.count_tokens(text)
        
        # Verify response
        assert isinstance(token_count, int)
        assert token_count > 0
        print(f"\nOpenAI token count: {token_count}")

    @pytest.mark.skipif(not get_anthropic_api_key(), reason="Anthropic API key not set")
    def test_anthropic_generate(self):
        """Test Anthropic provider with real API."""
        # Create provider with API key from environment
        config = ModelConfig(api_key=get_anthropic_api_key())
        provider = AnthropicProvider(model_name="claude-3-haiku-20240307", config=config)
        
        # Generate text
        prompt = "Write a one-sentence poem about coding."
        response = provider.generate(prompt, max_tokens=30)
        
        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nAnthropic response: {response}")

    @pytest.mark.skipif(not get_anthropic_api_key(), reason="Anthropic API key not set")
    def test_anthropic_count_tokens(self):
        """Test Anthropic token counting with real API."""
        # Create provider with API key from environment
        config = ModelConfig(api_key=get_anthropic_api_key())
        provider = AnthropicProvider(model_name="claude-3-haiku-20240307", config=config)
        
        # Count tokens
        text = "This is a test sentence for counting tokens."
        token_count = provider.count_tokens(text)
        
        # Verify response
        assert isinstance(token_count, int)
        assert token_count > 0
        print(f"\nAnthropic token count: {token_count}")
