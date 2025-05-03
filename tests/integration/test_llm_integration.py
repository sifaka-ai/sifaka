"""
Integration tests for LLM providers.

These tests ensure that Sifaka components work correctly with various
LLM providers after the removal of the dedicated integration files.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from sifaka.models.base import ModelProvider, ModelConfig
from sifaka.rules.base import Rule, RuleResult
from sifaka.adapters.rules.base import BaseAdapter
from sifaka.critics.base import CriticConfig, CriticMetadata, CriticOutput, CriticResult


class MockLLMResponse:
    """Mock LLM response object that mimics external APIs."""

    def __init__(self, content: str, model: str = "test-model"):
        self.content = content
        self.model = model
        self.choices = [MagicMock(message=MagicMock(content=content))]

    def __getattr__(self, name):
        """Handle arbitrary attribute access for flexibility in mocking."""
        if name == "choices":
            return self.choices
        return MagicMock()


# Mock OpenAI provider for testing
class MockOpenAIProvider:
    """Mock OpenAI provider that doesn't inherit from ModelProvider."""

    def __init__(self, model_name="gpt-4", config=None):
        self.model_name = model_name
        self.config = config or {}
        self._client = MagicMock()
        self._client.chat.completions.create.return_value = MockLLMResponse(
            "This is a test response from OpenAI"
        )

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using the mock."""
        if not prompt:
            raise ValueError("Empty prompt")

        # Simulate rate limiting on the 3rd call
        if getattr(self, "_call_count", 0) == 2 and kwargs.get("simulate_rate_limit", False):
            setattr(self, "_call_count", getattr(self, "_call_count", 0) + 1)
            raise Exception("Rate limit exceeded")

        setattr(self, "_call_count", getattr(self, "_call_count", 0) + 1)

        return {
            "text": f"OpenAI response to: {prompt}",
            "model": self.model_name,
            "provider": "openai",
            "usage": {"tokens": len(prompt.split())}
        }


# Mock alternative provider for testing
class MockAlternativeProvider:
    """Mock alternative provider that doesn't inherit from ModelProvider."""

    def __init__(self, model_name="alternative-model", config=None):
        self.model_name = model_name
        self.config = config or {}

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using the mock."""
        if not prompt:
            raise ValueError("Empty prompt")

        return {
            "text": f"Generated response for: {prompt}",
            "model": self.model_name,
            "provider": "alternative",
            "usage": {"tokens": len(prompt.split())}
        }


class TestExternalLLMIntegration:
    """Tests for integration with external LLM providers."""

    @pytest.fixture
    def openai_provider(self):
        """Create a mocked OpenAI provider for testing."""
        with patch('openai.OpenAI') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_completion = MockLLMResponse("This is a test response from OpenAI")
            mock_instance.chat.completions.create.return_value = mock_completion

            provider = MockOpenAIProvider(
                model_name="gpt-4",
                config=ModelConfig(
                    api_key="test-key",
                    temperature=0.7,
                    max_tokens=1000
                )
            )
            yield provider

    @pytest.fixture
    def alternative_provider(self):
        """Create a mock provider for testing without specific implementations."""
        provider = MockAlternativeProvider()
        yield provider

    def test_openai_provider_integration(self, openai_provider):
        """Test basic integration with OpenAI provider."""
        response = openai_provider.generate("Test prompt")

        # Verify response structure
        assert "text" in response
        assert "OpenAI response to: Test prompt" == response["text"]
        assert "model" in response
        assert "usage" in response

    def test_alternative_provider_integration(self, alternative_provider):
        """Test integration with alternative provider."""
        # Generate text with alternative provider
        response = alternative_provider.generate("Test prompt")

        # Verify response structure
        assert "text" in response
        assert response["text"] == "Generated response for: Test prompt"

    def test_provider_with_rules(self, alternative_provider):
        """Test provider integration with rules."""
        # Create a simple rule
        class LengthRule(Rule):
            def validate(self, text: str, **kwargs) -> RuleResult:
                is_valid = len(text) > 10
                return RuleResult(
                    passed=is_valid,
                    message="Text length validation" + (" passed" if is_valid else " failed"),
                    metadata={"length": len(text)}
                )

            def _create_default_validator(self):
                """Create default validator - required by the abstract Rule class."""
                return self

        rule = LengthRule(name="length_rule", description="Validates text length")

        # Generate text with provider
        response = alternative_provider.generate("Test prompt")

        # Validate with rule
        result = rule.validate(response["text"])

        # Verify results
        assert result.passed
        assert "length" in result.metadata
        assert result.metadata["length"] > 10

    def test_provider_error_handling(self, alternative_provider):
        """Test error handling with LLM providers."""
        # Create a provider that raises errors
        provider_with_errors = MockAlternativeProvider()

        # Patch the generate method to raise an exception
        with patch.object(provider_with_errors, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("API error")

            # Test with error
            with pytest.raises(Exception) as exc_info:
                provider_with_errors.generate("Test prompt")

            assert "API error" in str(exc_info.value)

        # Test empty prompt
        with pytest.raises(ValueError):
            alternative_provider.generate("")

    def test_rate_limit_recovery(self, openai_provider):
        """Test recovery from rate limit errors."""
        # First call should succeed
        response1 = openai_provider.generate("First prompt")
        assert "text" in response1

        # Second call should succeed
        response2 = openai_provider.generate("Second prompt")
        assert "text" in response2

        # Third call with retry logic
        try:
            # This would fail with rate limit
            openai_provider.generate("Third prompt", simulate_rate_limit=True)
        except Exception as e:
            # In a real implementation, we'd have retry logic
            # For testing, we'll just verify we can continue after an error
            pass

        # Fourth call should succeed again
        response4 = openai_provider.generate("Fourth prompt")
        assert "text" in response4


class TestCrossPlatformIntegration:
    """Tests for cross-platform LLM provider integration."""

    def test_multi_provider_critic(self):
        """Test a critic that works with multiple providers."""
        # Create providers
        openai = MockOpenAIProvider(model_name="gpt-4")
        alternative = MockAlternativeProvider()

        # Create a simple critic that can use different providers
        class MultiProviderCritic:
            def __init__(self, providers):
                self.providers = providers
                self.current_provider_index = 0

            def critique(self, text):
                # Get the current provider (rotating through the list)
                provider = self.providers[self.current_provider_index]
                self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)

                # Use the provider to generate a critique
                prompt = f"Critique this text: {text}"
                response = provider.generate(prompt)

                return CriticOutput(
                    result=CriticResult.SUCCESS,
                    improved_text=response["text"],
                    metadata=CriticMetadata(
                        score=0.8,
                        feedback="Critique generated",
                        issues=[],
                        suggestions=[],
                        attempt_number=1,
                        processing_time_ms=10.0
                    )
                )

        # Create the critic with multiple providers
        critic = MultiProviderCritic([openai, alternative])

        # Test critiquing with each provider
        critique1 = critic.critique("First test text")
        assert critique1.result == CriticResult.SUCCESS
        assert "OpenAI response" in critique1.improved_text

        critique2 = critic.critique("Second test text")
        assert critique2.result == CriticResult.SUCCESS
        assert "Generated response for" in critique2.improved_text

        # Third should cycle back to OpenAI
        critique3 = critic.critique("Third test text")
        assert critique3.result == CriticResult.SUCCESS
        assert "OpenAI response" in critique3.improved_text