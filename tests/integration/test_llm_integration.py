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

            # Import here to avoid load errors when OpenAI is not installed
            from sifaka.models.openai import OpenAIProvider
            provider = OpenAIProvider(
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
        class TestProvider(ModelProvider):
            def __init__(self, **kwargs):
                config = kwargs.get("config", {
                    "name": "test_provider",
                    "description": "Test provider for integration tests",
                    "params": {
                        "test_param": "test_value"
                    }
                })
                super().__init__(config)
                self._client = MagicMock()

            def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
                """Generate a mock response."""
                return {
                    "text": f"Generated response for: {prompt}",
                    "model": "test-model",
                    "provider": "test-provider",
                    "usage": {"tokens": len(prompt.split())}
                }

            def validate_config(self, config: Dict[str, Any]) -> None:
                """Validate the configuration."""
                super().validate_config(config)
                if not config.get("params"):
                    raise ValueError("params is required")

        provider = TestProvider()
        return provider

    def test_openai_provider_integration(self, openai_provider):
        """Test basic integration with OpenAI provider."""
        # Generate text with openai provider
        response = openai_provider.generate("Test prompt")

        # Verify response structure
        assert "text" in response
        assert "model" in response
        assert response["text"] == "This is a test response from OpenAI"

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

        rule = LengthRule(name="length_rule", description="Validates text length")

        # Generate text with provider
        response = alternative_provider.generate("Test prompt")

        # Validate with rule
        result = rule.validate(response["text"])

        # Verify results
        assert result.passed
        assert "length" in result.metadata
        assert result.metadata["length"] > 10

    def test_provider_error_handling(self):
        """Test error handling with LLM providers."""
        # Create a provider that will fail
        class ErrorProvider(ModelProvider):
            def __init__(self):
                super().__init__({"name": "error_provider", "params": {}})

            def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
                if "error" in prompt.lower():
                    raise ValueError("Simulated API error")
                return {"text": "Normal response"}

        provider = ErrorProvider()

        # Test normal request
        response = provider.generate("Normal prompt")
        assert response["text"] == "Normal response"

        # Test error condition
        with pytest.raises(ValueError) as excinfo:
            provider.generate("Trigger error please")
        assert "Simulated API error" in str(excinfo.value)

    def test_rate_limit_recovery(self, openai_provider):
        """Test recovery from rate limit errors."""
        with patch.object(openai_provider, '_client') as mock_client:
            # Setup to raise error on first call, succeed on second
            side_effects = [
                Exception("Rate limit exceeded"),
                MockLLMResponse("Successful retry response")
            ]
            mock_client.chat.completions.create.side_effect = side_effects

            # Add retry logic for the test
            def retry_generate(prompt, max_retries=1):
                for attempt in range(max_retries + 1):
                    try:
                        return openai_provider.generate(prompt)
                    except Exception as e:
                        if attempt < max_retries and "Rate limit" in str(e):
                            continue
                        raise

            # Should recover after retry
            response = retry_generate("Test prompt", max_retries=1)
            assert "text" in response
            assert response["text"] == "Successful retry response"


class TestCrossPlatformIntegration:
    """Tests for integration across different platforms and providers."""

    def test_multi_provider_critic(self, openai_provider, alternative_provider):
        """Test using critic with multiple providers."""
        # Simple critic that uses two providers
        class MultiProviderCritic:
            def __init__(self, primary_provider, fallback_provider):
                self.primary_provider = primary_provider
                self.fallback_provider = fallback_provider

            def process(self, text):
                try:
                    result = self.primary_provider.generate(
                        f"Improve this text: {text}"
                    )
                    return CriticOutput(
                        result=CriticResult.SUCCESS,
                        improved_text=result["text"],
                        metadata=CriticMetadata(
                            score=0.9,
                            feedback="Improved with primary provider",
                            issues=[],
                            suggestions=[],
                            attempt_number=1,
                            processing_time_ms=100.0
                        )
                    )
                except Exception:
                    # Fallback to alternative provider
                    result = self.fallback_provider.generate(
                        f"Improve this text: {text}"
                    )
                    return CriticOutput(
                        result=CriticResult.SUCCESS,
                        improved_text=result["text"],
                        metadata=CriticMetadata(
                            score=0.8,
                            feedback="Improved with fallback provider",
                            issues=[],
                            suggestions=[],
                            attempt_number=1,
                            processing_time_ms=100.0
                        )
                    )

        # Create critic
        critic = MultiProviderCritic(openai_provider, alternative_provider)

        # Test normal operation (primary provider)
        with patch.object(openai_provider, 'generate') as mock_generate:
            mock_generate.return_value = {"text": "Improved text"}
            result = critic.process("Test text")
            assert result.result == CriticResult.SUCCESS
            assert result.improved_text == "Improved text"
            assert "primary provider" in result.metadata.feedback

        # Test fallback operation
        with patch.object(openai_provider, 'generate') as mock_primary:
            mock_primary.side_effect = Exception("API error")
            result = critic.process("Test text")
            assert result.result == CriticResult.SUCCESS
            assert "Generated response for: Improve this text: Test text" in result.improved_text
            assert "fallback provider" in result.metadata.feedback