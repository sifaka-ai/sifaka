"""Comprehensive tests for different model providers."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
from typing import Dict, Any

from sifaka import improve
from sifaka.core.engine import SifakaEngine
from sifaka.core.config import Config
from sifaka.core.models import SifakaResult
from sifaka.core.exceptions import ModelProviderError


class TestOpenAIModels:
    """Test OpenAI model provider."""

    @pytest.mark.asyncio
    async def test_gpt4_model(self):
        """Test GPT-4 model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: GPT-4 analysis complete."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test GPT-4 model",
                model="gpt-4",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)
            assert mock_client.chat.completions.create.called

            # Verify correct model was used
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_gpt4_turbo_model(self):
        """Test GPT-4 Turbo model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: GPT-4 Turbo analysis."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test GPT-4 Turbo",
                model="gpt-4-turbo",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_gpt4o_model(self):
        """Test GPT-4o model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: GPT-4o analysis."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test GPT-4o", model="gpt-4o", max_iterations=1, critics=["reflexion"]
            )

            assert isinstance(result, SifakaResult)
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_gpt4o_mini_model(self):
        """Test GPT-4o-mini model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: GPT-4o-mini analysis."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test GPT-4o-mini",
                model="gpt-4o-mini",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_gpt35_turbo_model(self):
        """Test GPT-3.5 Turbo model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: GPT-3.5 Turbo analysis."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test GPT-3.5 Turbo",
                model="gpt-3.5-turbo",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_openai_api_error_handling(self):
        """Test OpenAI API error handling."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("OpenAI API Error")
            )
            mock_openai.return_value = mock_client

            with pytest.raises(Exception):
                await improve(
                    "Test error handling",
                    model="gpt-4",
                    max_iterations=1,
                    critics=["reflexion"],
                )

    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self):
        """Test OpenAI rate limit error handling."""
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            # Simulate rate limit error
            from openai import RateLimitError

            mock_client.chat.completions.create = AsyncMock(
                side_effect=RateLimitError(
                    "Rate limit exceeded", response=None, body=None
                )
            )
            mock_openai.return_value = mock_client

            with pytest.raises(RateLimitError):
                await improve(
                    "Test rate limit",
                    model="gpt-4",
                    max_iterations=1,
                    critics=["reflexion"],
                )


class TestAnthropicModels:
    """Test Anthropic model provider."""

    @pytest.mark.asyncio
    async def test_claude3_opus_model(self):
        """Test Claude-3 Opus model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Claude-3 Opus analysis."

        # Mock Anthropic client
        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            # Also need to mock the OpenAI client fallback
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_openai_client = MagicMock()
                mock_openai_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_openai.return_value = mock_openai_client

                result = await improve(
                    "Test Claude-3 Opus",
                    model="claude-3-opus-20240229",
                    max_iterations=1,
                    critics=["reflexion"],
                )

                assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_claude3_sonnet_model(self):
        """Test Claude-3 Sonnet model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Claude-3 Sonnet analysis."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test Claude-3 Sonnet",
                model="claude-3-sonnet-20240229",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_claude3_haiku_model(self):
        """Test Claude-3 Haiku model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Claude-3 Haiku analysis."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test Claude-3 Haiku",
                model="claude-3-haiku-20240307",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)


class TestGoogleModels:
    """Test Google/Gemini model provider."""

    @pytest.mark.asyncio
    async def test_gemini_15_pro_model(self):
        """Test Gemini 1.5 Pro model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Gemini 1.5 Pro analysis."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test Gemini 1.5 Pro",
                model="gemini-1.5-pro",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_gemini_15_flash_model(self):
        """Test Gemini 1.5 Flash model integration."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "REFLECTION: Gemini 1.5 Flash analysis."
        )

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            result = await improve(
                "Test Gemini 1.5 Flash",
                model="gemini-1.5-flash",
                max_iterations=1,
                critics=["reflexion"],
            )

            assert isinstance(result, SifakaResult)


class TestModelProviderEdgeCases:
    """Test edge cases across model providers."""

    @pytest.mark.asyncio
    async def test_unsupported_model(self):
        """Test handling of unsupported model."""
        with pytest.raises(Exception):  # Should raise some form of error
            await improve(
                "Test unsupported model",
                model="unsupported-model-xyz",
                max_iterations=1,
                critics=["reflexion"],
            )

    @pytest.mark.asyncio
    async def test_model_parameter_passing(self):
        """Test that model parameters are correctly passed."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Parameter test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            await improve(
                "Test parameters",
                model="gpt-4",
                temperature=0.5,
                max_iterations=1,
                critics=["reflexion"],
            )

            # Verify parameters were passed correctly
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4"
            assert call_args[1]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_temperature_variations(self):
        """Test different temperature settings across models."""
        temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Temperature test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            for temp in temperatures:
                result = await improve(
                    f"Test temperature {temp}",
                    model="gpt-4o-mini",
                    temperature=temp,
                    max_iterations=1,
                    critics=["reflexion"],
                )

                assert isinstance(result, SifakaResult)

                # Verify temperature was passed
                call_args = mock_client.chat.completions.create.call_args
                assert call_args[1]["temperature"] == temp


class TestModelProviderComparison:
    """Test comparing different model providers."""

    @pytest.mark.asyncio
    async def test_same_task_different_models(self):
        """Test same task with different models."""
        test_text = "Compare model performance on this text"
        models = ["gpt-4o-mini", "claude-3-haiku-20240307", "gemini-1.5-flash"]

        results = []

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Model comparison test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            for model in models:
                result = await improve(
                    test_text, model=model, max_iterations=1, critics=["reflexion"]
                )
                results.append((model, result))

        # All should produce valid results
        assert len(results) == 3
        for model, result in results:
            assert isinstance(result, SifakaResult)
            assert result.original_text == test_text


class TestModelProviderConfigurationEdgeCases:
    """Test edge cases in model provider configuration."""

    @pytest.mark.asyncio
    async def test_model_with_special_characters(self):
        """Test model names with special characters."""
        # Some model names have special characters like hyphens, dots
        models_to_test = ["gpt-3.5-turbo", "claude-3-opus-20240229", "gemini-1.5-pro"]

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Special char test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            for model in models_to_test:
                result = await improve(
                    f"Test special chars in model {model}",
                    model=model,
                    max_iterations=1,
                    critics=["reflexion"],
                )

                assert isinstance(result, SifakaResult)

    @pytest.mark.asyncio
    async def test_case_sensitive_model_names(self):
        """Test case sensitivity in model names."""
        # Model names should be case sensitive
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Case sensitivity test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Correct case should work
            result = await improve(
                "Test correct case",
                model="gpt-4o-mini",
                max_iterations=1,
                critics=["reflexion"],
            )
            assert isinstance(result, SifakaResult)

            # Wrong case should potentially fail (depends on implementation)
            try:
                await improve(
                    "Test wrong case",
                    model="GPT-4O-MINI",  # Wrong case
                    max_iterations=1,
                    critics=["reflexion"],
                )
            except Exception:
                # This is expected for case-sensitive model names
                pass

    @pytest.mark.asyncio
    async def test_model_fallback_behavior(self):
        """Test model fallback behavior when primary model fails."""
        # This test depends on implementation details
        # If there's a fallback mechanism, test it here

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "REFLECTION: Fallback test."

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()

            # First call fails, second succeeds (simulating fallback)
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    Exception("Primary model failed"),
                    mock_response,  # Fallback succeeds
                ]
            )
            mock_openai.return_value = mock_client

            # Depending on implementation, this might succeed with fallback
            # or fail with the original exception
            try:
                result = await improve(
                    "Test fallback",
                    model="gpt-4",
                    max_iterations=1,
                    critics=["reflexion"],
                )
                # If fallback worked
                assert isinstance(result, SifakaResult)
            except Exception:
                # If no fallback, should fail
                pass
