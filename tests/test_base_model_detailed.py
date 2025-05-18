"""
Detailed tests for the base model module.

This module contains more comprehensive tests for the base model module
to improve test coverage.
"""

from unittest.mock import MagicMock, patch

import pytest

from sifaka.errors import ConfigurationError, ModelNotFoundError
from sifaka.models.base import create_model


class TestBaseModelDetailed:
    """Detailed tests for the base model module."""

    def test_create_model_with_combined_string(self) -> None:
        """Test creating a model with a combined provider:model string."""
        with patch("sifaka.factories.create_model") as mock_factory:
            mock_model = MagicMock()
            mock_factory.return_value = mock_model

            model = create_model("openai:gpt-4", api_key="test-key")

            # Check that factory was called with correct arguments
            mock_factory.assert_called_once_with("openai:gpt-4", "", api_key="test-key")
            assert model == mock_model

    def test_create_model_with_separate_provider_and_model(self) -> None:
        """Test creating a model with separate provider and model name."""
        with patch("sifaka.factories.create_model") as mock_factory:
            mock_model = MagicMock()
            mock_factory.return_value = mock_model

            model = create_model("openai", "gpt-4", api_key="test-key")

            # Check that factory was called with correct arguments
            mock_factory.assert_called_once_with("openai", "gpt-4", api_key="test-key")
            assert model == mock_model

    def test_create_model_factory_error_openai_fallback(self) -> None:
        """Test fallback to direct import when factory fails for OpenAI."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with patch("sifaka.models.openai.OpenAIModel") as mock_openai_model:
                with patch("sifaka.models.openai.OPENAI_AVAILABLE", True):
                    mock_model = MagicMock()
                    mock_openai_model.return_value = mock_model

                    model = create_model("openai", "gpt-4", api_key="test-key")

                    # Check that OpenAIModel was created with correct arguments
                    mock_openai_model.assert_called_once_with(
                        model_name="gpt-4", api_key="test-key"
                    )
                    assert model == mock_model

    def test_create_model_factory_error_anthropic_fallback(self) -> None:
        """Test fallback to direct import when factory fails for Anthropic."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with patch("sifaka.models.anthropic.AnthropicModel") as mock_anthropic_model:
                mock_model = MagicMock()
                mock_anthropic_model.return_value = mock_model

                model = create_model("anthropic", "claude-3", api_key="test-key")

                # Check that AnthropicModel was created with correct arguments
                mock_anthropic_model.assert_called_once_with(
                    model_name="claude-3", api_key="test-key"
                )
                assert model == mock_model

    def test_create_model_factory_error_gemini_fallback(self) -> None:
        """Test fallback to direct import when factory fails for Gemini."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with patch("sifaka.models.gemini.GeminiModel") as mock_gemini_model:
                mock_model = MagicMock()
                mock_gemini_model.return_value = mock_model

                model = create_model("gemini", "gemini-pro", api_key="test-key")

                # Check that GeminiModel was created with correct arguments
                mock_gemini_model.assert_called_once_with(
                    model_name="gemini-pro", api_key="test-key"
                )
                assert model == mock_model

    def test_create_model_factory_error_mock_fallback(self) -> None:
        """Test fallback to mock model when factory fails for mock provider."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            model = create_model("mock", "test-model", param1="value1")

            # Check that a MockModel was created
            assert model.model_name == "test-model"
            assert model.kwargs == {"param1": "value1"}

            # Test the mock model's methods
            assert model.generate("Test prompt") == "Mock response from test-model for: Test prompt"
            assert model.count_tokens("This is a test") == 4

    def test_create_model_factory_error_unknown_provider(self) -> None:
        """Test error handling for unknown provider when factory fails."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with pytest.raises(ModelNotFoundError) as excinfo:
                create_model("unknown", "model")

            assert "Provider 'unknown' not found" in str(excinfo.value)

    def test_create_model_openai_not_available(self) -> None:
        """Test error handling when OpenAI package is not available."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with patch("sifaka.models.openai.OPENAI_AVAILABLE", False):
                with pytest.raises(ConfigurationError) as excinfo:
                    create_model("openai", "gpt-4")

                assert "OpenAI package not installed" in str(excinfo.value)
                assert "pip install openai tiktoken" in str(excinfo.value)

    def test_create_model_openai_import_error(self) -> None:
        """Test error handling when OpenAI import fails."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with patch(
                "sifaka.models.openai.OpenAIModel",
                side_effect=ImportError("No module named 'openai'"),
            ):
                with pytest.raises(ConfigurationError) as excinfo:
                    create_model("openai", "gpt-4")

                assert "OpenAI package not installed" in str(excinfo.value)
                assert "pip install openai tiktoken" in str(excinfo.value)

    def test_create_model_anthropic_import_error(self) -> None:
        """Test error handling when Anthropic import fails."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with patch(
                "sifaka.models.anthropic.AnthropicModel",
                side_effect=ImportError("No module named 'anthropic'"),
            ):
                with pytest.raises(ConfigurationError) as excinfo:
                    create_model("anthropic", "claude-3")

                assert "Anthropic package not installed" in str(excinfo.value)

    def test_create_model_gemini_import_error(self) -> None:
        """Test error handling when Gemini import fails."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            with patch(
                "sifaka.models.gemini.GeminiModel",
                side_effect=ImportError("No module named 'google.generativeai'"),
            ):
                with pytest.raises(ConfigurationError) as excinfo:
                    create_model("gemini", "gemini-pro")

                assert "Google Gemini package not installed" in str(excinfo.value)

    def test_mock_model_methods(self) -> None:
        """Test the methods of the mock model."""
        with patch("sifaka.factories.create_model", side_effect=Exception("Factory failed")):
            model = create_model("mock", "test-model")

            # Test generate method
            response = model.generate("Hello, world!", temperature=0.7)
            assert response == "Mock response from test-model for: Hello, world!"

            # Test count_tokens method
            token_count = model.count_tokens("This is a test sentence with multiple words.")
            assert token_count == 8  # Number of words in the sentence
