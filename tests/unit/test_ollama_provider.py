"""Tests for Ollama provider integration."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sifaka.core.llm_client import LLMClient, LLMManager, Provider
from sifaka.core.types import Provider as TypesProvider


class TestOllamaProvider:
    """Test Ollama provider functionality."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for Ollama."""
        mock = MagicMock()
        mock.chat = MagicMock()
        mock.chat.completions = MagicMock()
        mock.chat.completions.create = AsyncMock()
        return mock

    def test_ollama_provider_enum(self):
        """Test that OLLAMA is in Provider enum."""
        assert Provider.OLLAMA == "ollama"
        assert TypesProvider.OLLAMA == "ollama"
        assert "ollama" in [p.value for p in Provider]

    def test_ollama_base_url(self):
        """Test Ollama base URL configuration."""
        from sifaka.core.llm_client import LLMClient

        assert LLMClient.PROVIDER_URLS[Provider.OLLAMA] == "http://localhost:11434/v1"

    @patch("sifaka.core.llm_client.openai.AsyncOpenAI")
    def test_ollama_client_initialization(self, mock_openai_class):
        """Test Ollama client initialization."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Test default initialization
        client = LLMClient(Provider.OLLAMA, "llama3.2")

        # Should use Ollama base URL
        mock_openai_class.assert_called_with(
            api_key="ollama",  # Default API key for Ollama
            base_url="http://localhost:11434/v1",
        )

        assert client.provider == Provider.OLLAMA
        assert client.model == "llama3.2"

    @patch("sifaka.core.llm_client.openai.AsyncOpenAI")
    def test_ollama_custom_base_url(self, mock_openai_class):
        """Test Ollama with custom base URL from environment."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Set custom base URL
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:11434/v1"}):
            LLMClient(Provider.OLLAMA, "llama3.2")

            # Should use custom base URL
            mock_openai_class.assert_called_with(
                api_key="ollama", base_url="http://custom:11434/v1"
            )

    def test_ollama_model_mappings(self):
        """Test Ollama model mappings."""
        mappings = LLMClient.MODEL_MAPPINGS[Provider.OLLAMA]

        # Test direct model names
        assert mappings["llama3.2"] == "llama3.2"
        assert mappings["llama3.1:8b"] == "llama3.1:8b"
        assert mappings["mistral"] == "mistral"
        assert mappings["qwen2.5-coder"] == "qwen2.5-coder"

        # Test common aliases
        assert mappings["gpt-4o-mini"] == "llama3.2:3b"  # Fast model
        assert mappings["gpt-4"] == "llama3.1:70b"  # Large model
        assert mappings["gpt-3.5-turbo"] == "llama3.2"  # Default model

    @patch("sifaka.core.llm_client.openai.AsyncOpenAI")
    def test_ollama_api_key_optional(self, mock_openai_class):
        """Test that Ollama works without API key."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # No API key in environment
        with patch.dict(os.environ, {}, clear=True):
            LLMClient(Provider.OLLAMA, "llama3.2")

            # Should still work with default "ollama" key
            mock_openai_class.assert_called_with(
                api_key="ollama", base_url="http://localhost:11434/v1"
            )

    @patch("sifaka.core.llm_client.openai.AsyncOpenAI")
    def test_ollama_custom_api_key(self, mock_openai_class):
        """Test Ollama with custom API key."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Set custom API key
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "custom-key"}):
            LLMClient(Provider.OLLAMA, "llama3.2")

            # Should use custom API key
            mock_openai_class.assert_called_with(
                api_key="custom-key", base_url="http://localhost:11434/v1"
            )

    @pytest.mark.asyncio
    async def test_ollama_provider_detection(self):
        """Test automatic Ollama provider detection."""
        # Clear all API keys
        with patch.dict(os.environ, {}, clear=True):
            # Set only OLLAMA_BASE_URL
            with patch.dict(
                os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}
            ):
                with patch(
                    "sifaka.core.llm_client_pool.LLMClientPool.get_client"
                ) as mock_get:
                    mock_client = MagicMock()
                    mock_get.return_value = mock_client

                    # Should detect Ollama
                    await LLMManager.get_client()
                    mock_get.assert_called_with(
                        Provider.OLLAMA, "gpt-4o-mini", 0.7, None
                    )

    @pytest.mark.asyncio
    async def test_ollama_complete(self, mock_openai_client):
        """Test Ollama completion."""
        with patch("sifaka.core.llm_client.openai.AsyncOpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_openai_client

            # Mock response
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Ollama response"))
            ]
            mock_response.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            mock_openai_client.chat.completions.create.return_value = mock_response

            client = LLMClient(Provider.OLLAMA, "llama3.2")
            response = await client.complete([{"role": "user", "content": "Hello"}])

            assert response.content == "Ollama response"
            assert response.model == "llama3.2"
            assert response.usage["total_tokens"] == 15
