"""
Tests for the OpenAI model provider.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.models.base import ModelConfig
from sifaka.models.openai import OpenAIClient, OpenAIProvider, OpenAITokenCounter


class TestOpenAITokenCounter:
    """Tests for the OpenAITokenCounter class."""

    @patch("tiktoken.encoding_for_model")
    def test_initialization(self, mock_encoding_for_model):
        """Test successful initialization of token counter."""
        mock_encoding = MagicMock()
        mock_encoding_for_model.return_value = mock_encoding

        counter = OpenAITokenCounter(model="gpt-3.5-turbo")
        assert counter.encoding == mock_encoding
        mock_encoding_for_model.assert_called_once_with("gpt-3.5-turbo")

    @patch("tiktoken.encoding_for_model")
    @patch("tiktoken.get_encoding")
    def test_initialization_fallback(self, mock_get_encoding, mock_encoding_for_model):
        """Test fallback to cl100k_base encoding when model not found."""
        mock_encoding_for_model.side_effect = KeyError("Model not found")
        mock_fallback_encoding = MagicMock()
        mock_get_encoding.return_value = mock_fallback_encoding

        counter = OpenAITokenCounter(model="nonexistent-model")
        assert counter.encoding == mock_fallback_encoding
        mock_get_encoding.assert_called_once_with("cl100k_base")

    def test_count_tokens(self):
        """Test token counting functionality."""
        counter = OpenAITokenCounter()

        # Mock the encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        counter.encoding = mock_encoding

        count = counter.count_tokens("Test text")
        assert count == 5
        mock_encoding.encode.assert_called_once_with("Test text")

    def test_count_tokens_error(self):
        """Test error handling in count_tokens."""
        counter = OpenAITokenCounter()

        # Mock encoding with error
        mock_encoding = MagicMock()
        mock_encoding.encode.side_effect = Exception("Encoding error")
        counter.encoding = mock_encoding

        with pytest.raises(Exception) as exc_info:
            counter.count_tokens("Test text")

        assert "Encoding error" in str(exc_info.value)


class TestOpenAIClient:
    """Tests for the OpenAIClient class."""

    @patch("openai.OpenAI")
    def test_initialization(self, mock_openai):
        """Test client initialization."""
        client = OpenAIClient(api_key="test-key")
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch("openai.OpenAI")
    def test_send_prompt(self, mock_openai):
        """Test sending a prompt to the OpenAI API."""
        # Set up mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Generated response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(api_key="test-key")
        config = ModelConfig(temperature=0.5, max_tokens=100)

        response = client.send_prompt("Test prompt", config)

        assert response == "Generated response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.5,
            max_tokens=100,
        )

    @patch("openai.OpenAI")
    def test_send_prompt_api_error(self, mock_openai):
        """Test handling of API errors."""
        # Mock API error
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Simulate an OpenAI API error
        mock_client.chat.completions.create.side_effect = Exception("API error")

        client = OpenAIClient(api_key="test-key")
        config = ModelConfig()

        with pytest.raises(Exception) as exc_info:
            client.send_prompt("Test prompt", config)

        assert "API error" in str(exc_info.value)


class TestOpenAIProvider:
    """Tests for the OpenAIProvider class."""

    @patch("openai.OpenAI")
    def test_initialization(self, mock_openai):
        """Test provider initialization."""
        provider = OpenAIProvider(model_name="gpt-4", config=ModelConfig(api_key="test-key"))

        assert provider.model_name == "gpt-4"
        assert provider.config.api_key == "test-key"
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch("openai.OpenAI")
    def test_generate(self, mock_openai):
        """Test text generation."""
        # Set up mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Generated text"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(
            model_name="gpt-4", config=ModelConfig(temperature=0.7, max_tokens=200)
        )
        response = provider.generate("Test prompt")

        assert response == "Generated text"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=200,
        )

    @patch("openai.OpenAI")
    def test_generate_error(self, mock_openai):
        """Test error handling during generation."""
        # Mock generation error
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Simulate an error during generation
        mock_client.chat.completions.create.side_effect = Exception("Generation error")

        provider = OpenAIProvider()

        with pytest.raises(RuntimeError) as exc_info:
            provider.generate("Test prompt")

        assert "Failed to generate text" in str(exc_info.value)
        assert "Generation error" in str(exc_info.value)

    @patch("openai.OpenAI")
    def test_invoke(self, mock_openai):
        """Test the invoke method."""
        # Set up mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Invoked response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider()
        response = provider.invoke("Test prompt")

        assert response == "Invoked response"
