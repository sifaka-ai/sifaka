"""
Tests for the OpenAI model implementation.
"""

import os
import pytest
from unittest.mock import Mock, patch

from sifaka.models.openai import OpenAIModel
from sifaka.errors import ModelError, ModelAPIError, ConfigurationError


class TestOpenAIModel:
    """Tests for the OpenAIModel class."""
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", False)
    def test_openai_not_installed(self):
        """Test that an error is raised if the OpenAI package is not installed."""
        with pytest.raises(ConfigurationError):
            OpenAIModel("gpt-4")
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", True)
    def test_no_api_key(self):
        """Test that an error is raised if no API key is provided."""
        # Save the original environment variable
        original_api_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            # Remove the environment variable if it exists
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            
            with pytest.raises(ModelError):
                OpenAIModel("gpt-4")
        finally:
            # Restore the original environment variable
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", True)
    @patch("sifaka.models.openai.OpenAI")
    def test_initialization(self, mock_openai):
        """Test that the model is initialized correctly."""
        model = OpenAIModel("gpt-4", api_key="test-key")
        
        assert model.model_name == "gpt-4"
        assert model.api_key == "test-key"
        mock_openai.assert_called_once_with(api_key="test-key", organization=None)
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", True)
    @patch("sifaka.models.openai.OpenAI")
    def test_generate(self, mock_openai):
        """Test that generate calls the OpenAI API correctly."""
        # Set up the mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated text"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create the model and call generate
        model = OpenAIModel("gpt-4", api_key="test-key")
        result = model.generate("Test prompt", temperature=0.7)
        
        # Check the result
        assert result == "Generated text"
        
        # Check that the API was called correctly
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7
        )
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", True)
    @patch("sifaka.models.openai.OpenAI")
    def test_generate_with_options(self, mock_openai):
        """Test that generate passes options to the OpenAI API."""
        # Set up the mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated text"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create the model with default options
        model = OpenAIModel("gpt-4", api_key="test-key", temperature=0.5)
        
        # Call generate with additional options
        result = model.generate("Test prompt", temperature=0.7, max_tokens=100)
        
        # Check the result
        assert result == "Generated text"
        
        # Check that the API was called with the merged options
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=100
        )
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", True)
    @patch("sifaka.models.openai.OpenAI")
    @patch("sifaka.models.openai.RateLimitError", Exception)
    def test_generate_rate_limit_error(self, mock_openai):
        """Test that a rate limit error is handled correctly."""
        # Set up the mock to raise a rate limit error
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        
        # Create the model and call generate
        model = OpenAIModel("gpt-4", api_key="test-key")
        
        with pytest.raises(ModelAPIError) as excinfo:
            model.generate("Test prompt")
        
        assert "rate limit" in str(excinfo.value).lower()
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", True)
    @patch("sifaka.models.openai.OpenAI")
    @patch("sifaka.models.openai.tiktoken")
    def test_count_tokens(self, mock_tiktoken, mock_openai):
        """Test that count_tokens calls tiktoken correctly."""
        # Set up the mock
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        
        # Create the model and call count_tokens
        model = OpenAIModel("gpt-4", api_key="test-key")
        count = model.count_tokens("Test text")
        
        # Check the result
        assert count == 4
        
        # Check that tiktoken was called correctly
        mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")
        mock_encoding.encode.assert_called_once_with("Test text")
    
    @patch("sifaka.models.openai.OPENAI_AVAILABLE", True)
    @patch("sifaka.models.openai.OpenAI")
    @patch("sifaka.models.openai.tiktoken")
    def test_count_tokens_fallback(self, mock_tiktoken, mock_openai):
        """Test that count_tokens falls back to cl100k_base if the model encoding is not found."""
        # Set up the mock to raise a KeyError for the model
        mock_tiktoken.encoding_for_model.side_effect = KeyError("Model not found")
        
        # Set up the fallback encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4]
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        # Create the model and call count_tokens
        model = OpenAIModel("unknown-model", api_key="test-key")
        count = model.count_tokens("Test text")
        
        # Check the result
        assert count == 4
        
        # Check that tiktoken was called correctly
        mock_tiktoken.encoding_for_model.assert_called_once_with("unknown-model")
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        mock_encoding.encode.assert_called_once_with("Test text")
