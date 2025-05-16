"""
Tests for the Anthropic model implementation.
"""

import os
import pytest
from unittest.mock import Mock, patch

from sifaka.models.anthropic import AnthropicModel
from sifaka.errors import ModelError, ModelAPIError, ConfigurationError


class TestAnthropicModel:
    """Tests for the AnthropicModel class."""
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", False)
    def test_anthropic_not_installed(self):
        """Test that an error is raised if the Anthropic package is not installed."""
        with pytest.raises(ConfigurationError):
            AnthropicModel("claude-3-opus-20240229")
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True)
    def test_no_api_key(self):
        """Test that an error is raised if no API key is provided."""
        # Save the original environment variable
        original_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        try:
            # Remove the environment variable if it exists
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            
            with pytest.raises(ModelError):
                AnthropicModel("claude-3-opus-20240229")
        finally:
            # Restore the original environment variable
            if original_api_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_api_key
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True)
    @patch("sifaka.models.anthropic.Anthropic")
    def test_initialization(self, mock_anthropic):
        """Test that the model is initialized correctly."""
        model = AnthropicModel("claude-3-opus-20240229", api_key="test-key")
        
        assert model.model_name == "claude-3-opus-20240229"
        assert model.api_key == "test-key"
        mock_anthropic.assert_called_once_with(api_key="test-key")
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True)
    @patch("sifaka.models.anthropic.Anthropic")
    def test_generate(self, mock_anthropic):
        """Test that generate calls the Anthropic API correctly."""
        # Set up the mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Generated text"
        mock_response.content = [mock_content]
        
        mock_client.messages.create.return_value = mock_response
        
        # Create the model and call generate
        model = AnthropicModel("claude-3-opus-20240229", api_key="test-key")
        result = model.generate("Test prompt", temperature=0.7)
        
        # Check the result
        assert result == "Generated text"
        
        # Check that the API was called correctly
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7
        )
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True)
    @patch("sifaka.models.anthropic.Anthropic")
    def test_generate_with_options(self, mock_anthropic):
        """Test that generate passes options to the Anthropic API."""
        # Set up the mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Generated text"
        mock_response.content = [mock_content]
        
        mock_client.messages.create.return_value = mock_response
        
        # Create the model with default options
        model = AnthropicModel("claude-3-opus-20240229", api_key="test-key", temperature=0.5)
        
        # Call generate with additional options
        result = model.generate("Test prompt", temperature=0.7, max_tokens=100)
        
        # Check the result
        assert result == "Generated text"
        
        # Check that the API was called with the merged options
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens_to_sample=100
        )
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True)
    @patch("sifaka.models.anthropic.Anthropic")
    def test_generate_with_stop(self, mock_anthropic):
        """Test that generate converts stop to stop_sequences."""
        # Set up the mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Generated text"
        mock_response.content = [mock_content]
        
        mock_client.messages.create.return_value = mock_response
        
        # Create the model and call generate with stop
        model = AnthropicModel("claude-3-opus-20240229", api_key="test-key")
        result = model.generate("Test prompt", stop=["END"])
        
        # Check the result
        assert result == "Generated text"
        
        # Check that the API was called with stop_sequences
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Test prompt"}],
            stop_sequences=["END"]
        )
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True)
    @patch("sifaka.models.anthropic.Anthropic")
    @patch("sifaka.models.anthropic.RateLimitError", Exception)
    def test_generate_rate_limit_error(self, mock_anthropic):
        """Test that a rate limit error is handled correctly."""
        # Set up the mock to raise a rate limit error
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_client.messages.create.side_effect = Exception("Rate limit exceeded")
        
        # Create the model and call generate
        model = AnthropicModel("claude-3-opus-20240229", api_key="test-key")
        
        with pytest.raises(ModelAPIError) as excinfo:
            model.generate("Test prompt")
        
        assert "rate limit" in str(excinfo.value).lower()
    
    @patch("sifaka.models.anthropic.ANTHROPIC_AVAILABLE", True)
    @patch("sifaka.models.anthropic.Anthropic")
    def test_count_tokens(self, mock_anthropic):
        """Test that count_tokens calls the Anthropic API correctly."""
        # Set up the mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_client.count_tokens.return_value = 4
        
        # Create the model and call count_tokens
        model = AnthropicModel("claude-3-opus-20240229", api_key="test-key")
        count = model.count_tokens("Test text")
        
        # Check the result
        assert count == 4
        
        # Check that the API was called correctly
        mock_client.count_tokens.assert_called_once_with("Test text")
