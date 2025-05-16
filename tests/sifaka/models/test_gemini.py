"""
Tests for the Gemini model implementation.
"""

import os
import pytest
from unittest.mock import Mock, patch

from sifaka.models.gemini import GeminiModel
from sifaka.errors import ModelError, ModelAPIError, ConfigurationError


class TestGeminiModel:
    """Tests for the GeminiModel class."""
    
    @patch("sifaka.models.gemini.GEMINI_AVAILABLE", False)
    def test_gemini_not_installed(self):
        """Test that an error is raised if the Google Generative AI package is not installed."""
        with pytest.raises(ConfigurationError):
            GeminiModel("gemini-pro")
    
    @patch("sifaka.models.gemini.GEMINI_AVAILABLE", True)
    def test_no_api_key(self):
        """Test that an error is raised if no API key is provided."""
        # Save the original environment variable
        original_api_key = os.environ.get("GOOGLE_API_KEY")
        
        try:
            # Remove the environment variable if it exists
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]
            
            with pytest.raises(ModelError):
                GeminiModel("gemini-pro")
        finally:
            # Restore the original environment variable
            if original_api_key is not None:
                os.environ["GOOGLE_API_KEY"] = original_api_key
    
    @patch("sifaka.models.gemini.GEMINI_AVAILABLE", True)
    @patch("sifaka.models.gemini.genai")
    def test_initialization(self, mock_genai):
        """Test that the model is initialized correctly."""
        model = GeminiModel("gemini-pro", api_key="test-key")
        
        assert model.model_name == "gemini-pro"
        assert model.api_key == "test-key"
        mock_genai.configure.assert_called_once_with(api_key="test-key")
        mock_genai.GenerativeModel.assert_called_once_with(model_name="gemini-pro")
    
    @patch("sifaka.models.gemini.GEMINI_AVAILABLE", True)
    @patch("sifaka.models.gemini.genai")
    def test_generate(self, mock_genai):
        """Test that generate calls the Gemini API correctly."""
        # Set up the mock
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        mock_response = Mock()
        mock_response.text = "Generated text"
        
        mock_model.generate_content.return_value = mock_response
        
        # Create the model and call generate
        model = GeminiModel("gemini-pro", api_key="test-key")
        result = model.generate("Test prompt", temperature=0.7)
        
        # Check the result
        assert result == "Generated text"
        
        # Check that the API was called correctly
        mock_model.generate_content.assert_called_once_with(
            "Test prompt",
            generation_config={"temperature": 0.7}
        )
    
    @patch("sifaka.models.gemini.GEMINI_AVAILABLE", True)
    @patch("sifaka.models.gemini.genai")
    def test_generate_with_options(self, mock_genai):
        """Test that generate passes options to the Gemini API."""
        # Set up the mock
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        mock_response = Mock()
        mock_response.text = "Generated text"
        
        mock_model.generate_content.return_value = mock_response
        
        # Create the model with default options
        model = GeminiModel("gemini-pro", api_key="test-key", temperature=0.5)
        
        # Call generate with additional options
        result = model.generate("Test prompt", temperature=0.7, max_tokens=100)
        
        # Check the result
        assert result == "Generated text"
        
        # Check that the API was called with the merged options
        mock_model.generate_content.assert_called_once_with(
            "Test prompt",
            generation_config={"temperature": 0.7, "max_output_tokens": 100}
        )
    
    @patch("sifaka.models.gemini.GEMINI_AVAILABLE", True)
    @patch("sifaka.models.gemini.genai")
    @patch("sifaka.models.gemini.ResourceExhausted", Exception)
    def test_generate_rate_limit_error(self, mock_genai):
        """Test that a rate limit error is handled correctly."""
        # Set up the mock to raise a rate limit error
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        mock_model.generate_content.side_effect = Exception("Resource exhausted")
        
        # Create the model and call generate
        model = GeminiModel("gemini-pro", api_key="test-key")
        
        with pytest.raises(ModelAPIError) as excinfo:
            model.generate("Test prompt")
        
        assert "rate limit" in str(excinfo.value).lower()
    
    @patch("sifaka.models.gemini.GEMINI_AVAILABLE", True)
    @patch("sifaka.models.gemini.genai")
    def test_count_tokens(self, mock_genai):
        """Test that count_tokens calls the Gemini API correctly."""
        # Set up the mock
        mock_result = Mock()
        mock_result.total_tokens = 4
        
        mock_genai.count_tokens.return_value = mock_result
        
        # Create the model and call count_tokens
        model = GeminiModel("gemini-pro", api_key="test-key")
        count = model.count_tokens("Test text")
        
        # Check the result
        assert count == 4
        
        # Check that the API was called correctly
        mock_genai.count_tokens.assert_called_once_with(
            model="gemini-pro",
            prompt="Test text"
        )
