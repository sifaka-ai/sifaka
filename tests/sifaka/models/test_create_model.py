"""
Tests for the create_model function.
"""

import pytest
from unittest.mock import patch

from sifaka.models.base import create_model
from sifaka.errors import ModelNotFoundError


class TestCreateModel:
    """Tests for the create_model function."""
    
    @patch("sifaka.models.base.OpenAIModel")
    def test_create_openai_model(self, mock_openai_model):
        """Test creating an OpenAI model."""
        # Set up the mock
        mock_openai_model.return_value = "openai_model_instance"
        
        # Call create_model
        model = create_model("openai", "gpt-4", api_key="test-key")
        
        # Check the result
        assert model == "openai_model_instance"
        
        # Check that the model was created correctly
        mock_openai_model.assert_called_once_with(model_name="gpt-4", api_key="test-key")
    
    @patch("sifaka.models.base.AnthropicModel")
    def test_create_anthropic_model(self, mock_anthropic_model):
        """Test creating an Anthropic model."""
        # Set up the mock
        mock_anthropic_model.return_value = "anthropic_model_instance"
        
        # Call create_model
        model = create_model("anthropic", "claude-3-opus-20240229", api_key="test-key")
        
        # Check the result
        assert model == "anthropic_model_instance"
        
        # Check that the model was created correctly
        mock_anthropic_model.assert_called_once_with(
            model_name="claude-3-opus-20240229", api_key="test-key"
        )
    
    @patch("sifaka.models.base.GeminiModel")
    def test_create_gemini_model(self, mock_gemini_model):
        """Test creating a Gemini model."""
        # Set up the mock
        mock_gemini_model.return_value = "gemini_model_instance"
        
        # Call create_model
        model = create_model("gemini", "gemini-pro", api_key="test-key")
        
        # Check the result
        assert model == "gemini_model_instance"
        
        # Check that the model was created correctly
        mock_gemini_model.assert_called_once_with(model_name="gemini-pro", api_key="test-key")
    
    def test_create_unknown_model(self):
        """Test that an error is raised for an unknown provider."""
        with pytest.raises(ModelNotFoundError) as excinfo:
            create_model("unknown", "model-name")
        
        assert "Provider 'unknown' not found" in str(excinfo.value)
