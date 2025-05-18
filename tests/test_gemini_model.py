"""
Tests for the Gemini model implementation.

This module contains tests for the Gemini model implementation in Sifaka.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from sifaka.errors import ModelError, ModelAPIError, ConfigurationError


# Mock the google.generativeai module for testing
class MockGenerativeAI:
    """Mock for the google.generativeai module."""

    class GenerativeModel:
        """Mock for the GenerativeModel class."""

        def __init__(self, model_name=None):
            self.model_name = model_name
            self.generate_content = MagicMock()
            self.count_tokens = MagicMock()

    class GenerationConfig:
        """Mock for the GenerationConfig class."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def configure(self, api_key=None):
        """Mock for the configure function."""
        self.api_key = api_key


# Mock the google.api_core.exceptions module
class MockExceptions:
    """Mock for the google.api_core.exceptions module."""

    class GoogleAPIError(Exception):
        """Mock for the GoogleAPIError class."""

        pass

    class ResourceExhausted(GoogleAPIError):
        """Mock for the ResourceExhausted class."""

        pass

    class InvalidArgument(GoogleAPIError):
        """Mock for the InvalidArgument class."""

        pass


# Patch the necessary modules
@pytest.fixture
def mock_genai():
    """Fixture to mock the google.generativeai module."""
    mock_genai = MockGenerativeAI()
    mock_exceptions = MockExceptions()

    with patch.dict(
        "sys.modules",
        {"google.generativeai": mock_genai, "google.api_core.exceptions": mock_exceptions},
    ):
        # Set the GEMINI_AVAILABLE flag to True
        with patch("sifaka.models.gemini.GEMINI_AVAILABLE", True):
            # Also patch the specific imports
            with patch("sifaka.models.gemini.genai", mock_genai):
                with patch("sifaka.models.gemini.GoogleAPIError", mock_exceptions.GoogleAPIError):
                    with patch(
                        "sifaka.models.gemini.ResourceExhausted", mock_exceptions.ResourceExhausted
                    ):
                        with patch(
                            "sifaka.models.gemini.InvalidArgument", mock_exceptions.InvalidArgument
                        ):
                            with patch(
                                "sifaka.models.gemini.GenAIGenerationConfig",
                                mock_genai.GenerationConfig,
                            ):
                                yield mock_genai


class TestGeminiModel:
    """Tests for the GeminiModel class."""

    def test_init_without_gemini_package(self):
        """Test initialization when the Gemini package is not available."""
        # Import here to avoid early import errors
        from sifaka.models.gemini import GeminiModel

        # Patch GEMINI_AVAILABLE to False
        with patch("sifaka.models.gemini.GEMINI_AVAILABLE", False):
            with pytest.raises(ConfigurationError) as excinfo:
                GeminiModel("gemini-pro", api_key="test-key")

            assert "Google Generative AI package not installed" in str(excinfo.value)
            assert "pip install google-generativeai" in str(excinfo.value)

    def test_init_without_api_key(self, mock_genai):
        """Test initialization without an API key."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        # Remove any existing API key from the environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModelError) as excinfo:
                GeminiModel("gemini-pro")

            assert "Google API key not provided" in str(excinfo.value)
            assert "GOOGLE_API_KEY environment variable" in str(excinfo.value)

    def test_init_with_api_key_param(self, mock_genai):
        """Test initialization with an API key parameter."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        model = GeminiModel("gemini-pro", api_key="test-key")

        assert model.model_name == "gemini-pro"
        assert model.api_key == "test-key"
        assert mock_genai.api_key == "test-key"  # Should have configured the API

    def test_init_with_api_key_env(self, mock_genai):
        """Test initialization with an API key from the environment."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        # Set the API key in the environment
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key"}):
            model = GeminiModel("gemini-pro")

            assert model.model_name == "gemini-pro"
            assert model.api_key == "env-test-key"
            assert mock_genai.api_key == "env-test-key"  # Should have configured the API

    def test_init_with_options(self, mock_genai):
        """Test initialization with additional options."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        model = GeminiModel("gemini-pro", api_key="test-key", temperature=0.5, max_tokens=100)

        assert model.model_name == "gemini-pro"
        assert model.api_key == "test-key"
        assert model.options == {"temperature": 0.5, "max_tokens": 100}

    def test_generate_success(self, mock_genai):
        """Test successful text generation."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        # Create a mock response
        mock_response = MagicMock()
        mock_response.text = "Generated text response"

        # Set up the mock model
        model = GeminiModel("gemini-pro", api_key="test-key")
        model.model.generate_content.return_value = mock_response

        # Generate text
        result = model.generate("Test prompt", temperature=0.7)

        # Check the result
        assert result == "Generated text response"

        # Check that the model was called correctly
        model.model.generate_content.assert_called_once()

        # Check that the generation config was created with the right parameters
        args, kwargs = model.model.generate_content.call_args
        assert args[0] == "Test prompt"
        assert "generation_config" in kwargs

    def test_generate_with_options_conversion(self, mock_genai):
        """Test option conversion during generation."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        # Create a mock response
        mock_response = MagicMock()
        mock_response.text = "Generated text response"

        # Set up the mock model
        model = GeminiModel("gemini-pro", api_key="test-key")
        model.model.generate_content.return_value = mock_response

        # Generate text with options that need conversion
        result = model.generate(
            "Test prompt",
            max_tokens=100,  # Should be converted to max_output_tokens
            temperature=0.8,
        )

        # Check that the generation config was created with converted options
        args, kwargs = model.model.generate_content.call_args
        generation_config = kwargs["generation_config"]

        # In our mock, the config is just a simple object with attributes
        assert not hasattr(generation_config, "max_tokens")
        assert hasattr(generation_config, "max_output_tokens")
        assert generation_config.max_output_tokens == 100
        assert generation_config.temperature == 0.8

    def test_generate_resource_exhausted(self, mock_genai):
        """Test handling of resource exhausted errors during generation."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel
        from sifaka.models.gemini import ResourceExhausted

        # Set up the mock model to raise a resource exhausted error
        model = GeminiModel("gemini-pro", api_key="test-key")
        model.model.generate_content.side_effect = ResourceExhausted("Rate limit exceeded")

        # Generate text and check for the correct error
        with pytest.raises(ModelAPIError) as excinfo:
            model.generate("Test prompt")

        error = excinfo.value
        assert "rate limit exceeded" in str(error).lower()
        # The model name is in the error message, not as an attribute
        assert "gemini-pro" in str(error).lower()
        # The component is in the error message
        assert "geminimodel" in str(error).lower()
        # The operation is in the error message
        assert "generation" in str(error).lower()
        # The suggestions are in the error message
        assert "suggestions" in str(error).lower()

    def test_generate_invalid_argument(self, mock_genai):
        """Test handling of invalid argument errors during generation."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel
        from sifaka.models.gemini import InvalidArgument

        # Set up the mock model to raise an invalid argument error
        model = GeminiModel("gemini-pro", api_key="test-key")
        model.model.generate_content.side_effect = InvalidArgument("Invalid prompt")

        # Generate text and check for the correct error
        with pytest.raises(ModelAPIError) as excinfo:
            model.generate("Test prompt")

        error = excinfo.value
        # The error message contains "Invalid prompt"
        assert "invalid prompt" in str(error).lower()
        # The model name is in the error message
        assert "gemini-pro" in str(error).lower()
        # The component is in the error message
        assert "geminimodel" in str(error).lower()
        # The operation is in the error message
        assert "generation" in str(error).lower()
        # The suggestions are in the error message
        assert "suggestions" in str(error).lower()

    def test_count_tokens(self, mock_genai):
        """Test token counting."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        # Create a mock response
        mock_response = MagicMock()
        mock_response.total_tokens = 15

        # Set up the mock model
        model = GeminiModel("gemini-pro", api_key="test-key")
        model.model.count_tokens.return_value = mock_response

        # Count tokens
        token_count = model.count_tokens("Test text")

        # Check the result
        assert token_count == 15

        # Check that the model was called correctly
        model.model.count_tokens.assert_called_once_with("Test text")

    def test_count_tokens_error(self, mock_genai):
        """Test error handling during token counting."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel
        from sifaka.models.gemini import GoogleAPIError

        # Set up the mock model to raise an error
        model = GeminiModel("gemini-pro", api_key="test-key")
        model.model.count_tokens.side_effect = GoogleAPIError("Token counting failed")

        # Count tokens and check for the correct error
        with pytest.raises(ModelError) as excinfo:
            model.count_tokens("Test text")

        error = excinfo.value
        assert "counting tokens" in str(error).lower()
        assert error.component == "GeminiModel"
        assert error.operation == "token_counting"
        assert any("api" in suggestion.lower() for suggestion in error.suggestions)

    def test_configure(self, mock_genai):
        """Test model configuration."""
        # Import here after mocking
        from sifaka.models.gemini import GeminiModel

        # Create a model with initial options
        model = GeminiModel("gemini-pro", api_key="test-key", temperature=0.7)

        assert model.options == {"temperature": 0.7}

        # Configure with new options
        model.configure(temperature=0.5, max_tokens=100)

        # Check that the options were updated
        assert model.options == {"temperature": 0.5, "max_tokens": 100}


class TestGeminiModelFactory:
    """Tests for the Gemini model factory function."""

    def test_create_gemini_model_success(self, mock_genai):
        """Test successful creation of a Gemini model."""
        # Import here after mocking
        from sifaka.models.gemini import create_gemini_model

        # Create a model
        model = create_gemini_model("gemini-pro", api_key="test-key")

        # Check the model
        assert model.model_name == "gemini-pro"
        assert model.api_key == "test-key"

    def test_create_gemini_model_configuration_error(self, mock_genai):
        """Test handling of configuration errors during model creation."""
        # Import here after mocking
        from sifaka.models.gemini import create_gemini_model

        # Patch GeminiModel to raise a ConfigurationError
        with patch(
            "sifaka.models.gemini.GeminiModel",
            side_effect=ConfigurationError(
                message="Configuration error", component="Test", operation="test"
            ),
        ):
            # Create a model and check for the correct error
            with pytest.raises(ConfigurationError) as excinfo:
                create_gemini_model("gemini-pro", api_key="test-key")

            assert "failed to create gemini model" in str(excinfo.value).lower()
            assert excinfo.value.component == "GeminiModelFactory"
            assert excinfo.value.operation == "create_model"
