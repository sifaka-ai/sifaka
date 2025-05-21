"""
Tests for the Gemini model.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from core.thought import Thought
from models.factory import ModelConfigurationError, ModelError
from models.gemini_model import GEMINI_AVAILABLE, GeminiModel

# Skip all tests if Gemini package is not available
pytestmark = pytest.mark.skipif(
    not GEMINI_AVAILABLE, reason="Google Generative AI package not installed"
)


@pytest.fixture
def mock_genai():
    """Create a mock for the Google Generative AI package."""
    with patch("models.gemini_model.genai") as mock:
        # Mock the GenerativeModel class
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "Generated text from Gemini"
        mock.GenerativeModel.return_value = mock_model

        # Mock the GenerationConfig class
        mock.types.GenerationConfig = MagicMock

        yield mock


@pytest.fixture
def thought():
    """Create a thought for testing."""
    return Thought(prompt="Test prompt")


@pytest.fixture
def thought_with_context():
    """Create a thought with context for testing."""
    thought = Thought(prompt="Test prompt with context")
    thought.add_retrieved_context(
        source="test_source",
        content="test_context",
        metadata={"key": "value"},
        relevance_score=0.8,
    )
    return thought


@pytest.fixture
def thought_with_validation():
    """Create a thought with validation results for testing."""
    thought = Thought(prompt="Test prompt with validation")
    thought.add_validation_result(
        validator_name="test_validator",
        passed=False,
        score=0.5,
        details={"key": "value"},
        message="Test validation message",
    )
    return thought


@pytest.fixture
def thought_with_feedback():
    """Create a thought with critic feedback for testing."""
    thought = Thought(prompt="Test prompt with feedback")
    thought.add_critic_feedback(
        critic_name="test_critic",
        feedback="Test feedback",
        suggestions=["Suggestion 1", "Suggestion 2"],
        details={"key": "value"},
    )
    return thought


def test_gemini_model_initialization(mock_genai):
    """Test that a Gemini model can be initialized."""
    # Set API key in environment
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
        model = GeminiModel()

        assert model.name == "gemini-gemini-pro"
        assert model.model_name == "gemini-pro"
        assert model.temperature == 0.7
        assert model.max_tokens == 1000
        assert model.api_key == "test-api-key"
        assert model.system_prompt == "You are a helpful assistant."


def test_gemini_model_initialization_with_explicit_api_key(mock_genai):
    """Test that a Gemini model can be initialized with an explicit API key."""
    model = GeminiModel(api_key="explicit-api-key")

    assert model.api_key == "explicit-api-key"


def test_gemini_model_initialization_without_api_key(mock_genai):
    """Test that initializing a Gemini model without an API key raises an error."""
    # Clear API key from environment
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ModelError):
            GeminiModel()


def test_gemini_model_generate(mock_genai, thought):
    """Test that a Gemini model can generate text."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
        model = GeminiModel()
        result = model.generate(thought)

        assert result == "Generated text from Gemini"
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once()


def test_gemini_model_generate_with_context(mock_genai, thought_with_context):
    """Test that a Gemini model can generate text with context."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
        model = GeminiModel()
        result = model.generate(thought_with_context)

        assert result == "Generated text from Gemini"

        # Check that the context was included in the prompt
        call_args = mock_genai.GenerativeModel.return_value.generate_content.call_args[0][0]
        assert "test_context" in call_args
        assert "test_source" in call_args


def test_gemini_model_generate_with_validation(mock_genai, thought_with_validation):
    """Test that a Gemini model can generate text with validation results."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
        model = GeminiModel()
        result = model.generate(thought_with_validation)

        assert result == "Generated text from Gemini"

        # Check that the validation results were included in the prompt
        call_args = mock_genai.GenerativeModel.return_value.generate_content.call_args[0][0]
        assert "test_validator" in call_args
        assert "Test validation message" in call_args


def test_gemini_model_generate_with_feedback(mock_genai, thought_with_feedback):
    """Test that a Gemini model can generate text with critic feedback."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}):
        model = GeminiModel()
        result = model.generate(thought_with_feedback)

        assert result == "Generated text from Gemini"

        # Check that the feedback was included in the prompt
        call_args = mock_genai.GenerativeModel.return_value.generate_content.call_args[0][0]
        assert "test_critic" in call_args
        assert "Test feedback" in call_args
        assert "Suggestion 1" in call_args
        assert "Suggestion 2" in call_args
