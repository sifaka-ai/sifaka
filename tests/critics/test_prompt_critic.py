"""
Consolidated test suite for the PromptCritic class.

This file combines tests previously distributed across multiple files:
- Basic initialization and configuration
- Core functionality (validation, critique, improvement)
- Error handling
- Response parsing
"""

import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock

from sifaka.critics.prompt import (
    PromptCritic,
    PromptCriticConfig,
    DefaultPromptFactory,
    create_prompt_critic,
    CriticMetadata,
    DEFAULT_SYSTEM_PROMPT,
    LanguageModel,
)


class MockLanguageModel:
    """Mock language model for testing."""

    def __init__(self, responses=None):
        """Initialize with predefined responses."""
        self.responses = responses or {}
        self.invocation_count = 0
        self.last_prompt = None
        self.calls = []
        self.model_name_value = "mock-model"

    def generate(self, prompt):
        """Generate text based on the prompt."""
        self.invocation_count += 1
        self.last_prompt = prompt
        self.calls.append(("generate", prompt))
        return self.responses.get("generate", "Generated response")

    def invoke(self, prompt):
        """Invoke the model with structured output."""
        self.invocation_count += 1
        self.last_prompt = prompt
        self.calls.append(("invoke", prompt))

        if "Validate" in prompt or "VALIDATE" in prompt:
            return self.responses.get("validate", {"valid": True, "reason": "Test reason"})
        elif "Critique" in prompt or "CRITIQUE" in prompt:
            return self.responses.get(
                "critique",
                {
                    "score": 0.8,
                    "feedback": "Test feedback",
                    "issues": ["Issue 1", "Issue 2"],
                    "suggestions": ["Suggestion 1", "Suggestion 2"],
                },
            )
        elif "Improve" in prompt or "IMPROVE" in prompt:
            return self.responses.get("improve", {"improved_text": "Improved text"})
        else:
            return self.responses.get("default", "Default response")

    async def ainvoke(self, prompt):
        """Async invoke the model with a prompt."""
        self.calls.append(("ainvoke", prompt))

        # Handle different prompt types
        if "validate" in prompt.lower():
            return self.responses.get("avalidate", self.responses.get("validate", {"valid": True}))
        elif "critique" in prompt.lower():
            return self.responses.get(
                "acritique",
                self.responses.get(
                    "critique",
                    {"score": 0.8, "feedback": "Good text", "issues": [], "suggestions": []},
                ),
            )
        elif "improve" in prompt.lower():
            return self.responses.get(
                "aimprove", self.responses.get("improve", {"improved_text": "Improved version"})
            )
        else:
            return self.responses.get("default", "Default response")

    @property
    def model_name(self):
        """Get the model name."""
        return self.model_name_value


@pytest.fixture
def mock_model():
    """Create a mock language model."""
    return MockLanguageModel()


@pytest.fixture
def prompt_critic(mock_model):
    """Create a PromptCritic with a mock model."""
    config = PromptCriticConfig(
        name="test_critic",
        description="Test critic",
        system_prompt="You are a test critic",
        temperature=0.5,
        max_tokens=500,
        min_confidence=0.6,
        max_attempts=2,
    )
    return PromptCritic(
        name="test_critic", description="Test critic", llm_provider=mock_model, config=config
    )


#
# Configuration Tests
#
class TestPromptCriticConfig:
    """Tests for PromptCriticConfig."""

    def test_valid_config(self):
        """Test valid configuration initialization."""
        config = PromptCriticConfig(
            name="test",
            description="Test config",
            system_prompt="Test prompt",
            temperature=0.5,
            max_tokens=500,
        )

        assert config.name == "test"
        assert config.description == "Test config"
        assert config.system_prompt == "Test prompt"
        assert config.temperature == 0.5
        assert config.max_tokens == 500

    def test_default_values(self):
        """Test default values for PromptCriticConfig."""
        config = PromptCriticConfig(name="test_config", description="Test description")

        assert config.name == "test_config"
        assert config.description == "Test description"
        assert config.system_prompt == "You are an expert editor that improves text."
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.min_confidence == 0.7
        assert config.max_attempts == 3

    def test_invalid_config(self):
        """Test invalid configuration validation."""
        with pytest.raises(ValueError):
            PromptCriticConfig(name="test", description="Test", system_prompt="")

        with pytest.raises(ValueError):
            PromptCriticConfig(name="test", description="Test", temperature=1.5)

        with pytest.raises(ValueError):
            PromptCriticConfig(name="test", description="Test", max_tokens=0)

        with pytest.raises(ValueError):
            PromptCriticConfig(name="test", description="Test", system_prompt="   ")


#
# Initialization Tests
#
class TestPromptCriticInitialization:
    """Tests for PromptCritic initialization."""

    def test_initialization(self, mock_model):
        """Test critic initialization."""
        critic = PromptCritic(
            name="test_critic", description="Test critic", llm_provider=mock_model
        )

        assert critic.config.name == "test_critic"
        assert critic.config.description == "Test critic"
        assert critic._model == mock_model

    def test_initialization_with_config(self, mock_model):
        """Test initialization with explicit config."""
        config = PromptCriticConfig(
            name="config_critic",
            description="Config test",
            system_prompt="Custom system prompt",
            temperature=0.3,
            max_tokens=300,
        )

        critic = PromptCritic(llm_provider=mock_model, config=config)

        assert critic.config.name == "config_critic"
        assert critic.config.description == "Config test"
        assert critic.config.system_prompt == "Custom system prompt"
        assert critic.config.temperature == 0.3
        assert critic.config.max_tokens == 300

    def test_initialization_no_model(self):
        """Test initialization raises error when no model is provided."""
        with pytest.raises(Exception):
            PromptCritic()

    def test_backward_compatibility_with_model_param(self, mock_model):
        """Test backward compatibility using model instead of llm_provider."""
        critic = PromptCritic(
            name="compat_test",
            description="Backward compatibility test",
            model=mock_model,
        )

        assert critic._model == mock_model

        # Ensure it works with validation
        mock_model.responses["validate"] = {"valid": True}
        result = critic.validate("Test text")
        assert result is True


#
# Core Functionality Tests
#
class TestPromptCriticFunctionality:
    """Tests for core functionality of PromptCritic."""

    def test_validate(self, prompt_critic, mock_model):
        """Test validation with predefined response."""
        mock_model.responses["validate"] = {"valid": True, "reason": "Text is good"}

        result = prompt_critic.validate("Test text")
        assert result is True
        assert "VALIDATE" in mock_model.last_prompt.upper()
        assert "Test text" in mock_model.last_prompt

    def test_validate_false(self, prompt_critic, mock_model):
        """Test validation that fails."""
        mock_model.responses["validate"] = {"valid": False, "reason": "Text is bad"}

        result = prompt_critic.validate("Bad text")
        assert result is False
        assert "VALIDATE" in mock_model.last_prompt.upper()
        assert "Bad text" in mock_model.last_prompt

    def test_validate_empty_text(self, prompt_critic):
        """Test validation with empty text."""
        with pytest.raises(ValueError) as exc_info:
            prompt_critic.validate("")

        assert "text must be a non-empty string" in str(exc_info.value)

        with pytest.raises(ValueError):
            prompt_critic.validate("   ")

    def test_validate_with_string_response(self, prompt_critic, mock_model):
        """Test validation with string response."""
        mock_model.responses["validate"] = "valid: true\nreason: The text is correct."
        result = prompt_critic.validate("Test text")
        assert result is True

        mock_model.responses["validate"] = "valid: false\nreason: The text is incorrect."
        result = prompt_critic.validate("Bad text")
        assert result is False

    def test_validate_fallback_to_critique(self, prompt_critic, mock_model):
        """Test validation fallback to critique when structured response parsing fails."""
        # Provide a string response that doesn't match expected format
        mock_model.responses["validate"] = "This is just some random text without VALID: marker"

        # Set up the critique response for the fallback
        mock_model.responses["critique"] = {
            "score": 0.65,
            "feedback": "Fallback critique",
            "issues": ["Issue"],
            "suggestions": ["Suggestion"],
        }

        result = prompt_critic.validate("Test text")

        # Should use score from critique result compared to min_confidence (0.6)
        assert result is True

        # Test below threshold
        mock_model.responses["critique"] = {
            "score": 0.5,
            "feedback": "Fallback critique below threshold",
            "issues": ["Issue"],
            "suggestions": ["Suggestion"],
        }

        result = prompt_critic.validate("Test text")
        assert result is False

    def test_critique(self, prompt_critic, mock_model):
        """Test critique functionality."""
        mock_model.responses["critique"] = {
            "score": 0.75,
            "feedback": "Good text with some issues",
            "issues": ["Issue 1", "Issue 2"],
            "suggestions": ["Suggestion 1"],
        }

        result = prompt_critic.critique("Test text to critique")

        assert result["score"] == 0.75
        assert result["feedback"] == "Good text with some issues"
        assert len(result["issues"]) == 2
        assert result["issues"][0] == "Issue 1"
        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0] == "Suggestion 1"
        assert "CRITIQUE" in mock_model.last_prompt.upper()
        assert "Test text to critique" in mock_model.last_prompt

    def test_critique_string_response(self, prompt_critic, mock_model):
        """Test critique with string response instead of structured response."""
        mock_model.responses["critique"] = (
            "SCORE: 0.65\n"
            "FEEDBACK: The text needs work\n"
            "ISSUES:\n"
            "- Issue one\n"
            "- Issue two\n"
            "SUGGESTIONS:\n"
            "- Suggestion one\n"
            "- Suggestion two"
        )

        result = prompt_critic.critique("Test text")

        assert result["score"] == 0.65
        assert result["feedback"] == "The text needs work"
        assert len(result["issues"]) == 2
        assert "Issue one" in result["issues"]
        assert len(result["suggestions"]) == 2
        assert "Suggestion one" in result["suggestions"]

    def test_critique_empty_text(self, prompt_critic):
        """Test critique with empty text."""
        with pytest.raises(ValueError) as exc_info:
            prompt_critic.critique("")

        assert "text must be a non-empty string" in str(exc_info.value)

    def test_improve(self, prompt_critic, mock_model):
        """Test improve functionality."""
        mock_model.responses["improve"] = {"improved_text": "This is improved text"}

        result = prompt_critic.improve("Original text", "Make it better")

        assert result == "This is improved text"
        assert "IMPROVE" in mock_model.last_prompt.upper()
        assert "Original text" in mock_model.last_prompt
        assert "Make it better" in mock_model.last_prompt

    def test_improve_string_response(self, prompt_critic, mock_model):
        """Test improve with string response."""
        mock_model.responses["improve"] = "IMPROVED_TEXT: Better version of the text"

        result = prompt_critic.improve("Original text", "Make it better")

        assert "Better version of the text" in result

    def test_improve_invalid_response_handling(self, prompt_critic, mock_model):
        """Test improve with invalid response handling."""
        # Test non-dict, non-string response
        mock_model.responses["improve"] = 42  # Invalid response type

        result = prompt_critic.improve("Test text", "Improve this")
        assert "Failed to improve text: Invalid response format" in result

    def test_improve_empty_text(self, prompt_critic):
        """Test improve with empty text."""
        with pytest.raises(ValueError) as exc_info:
            prompt_critic.improve("", "Make it better")

        assert "text must be a non-empty string" in str(exc_info.value)


#
# Error Handling Tests
#
class TestPromptCriticErrorHandling:
    """Tests for error handling in PromptCritic."""

    def test_critique_with_error_handling(self, prompt_critic, mock_model):
        """Test critique with error handling."""

        # Force an exception during model invoke
        def raise_error(*args, **kwargs):
            raise Exception("Test error")

        mock_model.invoke = raise_error

        result = prompt_critic.critique("Test text")

        # Should return a failure result with error information
        assert result["score"] == 0.0
        assert "Failed to critique text" in result["feedback"]
        assert len(result["issues"]) > 0
        assert "Failed to parse model response" in result["issues"]

    def test_improve_with_exception(self, prompt_critic, mock_model):
        """Test improve with exception during model invocation."""

        def raise_error(*args, **kwargs):
            raise Exception("Improvement error")

        mock_model.invoke = raise_error

        with pytest.raises(ValueError) as exc_info:
            prompt_critic.improve("Test text", "Make it better")

        assert "Failed to improve text" in str(exc_info.value)
        assert "Improvement error" in str(exc_info.value)


#
# Async Method Tests
#
class TestPromptCriticAsyncMethods:
    """Tests for async methods in PromptCritic."""

    @pytest.mark.asyncio
    async def test_async_validate(self, prompt_critic, mock_model):
        """Test async validation."""
        # Mock the ainvoke method
        mock_model.ainvoke = MagicMock(return_value={"valid": True, "reason": "Good text"})

        result = await prompt_critic.avalidate("Test text")

        assert result is True
        mock_model.ainvoke.assert_called_once()
        assert "Test text" in mock_model.ainvoke.call_args[0][0]

    @pytest.mark.asyncio
    async def test_async_critique(self, prompt_critic, mock_model):
        """Test async critique."""
        # Mock the ainvoke method
        mock_model.ainvoke = MagicMock(
            return_value={
                "score": 0.8,
                "feedback": "Test feedback",
                "issues": ["Issue 1"],
                "suggestions": ["Suggestion 1"],
            }
        )

        result = await prompt_critic.acritique("Test text")

        assert result["score"] == 0.8
        mock_model.ainvoke.assert_called_once()
        assert "Test text" in mock_model.ainvoke.call_args[0][0]

    @pytest.mark.asyncio
    async def test_async_improve(self, prompt_critic, mock_model):
        """Test async improve."""
        # Mock the ainvoke method
        mock_model.ainvoke = MagicMock(return_value={"improved_text": "Better text"})

        result = await prompt_critic.aimprove("Test text", "Make it better")

        assert result == "Better text"
        mock_model.ainvoke.assert_called_once()
        assert "Test text" in mock_model.ainvoke.call_args[0][0]
        assert "Make it better" in mock_model.ainvoke.call_args[0][0]

    @pytest.mark.asyncio
    async def test_async_validate_error_handling(self, prompt_critic, mock_model):
        """Test async validation with error handling."""

        # Mock the ainvoke method to raise an exception
        async def raise_error(*args, **kwargs):
            raise Exception("Async error")

        mock_model.ainvoke = raise_error

        with pytest.raises(ValueError) as exc_info:
            await prompt_critic.avalidate("Test text")

        assert "Failed to validate text" in str(exc_info.value)
        assert "Async error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_critique_error_handling(self, prompt_critic, mock_model):
        """Test async critique with error handling."""

        # Mock the ainvoke method to raise an exception
        async def raise_error(*args, **kwargs):
            raise Exception("Async critique error")

        mock_model.ainvoke = raise_error

        with pytest.raises(ValueError) as exc_info:
            await prompt_critic.acritique("Test text")

        assert "Failed to critique text" in str(exc_info.value)
        assert "Async critique error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_improve_invalid_response(self, prompt_critic, mock_model):
        """Test async improve with invalid response handling."""

        # Mock the ainvoke method with non-dict, non-string response
        async def return_invalid(*args, **kwargs):
            return 42  # Invalid response type

        mock_model.ainvoke = return_invalid

        result = await prompt_critic.aimprove("Test text", "Improve this")
        assert "Failed to improve text: Invalid response format" in result


#
# Metadata Tests
#
class TestCriticMetadata:
    """Tests for CriticMetadata class."""

    def test_critic_metadata_class(self):
        """Test the CriticMetadata class."""
        metadata = CriticMetadata(
            score=0.8,
            feedback="Test feedback",
            issues=["Issue 1", "Issue 2"],
            suggestions=["Suggestion 1"],
            processing_time_ms=150.5,
        )

        assert metadata.score == 0.8
        assert metadata.feedback == "Test feedback"
        assert len(metadata.issues) == 2
        assert metadata.issues[0] == "Issue 1"
        assert len(metadata.suggestions) == 1
        assert metadata.suggestions[0] == "Suggestion 1"
        assert metadata.processing_time_ms == 150.5


#
# Protocol Adherence Tests
#
class TestProtocolAdherence:
    """Tests for protocol adherence."""

    def test_protocol_adherence(self, prompt_critic):
        """Test that PromptCritic adheres to the protocols."""
        from sifaka.critics.base import TextValidator, TextImprover, TextCritic

        assert isinstance(prompt_critic, TextValidator)
        assert isinstance(prompt_critic, TextImprover)
        assert isinstance(prompt_critic, TextCritic)

    def test_language_model_protocol_adherence(self, mock_model):
        """Test that mock_model adheres to the LanguageModel protocol."""
        assert isinstance(mock_model, LanguageModel)
