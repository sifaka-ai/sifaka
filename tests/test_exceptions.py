"""Tests for Sifaka exception handling."""

from sifaka.core.exceptions import (
    SifakaError,
    ConfigurationError,
    ModelProviderError,
    CriticError,
    ValidationError,
    StorageError,
    PluginError,
    TimeoutError,
    MemoryError,
    classify_openai_error,
)


class TestSifakaExceptions:
    """Test custom exception classes."""

    def test_sifaka_error_basic(self):
        """Test basic SifakaError functionality."""
        error = SifakaError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.suggestion is None

    def test_sifaka_error_with_suggestion(self):
        """Test SifakaError with suggestion."""
        error = SifakaError("Test error", "Try this fix")
        assert "Test error" in str(error)
        assert "Try this fix" in str(error)
        assert error.suggestion == "Try this fix"

    def test_configuration_error(self):
        """Test ConfigurationError with parameter info."""
        error = ConfigurationError(
            "Invalid temperature", parameter="temperature", valid_range="0.0-2.0"
        )
        assert error.parameter == "temperature"
        assert error.valid_range == "0.0-2.0"
        assert "temperature" in error.suggestion

    def test_model_provider_error(self):
        """Test ModelProviderError with provider and error code."""
        error = ModelProviderError(
            "API key invalid", provider="OpenAI", error_code="authentication"
        )
        assert error.provider == "OpenAI"
        assert error.error_code == "authentication"
        assert "API key" in error.suggestion

    def test_critic_error(self):
        """Test CriticError with retryable flag."""
        error = CriticError("Critic failed", "reflexion", retryable=True)
        assert error.critic_name == "reflexion"
        assert error.retryable is True
        assert "Try again" in error.suggestion

    def test_validation_error(self):
        """Test ValidationError with violations."""
        violations = ["Too short", "Missing keywords"]
        error = ValidationError("Validation failed", "length", violations)
        assert error.validator_name == "length"
        assert error.violations == violations

    def test_storage_error(self):
        """Test StorageError with operation context."""
        error = StorageError("Save failed", "file", "save")
        assert error.storage_type == "file"
        assert error.operation == "save"
        assert "permissions" in error.suggestion

    def test_plugin_error(self):
        """Test PluginError with plugin details."""
        error = PluginError("Plugin not found", "redis", "storage")
        assert error.plugin_name == "redis"
        assert error.plugin_type == "storage"
        assert "pip install" in error.suggestion

    def test_timeout_error(self):
        """Test TimeoutError with timing info."""
        error = TimeoutError(125.5, 120.0)
        assert error.elapsed_time == 125.5
        assert error.limit == 120.0
        assert "125.5s" in str(error)

    def test_memory_error(self):
        """Test MemoryError with collection info."""
        error = MemoryError("Memory limit reached", "generations")
        assert error.collection_type == "generations"
        assert "bounds" in error.suggestion


class TestOpenAIErrorClassification:
    """Test OpenAI error classification."""

    def test_authentication_error(self):
        """Test authentication error classification."""
        mock_error = Exception("Invalid API key provided")
        classified = classify_openai_error(mock_error)

        assert isinstance(classified, ModelProviderError)
        assert classified.provider == "OpenAI"
        assert classified.error_code == "authentication"

    def test_rate_limit_error(self):
        """Test rate limit error classification."""
        mock_error = Exception("Rate limit exceeded")
        classified = classify_openai_error(mock_error)

        assert isinstance(classified, ModelProviderError)
        assert classified.error_code == "rate_limit"

    def test_invalid_request_error(self):
        """Test invalid request error classification."""
        mock_error = Exception("Invalid request format")
        classified = classify_openai_error(mock_error)

        assert isinstance(classified, ModelProviderError)
        assert classified.error_code == "invalid_request"

    def test_quota_error(self):
        """Test quota error classification."""
        mock_error = Exception(
            "You exceeded your current quota, please check your plan"
        )
        classified = classify_openai_error(mock_error)

        assert isinstance(classified, ModelProviderError)
        assert classified.error_code == "insufficient_quota"

    def test_server_error(self):
        """Test server error classification."""
        mock_error = Exception("Internal server error 500")
        classified = classify_openai_error(mock_error)

        assert isinstance(classified, ModelProviderError)
        assert classified.error_code == "server_error"

    def test_generic_error(self):
        """Test generic error classification."""
        mock_error = Exception("Unknown error occurred")
        classified = classify_openai_error(mock_error)

        assert isinstance(classified, ModelProviderError)
        assert classified.provider == "OpenAI"
        assert classified.error_code is None
