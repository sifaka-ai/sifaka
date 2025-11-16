"""Simple tests for exception classes to improve coverage."""

from sifaka.core.exceptions import (
    ConfigurationError,
    CriticError,
    ModelProviderError,
    PluginError,
    SifakaError,
    StorageError,
    TimeoutError,
    ValidationError,
    classify_openai_error,
)


class TestExceptions:
    """Test exception classes."""

    def test_sifaka_error(self):
        """Test base SifakaError."""
        error = SifakaError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.suggestion is None

        error = SifakaError("Error", "Fix suggestion")
        assert "Error" in str(error)
        assert "Fix suggestion" in str(error)

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Bad config")
        assert isinstance(error, SifakaError)
        assert "Bad config" in str(error)

        error = ConfigurationError("Invalid", parameter="temp", valid_range="0-2")
        assert error.parameter == "temp"
        assert error.valid_range == "0-2"

    def test_model_provider_error(self):
        """Test ModelProviderError."""
        error = ModelProviderError("API failed")
        assert isinstance(error, SifakaError)

        error = ModelProviderError(
            "Rate limit", provider="OpenAI", error_code="rate_limit"
        )
        assert error.provider == "OpenAI"
        assert error.error_code == "rate_limit"

    def test_critic_error(self):
        """Test CriticError."""
        error = CriticError("Critic failed", "reflexion")
        assert error.critic_name == "reflexion"
        assert isinstance(error, SifakaError)

        error = CriticError("Retry", "style", retryable=True)
        assert error.retryable is True

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Failed", "length_check")
        assert error.validator_name == "length_check"

        error = ValidationError("Failed", "grammar", ["Error 1", "Error 2"])
        assert len(error.violations) == 2

    def test_storage_error(self):
        """Test StorageError."""
        error = StorageError("Save failed", "file", "save")
        assert error.storage_type == "file"
        assert error.operation == "save"

    def test_plugin_error(self):
        """Test PluginError."""
        error = PluginError("Not found", "custom", "critic")
        assert error.plugin_name == "custom"
        assert error.plugin_type == "critic"

    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError(10.5, 10.0)
        assert error.elapsed_time == 10.5
        assert error.limit == 10.0
        assert "10.5s" in str(error)
        assert "10.0s" in str(error)

    def test_memory_error(self):
        """Test MemoryError."""
        # Import locally to avoid shadowing Python's built-in
        from sifaka.core.exceptions import MemoryError as SifakaMemoryError

        error = SifakaMemoryError("Out of memory", "generations")
        assert isinstance(error, SifakaError)
        assert error.collection_type == "generations"

        error = SifakaMemoryError("Collection full", "critiques")
        assert error.collection_type == "critiques"

    def test_classify_openai_error(self):
        """Test OpenAI error classification."""
        # Test various error patterns
        patterns = [
            (Exception("Invalid API key"), "authentication"),
            (Exception("Rate limit exceeded"), "rate_limit"),
            (Exception("Invalid request"), "invalid_request"),
            (Exception("You exceeded your current quota"), "insufficient_quota"),
            (Exception("Internal server error"), "server_error"),
            (Exception("Random error"), None),
        ]

        for original_error, expected_code in patterns:
            classified = classify_openai_error(original_error)
            assert isinstance(classified, ModelProviderError)
            assert classified.provider == "OpenAI"
            assert classified.error_code == expected_code
