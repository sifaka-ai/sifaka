"""Tests for custom exceptions in sifaka.core.exceptions."""

from sifaka.core.exceptions import (
    ConfigurationError,
    CriticError,
    MemoryError,
    ModelProviderError,
    PluginError,
    SifakaError,
    StorageError,
    TimeoutError,
    ValidationError,
    classify_openai_error,
)


class TestSifakaError:
    """Test the base SifakaError class."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        error = SifakaError("Test error message")
        assert error.message == "Test error message"
        assert error.suggestion is None
        assert str(error) == "Test error message"

    def test_initialization_with_suggestion(self):
        """Test initialization with suggestion."""
        error = SifakaError("Test error", suggestion="Try this instead")
        assert error.message == "Test error"
        assert error.suggestion == "Try this instead"
        assert str(error) == "Test error\n💡 Suggestion: Try this instead"

    def test_empty_suggestion(self):
        """Test with empty string suggestion."""
        error = SifakaError("Test error", suggestion="")
        assert str(error) == "Test error"  # Empty suggestion is falsy


class TestConfigurationError:
    """Test the ConfigurationError class."""

    def test_basic_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid config")
        assert error.message == "Invalid config"
        assert error.parameter is None
        assert error.valid_range is None
        assert error.suggestion is None

    def test_with_parameter_only(self):
        """Test with parameter but no range."""
        error = ConfigurationError("Bad value", parameter="max_iterations")
        assert error.parameter == "max_iterations"
        assert error.valid_range is None
        assert error.suggestion == "Check the max_iterations configuration parameter"

    def test_with_parameter_and_range(self):
        """Test with both parameter and valid range."""
        error = ConfigurationError(
            "Value out of range", parameter="temperature", valid_range="0.0-2.0"
        )
        assert error.parameter == "temperature"
        assert error.valid_range == "0.0-2.0"
        assert error.suggestion == "Set temperature to a value within 0.0-2.0"

    def test_string_representation(self):
        """Test string representation with suggestion."""
        error = ConfigurationError(
            "Temperature too high", parameter="temperature", valid_range="0.0-2.0"
        )
        expected = "Temperature too high\n💡 Suggestion: Set temperature to a value within 0.0-2.0"
        assert str(error) == expected


class TestModelProviderError:
    """Test the ModelProviderError class."""

    def test_basic_error(self):
        """Test basic model provider error."""
        error = ModelProviderError("API failed")
        assert error.message == "API failed"
        assert error.provider == "model provider"
        assert error.error_code is None
        assert error.suggestion == "Check your model provider configuration and API key"

    def test_with_provider(self):
        """Test with specific provider."""
        error = ModelProviderError("Connection failed", provider="OpenAI")
        assert error.provider == "OpenAI"
        assert error.suggestion == "Check your OpenAI configuration and API key"

    def test_authentication_error(self):
        """Test authentication error code."""
        error = ModelProviderError(
            "Auth failed", provider="Anthropic", error_code="authentication"
        )
        assert error.error_code == "authentication"
        assert (
            error.suggestion
            == "Check your API key is set correctly in environment variables"
        )

    def test_rate_limit_error(self):
        """Test rate limit error code."""
        error = ModelProviderError("Too many requests", error_code="rate_limit")
        assert (
            error.suggestion
            == "Wait a moment and try again, or check your API usage limits"
        )

    def test_invalid_request_error(self):
        """Test invalid request error code."""
        error = ModelProviderError("Bad request", error_code="invalid_request")
        assert (
            error.suggestion == "Check your request parameters and model configuration"
        )

    def test_insufficient_quota_error(self):
        """Test insufficient quota error code."""
        error = ModelProviderError("No credits", error_code="insufficient_quota")
        assert error.suggestion == "Check your API account billing and usage limits"

    def test_server_error(self):
        """Test server error code."""
        error = ModelProviderError("Internal error", error_code="server_error")
        assert (
            error.suggestion
            == "This is a temporary server issue - try again in a few moments"
        )

    def test_unknown_error_code(self):
        """Test unknown error code falls back to default."""
        error = ModelProviderError(
            "Unknown error", provider="Custom", error_code="unknown_code"
        )
        assert error.suggestion == "Check your Custom configuration and API key"


class TestCriticError:
    """Test the CriticError class."""

    def test_basic_error(self):
        """Test basic critic error."""
        error = CriticError("Critic failed", critic_name="ReflexionCritic")
        assert error.message == "Critic failed"
        assert error.critic_name == "ReflexionCritic"
        assert error.retryable is True
        assert error.suggestion == "Try again with different critics or parameters"

    def test_non_retryable_error(self):
        """Test non-retryable critic error."""
        error = CriticError("Fatal error", critic_name="CustomCritic", retryable=False)
        assert error.retryable is False
        assert error.suggestion == "Manual review recommended"

    def test_string_representation(self):
        """Test string representation."""
        error = CriticError("Analysis failed", critic_name="TestCritic")
        expected = "Analysis failed\n💡 Suggestion: Try again with different critics or parameters"
        assert str(error) == expected


class TestValidationError:
    """Test the ValidationError class."""

    def test_basic_error(self):
        """Test basic validation error."""
        error = ValidationError("Validation failed", validator_name="LengthValidator")
        assert error.message == "Validation failed"
        assert error.validator_name == "LengthValidator"
        assert error.violations == []
        assert (
            error.suggestion
            == "Review the text and address the validation issues listed above"
        )

    def test_with_violations(self):
        """Test with specific violations."""
        violations = ["Text too short", "Missing required section"]
        error = ValidationError(
            "Multiple issues found",
            validator_name="ContentValidator",
            violations=violations,
        )
        assert error.violations == violations

    def test_empty_violations_list(self):
        """Test with explicitly empty violations list."""
        error = ValidationError(
            "Generic failure", validator_name="CustomValidator", violations=[]
        )
        assert error.violations == []


class TestStorageError:
    """Test the StorageError class."""

    def test_save_error(self):
        """Test save operation error."""
        error = StorageError("Cannot save", storage_type="filesystem", operation="save")
        assert error.storage_type == "filesystem"
        assert error.operation == "save"
        assert error.suggestion == "Check storage permissions and available space"

    def test_load_error(self):
        """Test load operation error."""
        error = StorageError("Not found", storage_type="redis", operation="load")
        assert (
            error.suggestion == "Verify the result ID exists and storage is accessible"
        )

    def test_delete_error(self):
        """Test delete operation error."""
        error = StorageError("Cannot delete", storage_type="s3", operation="delete")
        assert (
            error.suggestion == "Confirm the result exists before attempting deletion"
        )

    def test_search_error(self):
        """Test search operation error."""
        error = StorageError(
            "Search failed", storage_type="elasticsearch", operation="search"
        )
        assert error.suggestion == "Check search parameters and storage connectivity"

    def test_unknown_operation(self):
        """Test unknown operation falls back to default."""
        error = StorageError("Unknown op", storage_type="custom", operation="unknown")
        assert error.suggestion == "Check storage configuration and connectivity"


class TestPluginError:
    """Test the PluginError class."""

    def test_basic_error(self):
        """Test basic plugin error."""
        error = PluginError("Plugin failed", plugin_name="redis")
        assert error.message == "Plugin failed"
        assert error.plugin_name == "redis"
        assert error.plugin_type == "storage"
        assert (
            error.suggestion
            == "Ensure redis plugin is properly installed: pip install sifaka-redis"
        )

    def test_with_plugin_type(self):
        """Test with specific plugin type."""
        error = PluginError(
            "Cannot load", plugin_name="custom-critic", plugin_type="critic"
        )
        assert error.plugin_type == "critic"
        assert (
            error.suggestion and "pip install sifaka-custom-critic" in error.suggestion
        )


class TestTimeoutError:
    """Test the TimeoutError class."""

    def test_basic_timeout(self):
        """Test basic timeout error."""
        error = TimeoutError(elapsed_time=15.5, limit=10.0)
        assert error.elapsed_time == 15.5
        assert error.limit == 10.0
        assert error.message == "Operation timeout: 15.5s >= 10.0s"
        assert (
            error.suggestion
            == "Increase timeout_seconds parameter or reduce complexity to complete within 10s"
        )

    def test_precise_timing(self):
        """Test with more precise timing."""
        error = TimeoutError(elapsed_time=30.123, limit=30.0)
        assert "30.1s >= 30.0s" in error.message

    def test_string_representation(self):
        """Test full string representation."""
        error = TimeoutError(elapsed_time=5.5, limit=5.0)
        expected = (
            "Operation timeout: 5.5s >= 5.0s\n"
            "💡 Suggestion: Increase timeout_seconds parameter or reduce complexity to complete within 5s"
        )
        assert str(error) == expected


class TestMemoryError:
    """Test the MemoryError class."""

    def test_basic_error(self):
        """Test basic memory error."""
        error = MemoryError("Memory limit reached", collection_type="critiques")
        assert error.message == "Memory limit reached"
        assert error.collection_type == "critiques"
        assert (
            error.suggestion
            == "Memory bounds reached - older items have been removed to prevent memory issues"
        )

    def test_different_collection_type(self):
        """Test with different collection type."""
        error = MemoryError("Too many items", collection_type="improvements")
        assert error.collection_type == "improvements"
        # Suggestion is the same regardless of collection type


class TestClassifyOpenAIError:
    """Test the classify_openai_error function."""

    def test_authentication_error(self):
        """Test classification of authentication errors."""
        error = classify_openai_error("Authentication failed: Invalid API key")
        assert isinstance(error, ModelProviderError)
        assert error.provider == "OpenAI"
        assert error.error_code == "authentication"
        assert "authentication failed" in error.message.lower()

    def test_api_key_error(self):
        """Test classification of API key errors."""
        error = classify_openai_error("Error: API key not found")
        assert error.error_code == "authentication"

    def test_rate_limit_error(self):
        """Test classification of rate limit errors."""
        error = classify_openai_error("Rate limit exceeded for requests")
        assert error.error_code == "rate_limit"
        assert "rate limit exceeded" in error.message.lower()

    def test_quota_error(self):
        """Test classification of quota errors."""
        error = classify_openai_error("You have exceeded your quota")
        assert error.error_code == "insufficient_quota"
        assert "quota exceeded" in error.message.lower()

    def test_invalid_request_error(self):
        """Test classification of invalid request errors."""
        error = classify_openai_error("Invalid request: model not found")
        assert error.error_code == "invalid_request"
        assert "request invalid" in error.message.lower()

    def test_bad_request_error(self):
        """Test classification of bad request errors."""
        error = classify_openai_error("Bad request: malformed JSON")
        assert error.error_code == "invalid_request"

    def test_server_error_500(self):
        """Test classification of 500 server errors."""
        error = classify_openai_error("Error 500: Internal server error")
        assert error.error_code == "server_error"
        assert "server error" in error.message.lower()

    def test_server_error_text(self):
        """Test classification of server error text."""
        error = classify_openai_error("OpenAI server error occurred")
        assert error.error_code == "server_error"

    def test_unknown_error(self):
        """Test classification of unknown errors."""
        error = classify_openai_error("Some random error occurred")
        assert isinstance(error, ModelProviderError)
        assert error.provider == "OpenAI"
        assert error.error_code is None
        assert error.message == "OpenAI API error: Some random error occurred"

    def test_case_insensitive_matching(self):
        """Test that error matching is case insensitive."""
        error = classify_openai_error("AUTHENTICATION FAILED")
        assert error.error_code == "authentication"

    def test_partial_matching(self):
        """Test partial string matching."""
        error = classify_openai_error(
            "The rate limit has been exceeded for your account"
        )
        assert error.error_code == "rate_limit"

    def test_complex_error_message(self):
        """Test classification with complex error message."""
        complex_error = Exception(
            "HTTPError: 401 Client Error: Unauthorized for url: https://api.openai.com/v1/chat/completions. Invalid authentication credentials."
        )
        error = classify_openai_error(complex_error)
        assert error.error_code == "authentication"

    def test_non_string_error(self):
        """Test with non-string error object."""

        class CustomError:
            def __str__(self):
                return "Rate limit hit"

        error = classify_openai_error(CustomError())
        assert error.error_code == "rate_limit"
