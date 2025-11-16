"""Extended tests for exception classes."""

from sifaka.core.exceptions import (
    ConfigurationError,
    CriticError,
    MemoryError as SifakaMemoryError,
    ModelProviderError,
    PluginError,
    SifakaError,
    StorageError,
    TimeoutError,
    ValidationError,
    classify_openai_error,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_all_inherit_from_base(self):
        """Test all exceptions inherit from SifakaError."""
        exceptions = [
            ConfigurationError("test"),
            ValidationError("test", "validator"),
            ModelProviderError("test"),
            CriticError("test", "critic"),
            StorageError("test", "storage", "save"),
            PluginError("test", "plugin", "critic"),
            TimeoutError(1.0, 2.0),
            SifakaMemoryError("test", "generations"),
        ]

        for exc in exceptions:
            assert isinstance(exc, SifakaError)
            assert isinstance(exc, Exception)


class TestSifakaError:
    """Test base SifakaError."""

    def test_with_details(self):
        """Test error with suggestion."""
        error = SifakaError("Main error", "Try this fix")
        assert error.message == "Main error"
        assert error.suggestion == "Try this fix"

        # Test without suggestion
        error2 = SifakaError("Another error")
        assert error2.message == "Another error"
        assert error2.suggestion is None

    def test_string_formatting(self):
        """Test error string formatting."""
        # Just message
        error = SifakaError("Simple error")
        assert str(error) == "Simple error"

        # With suggestion
        error = SifakaError("Error occurred", "Fix it this way")
        error_str = str(error)
        assert "Error occurred" in error_str
        assert "Fix it this way" in error_str


class TestConfigurationError:
    """Test ConfigurationError specifics."""

    def test_with_all_fields(self):
        """Test with all configuration fields."""
        error = ConfigurationError(
            "Bad config", parameter="timeout", valid_range="0-300"
        )

        assert error.parameter == "timeout"
        assert error.valid_range == "0-300"
        assert "timeout" in str(error)

    def test_auto_suggestion(self):
        """Test automatic suggestion generation."""
        error = ConfigurationError(
            "Invalid value", parameter="temperature", valid_range="0.0-2.0"
        )

        # Should have auto-generated suggestion
        assert error.suggestion is not None
        assert "temperature" in error.suggestion
        assert "0.0-2.0" in error.suggestion


class TestValidationError:
    """Test ValidationError specifics."""

    def test_with_violations_list(self):
        """Test with list of violations."""
        violations = ["Too short", "Missing keywords", "Poor grammar"]
        error = ValidationError("Text failed validation", "quality_check", violations)

        assert error.validator_name == "quality_check"
        assert error.violations == violations
        assert len(error.violations) == 3

    def test_string_representation(self):
        """Test string representation includes violations."""
        error = ValidationError("Failed", "grammar", ["Error 1", "Error 2"])

        error_str = str(error)
        assert "Failed" in error_str
        assert error.violations == ["Error 1", "Error 2"]
        assert error.validator_name == "grammar"


class TestModelProviderError:
    """Test ModelProviderError specifics."""

    def test_with_retry_info(self):
        """Test with retry information."""
        error = ModelProviderError(
            "Rate limited", provider="OpenAI", error_code="rate_limit"
        )

        assert error.provider == "OpenAI"
        assert error.error_code == "rate_limit"

    def test_auto_suggestion_by_code(self):
        """Test suggestions based on error code."""
        # Authentication error
        error = ModelProviderError("Invalid key", error_code="authentication")
        assert "API key" in error.suggestion

        # Rate limit error
        error = ModelProviderError("Too many requests", error_code="rate_limit")
        assert "wait" in error.suggestion.lower() or "retry" in error.suggestion.lower()

        # Quota error
        error = ModelProviderError("Quota exceeded", error_code="insufficient_quota")
        assert (
            "quota" in error.suggestion.lower() or "billing" in error.suggestion.lower()
        )


class TestTimeoutError:
    """Test TimeoutError specifics."""

    def test_time_formatting(self):
        """Test time value formatting."""
        error = TimeoutError(65.5, 60.0)
        error_str = str(error)

        assert "65.5s" in error_str
        assert "60.0s" in error_str
        assert error.elapsed_time == 65.5
        assert error.limit == 60.0

    def test_with_operation(self):
        """Test with operation context."""
        error = TimeoutError(30.0, 25.0)
        assert error.elapsed_time == 30.0
        assert error.limit == 25.0
        assert "30.0s" in str(error)
        assert "25.0s" in str(error)


class TestStorageError:
    """Test StorageError specifics."""

    def test_operation_suggestions(self):
        """Test operation-specific suggestions."""
        # Save error
        error = StorageError("Failed", "file", "save")
        assert "permissions" in error.suggestion or "disk" in error.suggestion

        # Load error
        error = StorageError("Failed", "redis", "load")
        assert "connection" in error.suggestion or "exists" in error.suggestion

    def test_with_details(self):
        """Test with storage type and operation."""
        error = StorageError("Connection failed", "redis", "connect")
        assert error.storage_type == "redis"
        assert error.operation == "connect"


class TestOpenAIErrorClassification:
    """Test OpenAI error classification."""

    def test_with_openai_exceptions(self):
        """Test with actual OpenAI exception types."""

        # Mock OpenAI-like exceptions
        class MockAuthError(Exception):
            def __init__(self):
                super().__init__("Incorrect API key provided")

        class MockRateLimitError(Exception):
            def __init__(self):
                super().__init__("Rate limit exceeded")

        # Test classification
        auth_error = MockAuthError()
        classified = classify_openai_error(auth_error)
        assert isinstance(classified, ModelProviderError)
        assert classified.error_code == "authentication"

        rate_error = MockRateLimitError()
        classified = classify_openai_error(rate_error)
        assert isinstance(classified, ModelProviderError)
        assert classified.error_code == "rate_limit"

    def test_error_message_patterns(self):
        """Test various error message patterns."""
        patterns = [
            ("Invalid API key", "authentication"),
            ("Incorrect API key provided", "authentication"),
            ("Rate limit exceeded", "rate_limit"),
            ("You exceeded your current quota", "insufficient_quota"),
            ("Server error occurred", "server_error"),
            ("Internal server error", "server_error"),
            ("500 Internal Server Error", "server_error"),
        ]

        for message, expected_code in patterns:
            error = Exception(message)
            classified = classify_openai_error(error)
            assert classified.error_code == expected_code

    def test_unknown_error(self):
        """Test handling of unknown errors."""
        error = Exception("Some random error")
        classified = classify_openai_error(error)

        assert isinstance(classified, ModelProviderError)
        assert classified.provider == "OpenAI"
        assert classified.error_code is None
        assert "Some random error" in str(classified)
