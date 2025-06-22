"""Custom exceptions for Sifaka with specific error types and helpful messages."""

from typing import Optional, List, Any


class SifakaError(Exception):
    """Base exception for all Sifaka errors."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n💡 Suggestion: {self.suggestion}"
        return self.message


class ConfigurationError(SifakaError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        valid_range: Optional[str] = None,
    ):
        self.parameter = parameter
        self.valid_range = valid_range

        suggestion = None
        if parameter and valid_range:
            suggestion = f"Set {parameter} to a value within {valid_range}"
        elif parameter:
            suggestion = f"Check the {parameter} configuration parameter"

        super().__init__(message, suggestion)


class ModelProviderError(SifakaError):
    """Raised when there are issues with model providers (OpenAI, Anthropic, etc.)."""

    def __init__(
        self,
        message: str,
        provider: str = "model provider",
        error_code: Optional[str] = None,
    ):
        self.provider = provider
        self.error_code = error_code

        suggestions = {
            "authentication": "Check your API key is set correctly in environment variables",
            "rate_limit": "Wait a moment and try again, or check your API usage limits",
            "invalid_request": "Check your request parameters and model configuration",
            "insufficient_quota": "Check your API account billing and usage limits",
            "server_error": "This is a temporary server issue - try again in a few moments",
        }

        suggestion = suggestions.get(
            error_code or "", f"Check your {provider} configuration and API key"
        )
        super().__init__(message, suggestion)


class CriticError(SifakaError):
    """Raised when critics fail to evaluate text."""

    def __init__(self, message: str, critic_name: str, retryable: bool = True):
        self.critic_name = critic_name
        self.retryable = retryable

        suggestion = (
            "Try again with different critics or parameters"
            if retryable
            else "Manual review recommended"
        )
        super().__init__(message, suggestion)


class ValidationError(SifakaError):
    """Raised when text validation fails."""

    def __init__(
        self, message: str, validator_name: str, violations: Optional[List[str]] = None
    ):
        self.validator_name = validator_name
        self.violations = violations or []

        suggestion = "Review the text and address the validation issues listed above"
        super().__init__(message, suggestion)


class StorageError(SifakaError):
    """Raised when storage operations fail."""

    def __init__(self, message: str, storage_type: str, operation: str):
        self.storage_type = storage_type
        self.operation = operation

        suggestions = {
            "save": "Check storage permissions and available space",
            "load": "Verify the result ID exists and storage is accessible",
            "delete": "Confirm the result exists before attempting deletion",
            "search": "Check search parameters and storage connectivity",
        }

        suggestion = suggestions.get(
            operation, "Check storage configuration and connectivity"
        )
        super().__init__(message, suggestion)


class PluginError(SifakaError):
    """Raised when plugin loading or execution fails."""

    def __init__(self, message: str, plugin_name: str, plugin_type: str = "storage"):
        self.plugin_name = plugin_name
        self.plugin_type = plugin_type

        suggestion = f"Ensure {plugin_name} plugin is properly installed: pip install sifaka-{plugin_name}"
        super().__init__(message, suggestion)


class TimeoutError(SifakaError):
    """Raised when operations exceed time limits."""

    def __init__(self, elapsed_time: float, limit: float):
        self.elapsed_time = elapsed_time
        self.limit = limit

        message = f"Operation timeout: {elapsed_time:.1f}s >= {limit:.1f}s"
        suggestion = f"Increase timeout_seconds parameter or reduce complexity to complete within {limit:.0f}s"
        super().__init__(message, suggestion)


class MemoryError(SifakaError):
    """Raised when memory limits are approached."""

    def __init__(self, message: str, collection_type: str):
        self.collection_type = collection_type

        suggestion = "Memory bounds reached - older items have been removed to prevent memory issues"
        super().__init__(message, suggestion)


def classify_openai_error(error: Any) -> ModelProviderError:
    """Classify OpenAI API errors into specific types."""
    error_str = str(error).lower()

    if "authentication" in error_str or "api key" in error_str:
        return ModelProviderError(
            f"OpenAI authentication failed: {error}",
            provider="OpenAI",
            error_code="authentication",
        )
    elif "rate limit" in error_str:
        return ModelProviderError(
            f"OpenAI rate limit exceeded: {error}",
            provider="OpenAI",
            error_code="rate_limit",
        )
    elif "quota" in error_str:
        return ModelProviderError(
            f"OpenAI quota exceeded: {error}",
            provider="OpenAI",
            error_code="insufficient_quota",
        )
    elif "invalid request" in error_str or "bad request" in error_str:
        return ModelProviderError(
            f"OpenAI request invalid: {error}",
            provider="OpenAI",
            error_code="invalid_request",
        )
    elif "server error" in error_str or "500" in error_str:
        return ModelProviderError(
            f"OpenAI server error: {error}",
            provider="OpenAI",
            error_code="server_error",
        )
    else:
        return ModelProviderError(f"OpenAI API error: {error}", provider="OpenAI")
