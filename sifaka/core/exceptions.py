"""Comprehensive exception hierarchy for Sifaka error handling.

This module defines all custom exceptions used throughout Sifaka, providing:
- Clear error categorization by type
- Helpful suggestions for resolution
- Structured error information for debugging
- Consistent error messaging across the framework

## Exception Hierarchy:

    SifakaError (base)
    ├── ConfigurationError - Invalid configuration parameters
    ├── ModelProviderError - LLM API failures
    ├── CriticError - Critic evaluation failures
    ├── ValidationError - Text validation failures
    ├── StorageError - Storage backend issues
    ├── PluginError - Plugin loading/execution failures
    ├── TimeoutError - Operation time limit exceeded
    ├── MemoryError - Memory bounds reached
    └── EmbeddingError - Embedding generation failures

## Design Principles:

1. **User-Friendly**: Every exception includes helpful suggestions
2. **Structured**: Exceptions carry relevant context for debugging
3. **Actionable**: Clear guidance on how to resolve the issue
4. **Consistent**: All exceptions follow the same pattern

## Usage:

    >>> from sifaka.core.exceptions import ConfigurationError
    >>>
    >>> # Raise with helpful context
    >>> if temperature > 2.0:
    ...     raise ConfigurationError(
    ...         "Temperature too high",
    ...         parameter="temperature",
    ...         valid_range="0.0-2.0"
    ...     )
    >>>
    >>> # Exception includes suggestion
    >>> # ConfigurationError: Temperature too high
    >>> # 💡 Suggestion: Set temperature to a value within 0.0-2.0

## Error Classification:

The module also provides utilities for classifying external errors
(like OpenAI API errors) into appropriate Sifaka exceptions.
"""

from typing import Any, List, Optional


class SifakaError(Exception):
    """Base exception class for all Sifaka-specific errors.

    All Sifaka exceptions inherit from this base class, providing
    consistent error formatting and optional suggestions for resolution.
    The suggestion feature helps users quickly understand how to fix issues.

    Example:
        >>> raise SifakaError(
        ...     "Something went wrong",
        ...     suggestion="Try adjusting the parameters"
        ... )
        >>> # Output:
        >>> # Something went wrong
        >>> # 💡 Suggestion: Try adjusting the parameters

    Attributes:
        message: The primary error message
        suggestion: Optional helpful suggestion for resolution
    """

    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n💡 Suggestion: {self.suggestion}"
        return self.message


class ConfigurationError(SifakaError):
    """Exception for invalid configuration parameters.

    Raised when configuration values are out of range, incompatible,
    or otherwise invalid. Provides specific parameter information and
    valid ranges to help users correct the issue.

    Example:
        >>> # Temperature out of range
        >>> raise ConfigurationError(
        ...     "Temperature 3.5 exceeds maximum",
        ...     parameter="temperature",
        ...     valid_range="0.0-2.0"
        ... )
        >>>
        >>> # Missing required configuration
        >>> raise ConfigurationError(
        ...     "Model not specified",
        ...     parameter="model"
        ... )

    Common scenarios:
    - Parameter values outside valid ranges
    - Missing required configuration
    - Incompatible parameter combinations
    - Invalid model or critic names
    """

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
    """Exception for LLM provider API failures.

    Handles various failure modes when communicating with LLM providers
    like OpenAI, Anthropic, or Gemini. Includes specific error codes and
    targeted suggestions based on the failure type.

    Example:
        >>> # Authentication failure
        >>> raise ModelProviderError(
        ...     "Invalid API key",
        ...     provider="OpenAI",
        ...     error_code="authentication"
        ... )
        >>> # Suggestion: Check your API key is set correctly in environment variables
        >>>
        >>> # Rate limit exceeded
        >>> raise ModelProviderError(
        ...     "Too many requests",
        ...     provider="Anthropic",
        ...     error_code="rate_limit"
        ... )
        >>> # Suggestion: Wait a moment and try again, or check your API usage limits

    Error codes and their meanings:
    - authentication: API key invalid or missing
    - rate_limit: Request rate exceeded
    - invalid_request: Malformed request parameters
    - insufficient_quota: Account limits reached
    - server_error: Provider-side temporary failure
    - no_provider: No LLM provider configured
    """

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
            "no_provider": "Set up at least one API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, or OLLAMA_API_KEY",
        }

        suggestion = suggestions.get(
            error_code or "", f"Check your {provider} configuration and API key"
        )
        super().__init__(message, suggestion)


class CriticError(SifakaError):
    """Exception for critic evaluation failures.

    Raised when a critic fails to analyze text or provide feedback.
    Includes information about which critic failed and whether the
    error is likely transient (retryable) or permanent.

    Example:
        >>> # Transient failure (retryable)
        >>> raise CriticError(
        ...     "Timeout analyzing text",
        ...     critic_name="reflexion",
        ...     retryable=True
        ... )
        >>>
        >>> # Permanent failure
        >>> raise CriticError(
        ...     "Critic configuration invalid",
        ...     critic_name="constitutional",
        ...     retryable=False
        ... )

    The retryable flag helps the engine decide whether to:
    - Retry with the same critic (if True)
    - Skip the critic and continue (if False)
    """

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
    """Exception for text validation failures.

    Raised when text fails to meet validation criteria. Includes
    the validator name and specific violations to help users
    understand what needs to be fixed.

    Example:
        >>> # Length validation failure
        >>> raise ValidationError(
        ...     "Text too short",
        ...     validator_name="length",
        ...     violations=["Minimum 100 characters required", "Current: 45"]
        ... )
        >>>
        >>> # Content validation failure
        >>> raise ValidationError(
        ...     "Required terms missing",
        ...     validator_name="content",
        ...     violations=["Missing: 'AI'", "Missing: 'benefits'"]
        ... )

    Note:
        This exception is typically not raised during normal operation.
        Failed validations are recorded as ValidationResult objects.
        This exception is for catastrophic validation failures.
    """

    def __init__(
        self, message: str, validator_name: str, violations: Optional[List[str]] = None
    ):
        self.validator_name = validator_name
        self.violations = violations or []

        suggestion = "Review the text and address the validation issues listed above"
        super().__init__(message, suggestion)


class StorageError(SifakaError):
    """Exception for storage backend operation failures.

    Covers all storage-related errors including save, load, delete,
    and search operations. Provides operation-specific suggestions
    for resolution.

    Example:
        >>> # Save failure
        >>> raise StorageError(
        ...     "Permission denied",
        ...     storage_type="file",
        ...     operation="save"
        ... )
        >>> # Suggestion: Check storage permissions and available space
        >>>
        >>> # Load failure
        >>> raise StorageError(
        ...     "Result not found",
        ...     storage_type="redis",
        ...     operation="load"
        ... )
        >>> # Suggestion: Verify the result ID exists and storage is accessible

    Operations and common failures:
    - save: Permission issues, space limitations
    - load: Missing data, connectivity issues
    - delete: Permission issues, non-existent data
    - search: Invalid queries, backend failures
    """

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
    """Exception for plugin system failures.

    Raised when plugins fail to load, register, or execute. Includes
    guidance on installing missing plugins.

    Example:
        >>> # Missing plugin
        >>> raise PluginError(
        ...     "Redis storage plugin not found",
        ...     plugin_name="redis",
        ...     plugin_type="storage"
        ... )
        >>> # Suggestion: Ensure redis plugin is properly installed: pip install sifaka-redis
        >>>
        >>> # Plugin initialization failure
        >>> raise PluginError(
        ...     "Failed to initialize PostgreSQL connection",
        ...     plugin_name="postgres",
        ...     plugin_type="storage"
        ... )

    Plugin types:
    - storage: Storage backend plugins
    - critic: Custom critic plugins
    - validator: Custom validator plugins
    """

    def __init__(self, message: str, plugin_name: str, plugin_type: str = "storage"):
        self.plugin_name = plugin_name
        self.plugin_type = plugin_type

        suggestion = f"Ensure {plugin_name} plugin is properly installed: pip install sifaka-{plugin_name}"
        super().__init__(message, suggestion)


class TimeoutError(SifakaError):
    """Exception for operations that exceed configured time limits.

    Provides specific timing information to help users adjust timeouts
    or optimize their operations.

    Example:
        >>> # Operation timeout
        >>> raise TimeoutError(
        ...     elapsed_time=305.2,
        ...     limit=300.0
        ... )
        >>> # Operation timeout: 305.2s >= 300.0s
        >>> # 💡 Suggestion: Increase timeout_seconds parameter or reduce complexity to complete within 300s

    Common causes:
    - Complex text requiring many iterations
    - Slow LLM API responses
    - Validators or critics taking too long
    - Network latency issues
    """

    def __init__(self, elapsed_time: float, limit: float):
        self.elapsed_time = elapsed_time
        self.limit = limit

        message = f"Operation timeout: {elapsed_time:.1f}s >= {limit:.1f}s"
        suggestion = f"Increase timeout_seconds parameter or reduce complexity to complete within {limit:.0f}s"
        super().__init__(message, suggestion)


class MemoryError(SifakaError):
    """Exception for memory bound violations.

    Raised when collections (generations, critiques, validations)
    approach memory limits. This is more of a warning than an error,
    as the system automatically prunes old entries.

    Example:
        >>> # Memory bounds reached
        >>> raise MemoryError(
        ...     "Critique history at capacity",
        ...     collection_type="critiques"
        ... )
        >>> # Suggestion: Memory bounds reached - older items have been removed to prevent memory issues

    Collection types:
    - generations: Text generation history
    - critiques: Critic feedback history
    - validations: Validation result history

    Note:
        This exception is rarely seen in practice as the system
        handles memory management automatically.
    """

    def __init__(self, message: str, collection_type: str):
        self.collection_type = collection_type

        suggestion = "Memory bounds reached - older items have been removed to prevent memory issues"
        super().__init__(message, suggestion)


class EmbeddingError(SifakaError):
    """Exception for embedding generation failures.

    Covers errors in generating text embeddings for semantic search
    and RAG functionality.

    Example:
        >>> raise EmbeddingError(
        ...     "API key not found",
        ...     provider="openai"
        ... )
    """

    def __init__(
        self, message: str, provider: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize embedding error with provider context."""
        if provider:
            message = f"[{provider}] {message}"

        suggestions = []
        if "API key" in message:
            suggestions.append("Set the appropriate API key environment variable")
        if "dimension" in message.lower():
            suggestions.append("Check model supports requested dimensions")
        if not suggestions:
            suggestions.append("Check provider documentation for embedding models")

        suggestion = " | ".join(suggestions)
        super().__init__(message, suggestion, **kwargs)


def classify_openai_error(error: Any) -> ModelProviderError:
    """Convert OpenAI API errors into appropriate ModelProviderError types.

    Analyzes OpenAI error messages to determine the specific failure type
    and creates a ModelProviderError with appropriate error code and
    suggestion.

    Args:
        error: The original error from OpenAI API (any type)

    Returns:
        ModelProviderError with specific error_code and helpful suggestion

    Example:
        >>> try:
        ...     response = await openai_client.complete(...)
        >>> except Exception as e:
        ...     raise classify_openai_error(e)

    Error classifications:
    - "authentication" or "api key" → authentication error
    - "rate limit" → rate limit error
    - "quota" → insufficient quota error
    - "invalid request" → invalid request error
    - "server error" or "500" → server error
    - Others → generic OpenAI error
    """
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
