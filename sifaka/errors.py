"""
Error types for Sifaka operations.

This module defines the custom exceptions used throughout the Sifaka framework.
"""

from typing import Dict, Any, Optional, List


class SifakaError(Exception):
    """Base class for all Sifaka exceptions.

    This class provides a standardized structure for Sifaka exceptions,
    including a message, component name, operation name, suggestions for resolution,
    and additional metadata.

    Attributes:
        message (str): Human-readable error message
        component (str, optional): Name of the component that raised the error
        operation (str, optional): Name of the operation that failed
        suggestions (list, optional): List of suggestions for resolving the error
        metadata (dict, optional): Additional error context and details
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.component = component
        self.operation = operation
        self.suggestions = suggestions or []
        self.metadata = metadata or {}

        # Build the full error message
        full_message = message
        if component:
            full_message = f"[{component}] {full_message}"
        if operation:
            full_message = f"{full_message} (during {operation})"
        if suggestions:
            suggestion_text = "; ".join(suggestions)
            full_message = f"{full_message}. Suggestions: {suggestion_text}"

        super().__init__(full_message)


class ConfigurationError(SifakaError):
    """Raised when there is an error in the configuration.

    This error is raised when a component's configuration is invalid,
    such as when required parameters are missing or have invalid values.
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = "configuration",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, component, operation, suggestions, metadata)


class ModelError(SifakaError):
    """Base class for model-related errors.

    This error is raised when there is an issue with a model,
    such as when a model cannot be found or when there is an error
    communicating with a model API.
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = "Model",
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, component, operation, suggestions, metadata)


class ModelNotFoundError(ModelError):
    """Raised when a specified model cannot be found.

    This error is raised when a model specified by name or ID
    cannot be found in the available models.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        component: Optional[str] = "Model",
        operation: Optional[str] = "model lookup",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if model_name:
            message = f"Model '{model_name}' not found: {message}"
            if not suggestions:
                suggestions = ["Check the model name and ensure it is available"]

        metadata = metadata or {}
        if model_name:
            metadata["model_name"] = model_name

        super().__init__(message, component, operation, suggestions, metadata)


class ModelAPIError(ModelError):
    """Raised when there is an error communicating with a model API.

    This error is raised when there is an issue communicating with
    a model API, such as when the API is unavailable or returns an error.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        component: Optional[str] = "Model",
        operation: Optional[str] = "API call",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if model_name:
            message = f"Error communicating with model '{model_name}': {message}"
            if not suggestions:
                suggestions = [
                    "Check your API key and ensure it is valid",
                    "Verify that the model API is available",
                    "Check your network connection",
                ]

        metadata = metadata or {}
        if model_name:
            metadata["model_name"] = model_name

        super().__init__(message, component, operation, suggestions, metadata)


class ValidationError(SifakaError):
    """Raised when validation fails.

    This error is raised when a validator encounters an error
    during validation, not when validation fails normally.
    For normal validation failures, a ValidationResult with
    passed=False should be returned.
    """

    def __init__(
        self,
        message: str,
        validator_name: Optional[str] = None,
        component: Optional[str] = "Validator",
        operation: Optional[str] = "validation",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if validator_name:
            message = f"Validation error in '{validator_name}': {message}"

        metadata = metadata or {}
        if validator_name:
            metadata["validator_name"] = validator_name

        super().__init__(message, component, operation, suggestions, metadata)


class ImproverError(SifakaError):
    """Raised when an improver fails.

    This error is raised when a critic or improver encounters an error
    during improvement, such as when the model fails to generate a response.
    """

    def __init__(
        self,
        message: str,
        improver_name: Optional[str] = None,
        component: Optional[str] = "Improver",
        operation: Optional[str] = "improvement",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if improver_name:
            message = f"Improvement error in '{improver_name}': {message}"

        metadata = metadata or {}
        if improver_name:
            metadata["improver_name"] = improver_name

        super().__init__(message, component, operation, suggestions, metadata)


class ChainError(SifakaError):
    """Raised when there is an error in chain execution.

    This error is raised when a chain encounters an error during execution,
    such as when a component in the chain fails.
    """

    def __init__(
        self,
        message: str,
        chain_name: Optional[str] = None,
        component: Optional[str] = "Chain",
        operation: Optional[str] = "execution",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if chain_name:
            message = f"Chain '{chain_name}' execution error: {message}"

        metadata = metadata or {}
        if chain_name:
            metadata["chain_name"] = chain_name

        super().__init__(message, component, operation, suggestions, metadata)


class CacheError(SifakaError):
    """Raised when there is an error with the cache.

    This error is raised when there is an issue with the cache,
    such as when the cache is corrupted or cannot be accessed.
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = "Cache",
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, component, operation, suggestions, metadata)


class RetryError(SifakaError):
    """Raised when retry attempts are exhausted.

    This error is raised when all retry attempts for an operation
    have been exhausted without success.
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        component: Optional[str] = None,
        operation: Optional[str] = "retry",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if attempts > 0:
            message = f"All {attempts} retry attempts failed: {message}"

        metadata = metadata or {}
        if attempts > 0:
            metadata["attempts"] = attempts

        super().__init__(message, component, operation, suggestions, metadata)


class FallbackError(SifakaError):
    """Raised when all fallback options fail.

    This error is raised when all fallback options for an operation
    have been exhausted without success.
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = "fallback",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, component, operation, suggestions, metadata)


class StreamingError(SifakaError):
    """Raised when there is an error with streaming.

    This error is raised when there is an issue with streaming,
    such as when a streaming connection is interrupted.
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = "streaming",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, component, operation, suggestions, metadata)


class RetrieverError(SifakaError):
    """Raised when there is an error with retrieval operations.

    This error is raised when there is an issue with retrieval,
    such as when a retriever fails to retrieve documents.
    """

    def __init__(
        self,
        message: str,
        retriever_name: Optional[str] = None,
        component: Optional[str] = "Retriever",
        operation: Optional[str] = "retrieval",
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if retriever_name:
            message = f"Retrieval error in '{retriever_name}': {message}"

        metadata = metadata or {}
        if retriever_name:
            metadata["retriever_name"] = retriever_name

        super().__init__(message, component, operation, suggestions, metadata)
