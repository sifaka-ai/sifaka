"""
Error types for Sifaka operations.

This module defines the custom exceptions used throughout the Sifaka framework.
"""


class SifakaError(Exception):
    """Base class for all Sifaka exceptions."""

    pass


class ConfigurationError(SifakaError):
    """Raised when there is an error in the configuration."""

    pass


class ModelError(SifakaError):
    """Base class for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a specified model cannot be found."""

    pass


class ModelAPIError(ModelError):
    """Raised when there is an error communicating with a model API."""

    pass


class ValidationError(SifakaError):
    """Raised when validation fails."""

    pass


class ImproverError(SifakaError):
    """Raised when an improver fails."""

    pass


class ChainError(SifakaError):
    """Raised when there is an error in chain execution."""

    pass


class CacheError(SifakaError):
    """Raised when there is an error with the cache."""

    pass


class RetryError(SifakaError):
    """Raised when retry attempts are exhausted."""

    pass


class FallbackError(SifakaError):
    """Raised when all fallback options fail."""

    pass


class StreamingError(SifakaError):
    """Raised when there is an error with streaming."""

    pass


class RetrieverError(SifakaError):
    """Raised when there is an error with retrieval operations."""

    pass
