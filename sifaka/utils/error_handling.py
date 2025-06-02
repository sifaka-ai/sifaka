"""
Error handling utilities for Sifaka.

This module provides essential error handling utilities including:
- Custom exception classes with rich context
- Error context managers for consistent error handling
- Error logging utilities
- Error formatting and suggestion generation

The error handling system is designed to provide actionable feedback
to users and developers, with consistent formatting and logging.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class SifakaError(Exception):
    """Base class for all Sifaka errors."""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the error.

        Args:
            message: The error message
            component: Optional component name
            operation: Optional operation name
            suggestions: Optional list of suggestions
            metadata: Optional metadata to include
        """
        self.message = message
        self.component = component
        self.operation = operation
        self.suggestions = suggestions or []
        self.metadata = metadata or {}

        # Format the error message
        formatted_message = self._format_message()
        super().__init__(formatted_message)

    def _format_message(self) -> str:
        """Format the error message with component, operation, and suggestions.

        Returns:
            A formatted error message
        """
        formatted_message = self.message

        if self.component:
            formatted_message = f"[{self.component}] {formatted_message}"

        if self.operation:
            formatted_message = f"{formatted_message} (during {self.operation})"

        if self.suggestions:
            suggestion_text = "; ".join(self.suggestions)
            formatted_message = f"{formatted_message}. Suggestions: {suggestion_text}"

        return formatted_message


class ModelError(SifakaError):
    """Error raised by model components."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the error.

        Args:
            message: The error message
            model_name: The name of the model
            component: Optional component name
            operation: Optional operation name
            suggestions: Optional list of suggestions
            metadata: Optional metadata to include
        """
        # Add model_name to metadata
        metadata = metadata or {}
        if model_name:
            metadata["model_name"] = model_name

        super().__init__(
            message=message,
            component=component,
            operation=operation,
            suggestions=suggestions,
            metadata=metadata,
        )

        # Store as attribute for backward compatibility
        self.model_name = model_name


class ModelAPIError(ModelError):
    """Error raised by model API calls."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the error.

        Args:
            message: The error message
            model_name: The name of the model
            component: Optional component name
            operation: Optional operation name
            suggestions: Optional list of suggestions
            metadata: Optional metadata to include
        """
        # Add model_name to metadata
        metadata = metadata or {}
        if model_name:
            metadata["model_name"] = model_name

        super().__init__(
            message=message,
            component=component,
            operation=operation,
            suggestions=suggestions,
            metadata=metadata,
        )


class ValidationError(SifakaError):
    """Error raised by validator components."""

    def __init__(
        self,
        message: str,
        validator_name: Optional[str] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the error.

        Args:
            message: The error message
            validator_name: The name of the validator
            component: Optional component name
            operation: Optional operation name
            suggestions: Optional list of suggestions
            metadata: Optional metadata to include
        """
        # Add validator_name to metadata
        metadata = metadata or {}
        if validator_name:
            metadata["validator_name"] = validator_name

        super().__init__(
            message=message,
            component=component,
            operation=operation,
            suggestions=suggestions,
            metadata=metadata,
        )

        # Store as attribute for backward compatibility
        self.validator_name = validator_name


class ImproverError(SifakaError):
    """Error raised by improver components."""


class RetrieverError(SifakaError):
    """Error raised by retriever components."""


class ChainError(SifakaError):
    """Error raised by chain components."""


class ConfigurationError(SifakaError):
    """Error raised for configuration issues."""


class StorageError(SifakaError):
    """Error raised by storage components."""

    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the error.

        Args:
            message: The error message
            storage_type: The type of storage
            component: Optional component name
            operation: Optional operation name
            suggestions: Optional list of suggestions
            metadata: Optional metadata to include
        """
        # Add storage_type to metadata
        metadata = metadata or {}
        if storage_type:
            metadata["storage_type"] = storage_type

        super().__init__(
            message=message,
            component=component,
            operation=operation,
            suggestions=suggestions,
            metadata=metadata,
        )

        # Store as attribute for backward compatibility
        self.storage_type = storage_type


def format_error_message(
    message: str,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
) -> str:
    """
    Format an error message with component, operation, and suggestions.

    Args:
        message: The base error message
        component: Optional component name
        operation: Optional operation name
        suggestions: Optional list of suggestions

    Returns:
        A formatted error message
    """
    formatted_message = message

    if component:
        formatted_message = f"[{component}] {formatted_message}"

    if operation:
        formatted_message = f"{formatted_message} (during {operation})"

    if suggestions:
        suggestion_text = "; ".join(suggestions)
        formatted_message = f"{formatted_message}. Suggestions: {suggestion_text}"

    return formatted_message


def log_error(
    error: Exception,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    include_traceback: bool = True,
    component: Optional[str] = None,
    operation: Optional[str] = None,
) -> None:
    """
    Log an error with consistent formatting.

    Args:
        error: The exception to log
        logger_instance: Optional logger instance to use
        level: Logging level
        include_traceback: Whether to include the traceback
        component: Optional component name
        operation: Optional operation name
    """
    logger_to_use = logger_instance or logging.getLogger(__name__)

    # Format the error message
    error_message = str(error)
    if component or operation:
        error_message = format_error_message(error_message, component, operation)

    # Log the error
    if include_traceback:
        logger_to_use.log(level, error_message, exc_info=True)
    else:
        logger_to_use.log(level, error_message)


class ErrorContext:
    """A context object for error handling."""

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the context object.

        Args:
            metadata: Optional metadata to include
        """
        self.metadata = metadata or {}


@contextmanager
def error_context(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    error_class: Type[Exception] = SifakaError,
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    logger_instance: Optional[logging.Logger] = None,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
) -> Any:
    """
    Context manager for consistent error handling.

    Args:
        component: Optional component name
        operation: Optional operation name
        error_class: Exception class to raise
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include
        logger_instance: Optional logger instance to use
        log_level: Logging level
        include_traceback: Whether to include the traceback

    Yields:
        ErrorContext: A context object that can be used to store additional metadata

    Raises:
        The specified error_class with additional context
    """
    # Create a context object with the metadata
    context = ErrorContext(metadata)

    try:
        yield context
    except Exception as e:
        # Log the error
        log_error(
            e,
            logger_instance=logger_instance,
            level=log_level,
            include_traceback=include_traceback,
            component=component,
            operation=operation,
        )

        # Create the new message
        original_message = str(e)
        new_message = original_message
        if message_prefix:
            new_message = f"{message_prefix}: {new_message}"

        # Create the new exception
        if issubclass(error_class, SifakaError):
            # If the target is a SifakaError, use its constructor with all parameters
            # Handle specialized error classes with their specific parameters
            if error_class == ModelError:
                new_exception: Exception = error_class(
                    message=new_message,
                    model_name=context.metadata.get("model_name"),
                    component=component,
                    operation=operation,
                    suggestions=suggestions,
                    metadata=context.metadata,
                )
            elif error_class == ValidationError:
                new_exception = error_class(
                    message=new_message,
                    validator_name=context.metadata.get("validator_name"),
                    component=component,
                    operation=operation,
                    suggestions=suggestions,
                    metadata=context.metadata,
                )
            elif error_class == StorageError:
                new_exception = error_class(
                    message=new_message,
                    storage_type=context.metadata.get("storage_type"),
                    component=component,
                    operation=operation,
                    suggestions=suggestions,
                    metadata=context.metadata,
                )
            else:
                # Generic SifakaError or other subclasses
                new_exception = error_class(
                    message=new_message,
                    component=component,
                    operation=operation,
                    suggestions=suggestions,
                    metadata=context.metadata,
                )
        else:
            # Otherwise, use the standard constructor
            new_exception = error_class(new_message)

        # Raise the new exception
        raise new_exception from e


# Specialized context managers for common operations


@contextmanager
def model_context(
    model_name: Optional[str] = None,
    operation: Optional[str] = "generation",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Context manager for model operations.

    Args:
        model_name: Optional model name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include

    Yields:
        None

    Raises:
        ModelError with additional context
    """
    with error_context(
        component="Model",
        operation=operation,
        error_class=ModelError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add model name to the context if provided
        if model_name:
            context.metadata["model_name"] = model_name
        yield


@contextmanager
def validation_context(
    validator_name: Optional[str] = None,
    operation: Optional[str] = "validation",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Context manager for validation operations.

    Args:
        validator_name: Optional validator name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include

    Yields:
        None

    Raises:
        ValidationError with additional context
    """
    with error_context(
        component="Validator",
        operation=operation,
        error_class=ValidationError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add validator name to the context if provided
        if validator_name:
            context.metadata["validator_name"] = validator_name
        yield


@contextmanager
def critic_context(
    critic_name: Optional[str] = None,
    operation: Optional[str] = "critique",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Context manager for critic operations.

    Args:
        critic_name: Optional critic name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include

    Yields:
        None

    Raises:
        ImproverError with additional context
    """
    with error_context(
        component="Critic",
        operation=operation,
        error_class=ImproverError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add critic name to the context if provided
        if critic_name:
            context.metadata["critic_name"] = critic_name
        yield


@contextmanager
def chain_context(
    operation: Optional[str] = None,
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Context manager for chain operations.

    Args:
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include

    Yields:
        None

    Raises:
        ChainError with additional context
    """
    with error_context(
        component="Chain",
        operation=operation,
        error_class=ChainError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        yield context


@contextmanager
def storage_context(
    storage_type: Optional[str] = None,
    operation: Optional[str] = "storage",
    message_prefix: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Context manager for storage operations.

    Args:
        storage_type: Optional storage type name
        operation: Optional operation name
        message_prefix: Optional prefix for the error message
        suggestions: Optional list of suggestions
        metadata: Optional metadata to include

    Yields:
        None

    Raises:
        StorageError with additional context
    """
    with error_context(
        component="Storage",
        operation=operation,
        error_class=StorageError,
        message_prefix=message_prefix,
        suggestions=suggestions,
        metadata=metadata,
    ) as context:
        # Add storage type to the context if provided
        if storage_type:
            context.metadata["storage_type"] = storage_type
        yield context


# Utility functions for enhanced error messages


def create_actionable_suggestions(error: Exception, component: Optional[str] = None) -> List[str]:
    """Create actionable suggestions based on error type and component.

    Args:
        error: The exception that occurred.
        component: Optional component name.

    Returns:
        List of actionable suggestions.
    """
    suggestions = []
    error_type = type(error).__name__
    error_message = str(error).lower()

    # Connection-related errors
    if "Connection" in error_type or "Network" in error_type or "connection" in error_message:
        suggestions.extend(
            [
                "Check network connectivity",
                "Verify service endpoints are correct",
                "Check if firewall or proxy is blocking connections",
                "Ensure the service is running and accessible",
            ]
        )

    # Timeout errors
    elif "Timeout" in error_type or "timeout" in error_message:
        suggestions.extend(
            [
                "Increase timeout values",
                "Check if the service is overloaded",
                "Verify network latency is acceptable",
                "Consider using async operations for long-running tasks",
            ]
        )

    # Authentication errors
    elif (
        "Auth" in error_type
        or "Permission" in error_type
        or any(
            keyword in error_message
            for keyword in ["auth", "permission", "unauthorized", "forbidden", "api key"]
        )
    ):
        suggestions.extend(
            [
                "Check API keys and credentials",
                "Verify permissions and access rights",
                "Ensure tokens are not expired",
                "Check if the service requires authentication",
            ]
        )

    # Rate limiting errors
    elif (
        "Rate" in error_type
        or "Limit" in error_type
        or any(keyword in error_message for keyword in ["rate limit", "quota", "too many requests"])
    ):
        suggestions.extend(
            [
                "Implement exponential backoff",
                "Reduce request frequency",
                "Check rate limit quotas",
                "Consider upgrading service plan for higher limits",
            ]
        )

    # Model-specific errors
    elif "model" in error_message or component == "Model":
        if "not found" in error_message or "does not exist" in error_message:
            suggestions.extend(
                [
                    "Verify the model name is correct",
                    "Check if the model is available in your region",
                    "Ensure you have access to the specified model",
                    "Try using a different model variant",
                ]
            )
        elif "memory" in error_message or "cuda" in error_message:
            suggestions.extend(
                [
                    "Reduce batch size or sequence length",
                    "Use model quantization (4bit/8bit)",
                    "Try using a smaller model variant",
                    "Clear GPU memory cache",
                ]
            )

    # Storage-specific errors
    elif component in ["Storage", "Redis"]:
        if "connection" in error_message:
            suggestions.extend(
                [
                    "Check if the storage service is running",
                    "Verify connection parameters (host, port, credentials)",
                    "Test network connectivity to the storage service",
                    "Check firewall and security group settings",
                ]
            )

    # Validation-specific errors
    elif component == "Validator":
        suggestions.extend(
            [
                "Check input text format and encoding",
                "Verify validator configuration parameters",
                "Ensure required dependencies are installed",
                "Try with simpler validation rules",
            ]
        )

    # Import/dependency errors
    elif "ImportError" in error_type or "ModuleNotFoundError" in error_type:
        suggestions.extend(
            [
                "Install missing dependencies with pip",
                "Check if optional dependencies are needed",
                "Verify Python environment and package versions",
                "Try reinstalling the package",
            ]
        )

    # File/path errors
    elif "FileNotFoundError" in error_type or "PermissionError" in error_type:
        suggestions.extend(
            [
                "Check if the file or directory exists",
                "Verify file permissions and access rights",
                "Ensure the path is correct and accessible",
                "Check if the directory needs to be created",
            ]
        )

    # JSON/parsing errors
    elif "JSON" in error_type or "json" in error_message:
        suggestions.extend(
            [
                "Check JSON format and syntax",
                "Verify data encoding (UTF-8)",
                "Ensure all quotes and brackets are properly closed",
                "Try validating JSON with an online validator",
            ]
        )

    # Configuration errors
    elif "Config" in error_type or "Setting" in error_type:
        suggestions.extend(
            [
                "Check configuration files and environment variables",
                "Verify all required settings are provided",
                "Ensure configuration values are valid",
                "Check for typos in configuration keys",
            ]
        )

    # Component-specific suggestions
    if component:
        if component.lower() in ["model", "llm"]:
            suggestions.extend(
                [
                    "Verify model name is correct and available",
                    "Check if model supports the requested parameters",
                    "Ensure sufficient API quota for model usage",
                ]
            )
        elif component.lower() in ["retriever", "vector", "database"]:
            suggestions.extend(
                [
                    "Check database connection and credentials",
                    "Verify index exists and is properly configured",
                    "Ensure vector dimensions match expected values",
                ]
            )
        elif component.lower() in ["validator", "critic"]:
            suggestions.extend(
                [
                    "Check if validation rules are correctly configured",
                    "Verify input format matches expected schema",
                    "Ensure all required dependencies are available",
                ]
            )

    # Add generic suggestions if none were added
    if not suggestions:
        suggestions.extend(
            [
                "Check the error message for specific details",
                "Verify configuration and parameters",
                "Consult the documentation for troubleshooting",
                "Enable debug logging for more information",
            ]
        )

    return suggestions[:5]  # Limit to top 5 suggestions


def enhance_error_message(
    error: Exception, component: Optional[str] = None, operation: Optional[str] = None
) -> str:
    """Enhance an error message with context and actionable suggestions.

    Args:
        error: The exception that occurred.
        component: Optional component name.
        operation: Optional operation name.

    Returns:
        An enhanced error message with suggestions.
    """
    base_message = str(error)
    suggestions = create_actionable_suggestions(error, component)

    # Format the enhanced message
    enhanced_message = base_message

    if component:
        enhanced_message = f"[{component}] {enhanced_message}"

    if operation:
        enhanced_message = f"{enhanced_message} (during {operation})"

    if suggestions:
        suggestion_text = "\n  • " + "\n  • ".join(suggestions)
        enhanced_message = f"{enhanced_message}\n\nSuggestions:{suggestion_text}"

    return enhanced_message


def create_troubleshooting_guide(
    error: Exception, component: Optional[str] = None
) -> Dict[str, Any]:
    """Create a comprehensive troubleshooting guide for an error.

    Args:
        error: The exception that occurred.
        component: Optional component name.

    Returns:
        A dictionary with troubleshooting information.
    """
    error_type = type(error).__name__
    error_message = str(error)
    suggestions = create_actionable_suggestions(error, component)

    guide = {
        "error_type": error_type,
        "error_message": error_message,
        "component": component,
        "suggestions": suggestions,
        "common_causes": [],
        "related_docs": [],
    }

    # Add common causes based on error type
    if "Connection" in error_type or "connection" in error_message.lower():
        guide["common_causes"] = [
            "Network connectivity issues",
            "Service not running or accessible",
            "Firewall blocking connections",
            "Incorrect connection parameters",
        ]
        guide["related_docs"] = [
            "docs/guides/storage-setup.md - Storage configuration",
            "docs/troubleshooting/common-issues.md - Network issues",
        ]

    elif "Auth" in error_type or "api key" in error_message.lower():
        guide["common_causes"] = [
            "Missing or invalid API key",
            "Expired authentication token",
            "Insufficient permissions",
            "Service authentication requirements changed",
        ]
        guide["related_docs"] = [
            "docs/api/api-reference.md - Authentication",
            "README.md - Environment setup",
        ]

    elif "Import" in error_type or "Module" in error_type:
        guide["common_causes"] = [
            "Missing optional dependencies",
            "Incorrect package installation",
            "Python environment issues",
            "Package version conflicts",
        ]
        guide["related_docs"] = [
            "README.md - Installation instructions",
            "docs/CONTRIBUTING.md - Development setup",
        ]

    return guide
