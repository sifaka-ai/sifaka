"""
Error handling utilities for Sifaka.

This module provides standardized error handling utilities for the Sifaka framework,
including exception classes, error handling functions, and logging utilities.

## Exception Hierarchy

Sifaka uses a structured exception hierarchy:

1. **SifakaError**: Base class for all Sifaka exceptions
   - **ValidationError**: Raised when validation fails
   - **ConfigurationError**: Raised when configuration is invalid
   - **ProcessingError**: Raised when processing fails
     - **ResourceError**: Raised when a resource is unavailable
     - **TimeoutError**: Raised when an operation times out

2. **Component-Specific Errors**:
   - **ChainError**: Raised by chain components
   - **ModelError**: Raised by model providers
   - **RuleError**: Raised during rule validation
   - **CriticError**: Raised by critics
   - **ClassifierError**: Raised by classifiers
   - **RetrievalError**: Raised during retrieval operations

3. **Common Error Types**:
   - **InputError**: Raised when input is invalid
   - **StateError**: Raised when state is invalid
   - **DependencyError**: Raised when a dependency fails

## Error Handling Patterns

The module provides standardized error handling patterns:

1. **try_operation**: Execute an operation with standardized error handling
2. **handle_error**: Process an error and return standardized metadata
3. **log_error**: Log an error with standardized formatting

## Usage Examples

```python
from sifaka.utils.errors import (
    SifakaError, ValidationError, try_operation, handle_error
)

# Using exception classes
try:
    # Some operation
    if invalid_condition:
        raise ValidationError("Invalid input", metadata={"field": "name"})
except SifakaError as e:
    # Handle Sifaka-specific error
    print(f"Error: {e.message}")
    print(f"Metadata: {e.metadata}")
except Exception as e:
    # Handle other errors
    error_info = handle_error(e, "MyComponent")
    print(f"Unexpected error: {error_info['error_message']}")

# Using try_operation
result = try_operation(
    lambda: process_data(input_data),
    component_name="DataProcessor",
    default_value=None,
    log_level="error"
)
```
"""

import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from .logging import get_logger

# Configure logger
logger = get_logger(__name__)

__all__ = [
    # Base error classes
    "SifakaError",
    "ValidationError",
    "ConfigurationError",
    "ProcessingError",
    "ResourceError",
    "TimeoutError",
    "InputError",
    "StateError",
    "DependencyError",
    # Component-specific error classes
    "ChainError",
    "ImproverError",
    "FormatterError",
    "PluginError",
    "ModelError",
    "RuleError",
    "CriticError",
    "ClassifierError",
    "RetrievalError",
    # Error handling functions
    "handle_error",
    "try_operation",
    "log_error",
    # New standardized error handling functions
    "try_component_operation",
]

# Type variable for return type
T = TypeVar("T")


class SifakaError(Exception):
    """Base class for all Sifaka exceptions.

    This class provides a standardized structure for Sifaka exceptions,
    including a message and optional metadata.

    Attributes:
        message: Human-readable error message
        metadata: Additional error context and details
    """

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a SifakaError.

        Args:
            message: Human-readable error message
            metadata: Additional error context and details
        """
        self.message = message
        self.metadata = metadata or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Get string representation of the error."""
        if self.metadata:
            return f"{self.message} (metadata: {self.metadata})"
        return self.message


class ValidationError(SifakaError):
    """Error raised when validation fails.

    This error is raised when input validation fails, such as when
    a rule, validator, or classifier encounters invalid input.
    """

    pass


class ConfigurationError(SifakaError):
    """Error raised when configuration is invalid.

    This error is raised when a component's configuration is invalid,
    such as when required parameters are missing or have invalid values.
    """

    pass


class ProcessingError(SifakaError):
    """Error raised when processing fails.

    This error is raised when a processing operation fails, such as
    when a classifier, critic, or rule encounters an error during processing.
    """

    pass


class ResourceError(ProcessingError):
    """Error raised when a resource is unavailable.

    This error is raised when a required resource is unavailable,
    such as when a model, database, or external service is unreachable.
    """

    pass


class TimeoutError(ProcessingError):
    """Error raised when an operation times out.

    This error is raised when an operation takes too long to complete,
    such as when a model inference or external API call times out.
    """

    pass


class InputError(ValidationError):
    """Error raised when input is invalid.

    This error is raised when input validation fails due to invalid
    input format, type, or content.
    """

    pass


class StateError(SifakaError):
    """Error raised when state is invalid.

    This error is raised when a component's state is invalid, such as
    when a required resource is not initialized or a state transition is invalid.
    """

    pass


class DependencyError(SifakaError):
    """Error raised when a dependency fails.

    This error is raised when a dependency fails, such as when an
    external service, library, or component encounters an error.
    """

    pass


# Component-specific error classes


class ChainError(SifakaError):
    """Error raised by chain components.

    This error is raised when a chain component encounters an error,
    such as during orchestration, execution, or result processing.
    """

    pass


class ImproverError(ChainError):
    """Error raised when improver refinement fails.

    This error is raised when an improver component fails to refine output,
    such as during critic-based improvement or other refinement processes.
    """

    pass


class FormatterError(ChainError):
    """Error raised when result formatting fails.

    This error is raised when a formatter component fails to format results,
    such as during output formatting or structure conversion.
    """

    pass


class PluginError(ChainError):
    """Error raised when plugin operations fail.

    This error is raised when a plugin encounters an error during execution,
    such as during initialization, processing, or cleanup.
    """

    pass


class ModelError(SifakaError):
    """Error raised by model providers.

    This error is raised when a model provider encounters an error,
    such as during model initialization, inference, or API communication.
    """

    pass


class RuleError(ValidationError):
    """Error raised during rule validation.

    This error is raised when a rule encounters an error during validation,
    such as when a rule's validation logic fails or produces unexpected results.
    """

    pass


class CriticError(SifakaError):
    """Error raised by critics.

    This error is raised when a critic encounters an error,
    such as during critique generation, feedback processing, or improvement.
    """

    pass


class ClassifierError(SifakaError):
    """Error raised by classifiers.

    This error is raised when a classifier encounters an error,
    such as during classification, model inference, or result processing.
    """

    pass


class RetrievalError(SifakaError):
    """Error raised during retrieval operations.

    This error is raised when a retrieval operation encounters an error,
    such as during document retrieval, indexing, or query processing.
    """

    pass


def handle_error(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Handle an error and return standardized error metadata.

    This function processes an error, logs it with the specified log level,
    and returns standardized error metadata.

    Args:
        error: The exception to handle
        component_name: Name of the component where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error metadata dictionary

    Examples:
        ```python
        from sifaka.utils.errors import handle_error

        # Handle an error with default settings
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            error_metadata = handle_error(e, "DataProcessor")
            print(f"Error type: {error_metadata['error_type']}")
            print(f"Error message: {error_metadata['error_message']}")

        # Handle an error with custom log level
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            error_metadata = handle_error(e, "DataProcessor", log_level="warning")

        # Handle an error without traceback
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            error_metadata = handle_error(e, "DataProcessor", include_traceback=False)

        # Handle an error with additional metadata
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            error_metadata = handle_error(
                e,
                "DataProcessor",
                additional_metadata={
                    "input_size": len(input_data),
                    "operation": "preprocessing"
                }
            )
        ```
    """
    # Extract error details
    error_type = type(error).__name__
    error_message = str(error)

    # Create metadata
    metadata = {
        "error_type": error_type,
        "error_message": error_message,
        "component": component_name,
    }

    # Add traceback if requested
    if include_traceback:
        metadata["traceback"] = traceback.format_exc()

    # Add metadata from SifakaError
    if isinstance(error, SifakaError) and error.metadata:
        metadata.update(error.metadata)

    # Add additional metadata
    if additional_metadata:
        metadata.update(additional_metadata)

    # Log the error
    log_message = f"{component_name}: {error_type} - {error_message}"
    getattr(logger, log_level)(log_message, extra={"metadata": metadata})

    return metadata


def try_operation(
    operation: Callable[[], T],
    component_name: str,
    component_type: Optional[str] = None,
    error_class: Optional[Type[SifakaError]] = None,
    default_value: Optional[T] = None,
    log_level: str = "error",
    error_handler: Optional[Callable[[Exception], Optional[T]]] = None,
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> T:
    """Execute an operation with standardized error handling.

    This function executes an operation and handles any errors that occur,
    providing standardized error handling and logging. It can also wrap generic
    exceptions in component-specific error classes.

    Args:
        operation: The operation to execute
        component_name: Name of the component executing the operation
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for wrapping generic exceptions
        default_value: Value to return if operation fails
        log_level: Log level to use for errors
        error_handler: Custom error handler function
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Result of the operation or default value if it fails

    Raises:
        SifakaError: If error_class is provided and a generic exception occurs
        Exception: If error_handler raises an exception or no default value is provided

    Examples:
        ```python
        from sifaka.utils.errors import try_operation, ChainError

        # Basic usage with default value
        result = try_operation(
            lambda: process_data(input_data),
            component_name="DataProcessor",
            default_value=None
        )

        # With component type and error class
        result = try_operation(
            lambda: process_data(input_data),
            component_name="MyChain",
            component_type="Chain",
            error_class=ChainError,
            default_value=None
        )

        # With custom log level
        result = try_operation(
            lambda: process_data(input_data),
            component_name="DataProcessor",
            default_value=None,
            log_level="warning"
        )

        # With custom error handler
        def custom_error_handler(e: Exception) -> Optional[str]:
            if isinstance(e, ValueError):
                return "Invalid value"
            return None

        result = try_operation(
            lambda: process_data(input_data),
            component_name="DataProcessor",
            default_value="Unknown error",
            error_handler=custom_error_handler
        )

        # With additional metadata
        result = try_operation(
            lambda: process_data(input_data),
            component_name="DataProcessor",
            default_value=None,
            additional_metadata={"input_size": len(input_data)}
        )

        # Using with a function that returns a specific type
        def get_user_count() -> int:
            # Database operation that might fail
            return db.query("SELECT COUNT(*) FROM users").scalar()

        user_count = try_operation(
            get_user_count,
            component_name="UserCounter",
            default_value=0
        )
        ```
    """
    try:
        return operation()
    except Exception as e:
        # Handle the error
        error_metadata = handle_error(
            e, component_name, log_level, include_traceback, additional_metadata
        )

        # Use custom error handler if provided
        if error_handler:
            result = error_handler(e)
            if result is not None:
                return result

        # Return default value if provided
        if default_value is not None:
            return default_value

        # Wrap in component-specific error if requested
        if component_type and error_class and not isinstance(e, SifakaError):
            error_message = f"{component_type} error in {component_name}: {str(e)}"
            raise error_class(error_message, metadata=error_metadata) from e

        # Re-raise the original exception
        raise


def try_component_operation(
    operation: Callable[[], T],
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    default_value: Optional[T] = None,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> T:
    """Execute an operation with component-specific error handling.

    This function is a convenience wrapper around try_operation that provides
    component-specific error handling. It automatically wraps generic exceptions
    in the specified error class and includes component type information.

    Args:
        operation: The operation to execute
        component_name: Name of the component executing the operation
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for wrapping generic exceptions
        default_value: Value to return if operation fails
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Result of the operation or default value if it fails

    Raises:
        SifakaError: If a generic exception occurs
        Exception: If no default value is provided

    Examples:
        ```python
        from sifaka.utils.errors import try_component_operation, ChainError

        # Basic usage with default value
        result = try_component_operation(
            lambda: process_data(input_data),
            component_name="MyChain",
            component_type="Chain",
            error_class=ChainError,
            default_value=None
        )

        # With additional metadata
        result = try_component_operation(
            lambda: process_data(input_data),
            component_name="MyChain",
            component_type="Chain",
            error_class=ChainError,
            default_value=None,
            additional_metadata={"input_size": len(input_data)}
        )
        ```
    """
    return try_operation(
        operation=operation,
        component_name=component_name,
        component_type=component_type,
        error_class=error_class,
        default_value=default_value,
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )


def log_error(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    additional_message: Optional[str] = None,
) -> None:
    """Log an error with standardized formatting.

    This function logs an error with standardized formatting, including
    component name, error type, and error message.

    Args:
        error: The exception to log
        component_name: Name of the component where the error occurred
        log_level: Log level to use
        additional_message: Additional message to include in the log

    Examples:
        ```python
        from sifaka.utils.errors import log_error

        # Log an error with default log level (error)
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            log_error(e, "DataProcessor")

        # Log an error with custom log level
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            log_error(e, "DataProcessor", log_level="warning")

        # Log an error with additional message
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            log_error(
                e,
                "DataProcessor",
                additional_message="Failed during data preprocessing"
            )
        ```
    """
    # Extract error details
    error_type = type(error).__name__
    error_message = str(error)

    # Create log message
    log_message = f"{component_name}: {error_type} - {error_message}"
    if additional_message:
        log_message = f"{log_message} - {additional_message}"

    # Log the error
    getattr(logger, log_level)(log_message, exc_info=True)
