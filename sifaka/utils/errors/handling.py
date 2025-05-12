"""
Error Handling Functions

This module provides functions for standardized error handling in the Sifaka framework.
These functions handle errors in a consistent way, providing detailed error information
and appropriate logging.

## Functions
- **handle_error**: Process an error and return standardized error metadata
- **try_operation**: Execute an operation with standardized error handling
- **log_error**: Log an error with standardized formatting
- **handle_component_error**: Handle errors from generic components
- **create_error_handler**: Create a component-specific error handler
"""

import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from ..logging import get_logger
from .base import SifakaError
from .component import (
    ChainError,
    ClassifierError,
    CriticError,
    ModelError,
    RetrievalError,
    RuleError,
)

# Import ErrorResult type for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import ErrorResult

# Configure logger
logger = get_logger(__name__)

# Type variable for return type
T = TypeVar("T")


def handle_error(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Handle an error and return standardized error metadata.

    This function processes an error, logs it with the specified log level,
    and returns standardized error metadata. It extracts information from
    the error and formats it in a consistent way, making it easier to handle
    errors across the codebase.

    Args:
        error (Exception): The exception to handle
        component_name (str): Name of the component where the error occurred
        log_level (str): Log level to use (default: "error")
        include_traceback (bool): Whether to include traceback in metadata
        additional_metadata (Optional[Dict[str, Any]]): Additional metadata to include

    Returns:
        Dict[str, Any]: Standardized error metadata dictionary containing:
            - error_type: The type of the error (class name)
            - error_message: The error message
            - component: The name of the component where the error occurred
            - traceback: The error traceback (if include_traceback is True)
            - Any additional metadata provided

    Raises:
        Exception: If an error occurs during error handling (rare)

    Examples:
        ```python
        from sifaka.utils.errors.handling import handle_error

        # Handle an error with default settings
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            error_metadata = handle_error(e, "DataProcessor")
            print(f"Error type: {error_metadata['error_type']}")
            print(f"Error message: {error_metadata['error_message']}")
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


def log_error(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    additional_message: Optional[str] = None,
) -> None:
    """Log an error with standardized formatting.

    This function logs an error with standardized formatting, including
    component name, error type, and error message. It provides a consistent
    way to log errors across the codebase.

    Args:
        error (Exception): The exception to log
        component_name (str): Name of the component where the error occurred
        log_level (str): Log level to use (default: "error")
        additional_message (Optional[str]): Additional message to include in the log

    Raises:
        Exception: If an error occurs during logging (rare)
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


def handle_component_error(
    error: Exception,
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> "ErrorResult":
    """Generic error handler for any component type.

    This function handles errors for any component type, converting
    generic exceptions to specific SifakaError types and returning
    standardized error results.

    Args:
        error: The exception to handle
        component_name: Name of the component where the error occurred
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    # Convert to specific error type if not already a SifakaError
    if not isinstance(error, SifakaError):
        error = error_class(
            f"{component_type} error in {component_name}: {str(error)}",
            metadata=additional_metadata,
        )

    # Handle the error
    error_metadata = handle_error(
        error,
        component_name=f"{component_type}:{component_name}",
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Return error result
    return ErrorResult(
        error_type=error_metadata["error_type"],
        error_message=error_metadata["error_message"],
        component_name=component_name,
        metadata=error_metadata,
    )


def create_error_handler(
    component_type: str, error_class: Type[SifakaError]
) -> Callable[[Exception, str, str, bool, Optional[Dict[str, Any]]], "ErrorResult"]:
    """Create an error handler for a specific component type.

    This factory function creates an error handler for a specific component type,
    using the generic handle_component_error function with the appropriate
    component type and error class.

    Args:
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion

    Returns:
        An error handler function for the specified component type
    """

    def handler(
        error: Exception,
        component_name: str,
        log_level: str = "error",
        include_traceback: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> "ErrorResult":
        return handle_component_error(
            error=error,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

    # Set function name and docstring
    handler.__name__ = f"handle_{component_type.lower()}_error"
    handler.__doc__ = f"""Handle a {component_type.lower()} error and return a standardized error result.

    Args:
        error: The exception to handle
        component_name: Name of the {component_type.lower()} where the error occurred
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """

    return handler


# Create specific error handlers using the factory
handle_chain_error = create_error_handler("Chain", ChainError)
handle_model_error = create_error_handler("Model", ModelError)
handle_rule_error = create_error_handler("Rule", RuleError)
handle_critic_error = create_error_handler("Critic", CriticError)
handle_classifier_error = create_error_handler("Classifier", ClassifierError)
handle_retrieval_error = create_error_handler("Retrieval", RetrievalError)
