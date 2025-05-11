"""
Error Handling Module

This module provides standardized error handling utilities for the Sifaka framework,
including exception classes, error handling functions, and logging utilities.

## Overview
The error handling system in Sifaka provides a structured approach to handling
errors across the framework. It includes a comprehensive exception hierarchy,
standardized error handling patterns, and component-specific error handlers.

The system is designed to provide consistent error handling behavior, detailed
error information, and appropriate logging across all components.

## Components
- **Exception Classes**: Structured hierarchy of exception classes
- **Error Handling Functions**: Functions for standardized error handling
- **Error Result Classes**: Models for representing errors in results
- **Component-Specific Error Handlers**: Specialized handlers for different components
- **Safe Execution Functions**: Functions for safely executing operations

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
4. **safely_execute_component_operation**: Safely execute a component operation with standardized error handling
5. **Component-specific error handlers**: Specialized error handlers for different component types

## Usage Examples

```python
from sifaka.utils.errors import (
    SifakaError, ValidationError, try_operation, handle_error,
    safely_execute_chain, ErrorResult
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

# Using component-specific error handlers
try:
    # Chain operation
    result = chain.run(prompt)
except Exception as e:
    # Handle chain error
    error_result = handle_chain_error(e, component_name="MyChain")
    print(f"Chain error: {error_result.error_message}")

# Using safely_execute_component_operation
result = safely_execute_chain(
    lambda: chain.run(prompt),
    component_name="MyChain"
)
if isinstance(result, ErrorResult):
    print(f"Chain error: {result.error_message}")
else:
    print(f"Chain result: {result}")
```

## Error Handling
The error handling system itself is designed to be robust and fail gracefully.
If an error occurs during error handling, it will be logged and a default
error result will be returned.

## Configuration
Error handling behavior can be configured through log levels and additional
metadata. Component-specific error handlers can be customized for different
error handling requirements.
"""

import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast
from pydantic import BaseModel

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
    "InitializationError",
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
    # Error result classes
    "ErrorResult",
    # Component-specific error handlers
    "handle_component_error",
    "create_error_handler",
    "handle_chain_error",
    "handle_model_error",
    "handle_rule_error",
    "handle_critic_error",
    "handle_classifier_error",
    "handle_retrieval_error",
    # Error result creation functions
    "create_error_result",
    "create_error_result_factory",
    "create_chain_error_result",
    "create_model_error_result",
    "create_rule_error_result",
    "create_critic_error_result",
    "create_classifier_error_result",
    "create_retrieval_error_result",
    # Safe execution functions
    "try_component_operation",
    "safely_execute_component_operation",
    "create_safe_execution_factory",
    "safely_execute_chain",
    "safely_execute_model",
    "safely_execute_rule",
    "safely_execute_critic",
    "safely_execute_classifier",
    "safely_execute_retrieval",
]

# Type variable for return type
T = TypeVar("T")


class SifakaError(Exception):
    """Base class for all Sifaka exceptions.

    This class provides a standardized structure for Sifaka exceptions,
    including a message and optional metadata. All other exceptions in the
    Sifaka framework should inherit from this class.

    ## Architecture
    SifakaError serves as the root of the Sifaka exception hierarchy. It extends
    the standard Python Exception class and adds structured metadata support.
    The message and metadata attributes provide a consistent way to include
    detailed error information.

    ## Lifecycle
    1. **Creation**: Instantiated with a message and optional metadata
    2. **Usage**: Raised to signal errors in Sifaka components
    3. **Handling**: Caught and processed by error handling utilities

    ## Examples
    ```python
    # Creating and raising a SifakaError
    raise SifakaError("Operation failed", metadata={"operation": "process_data"})

    # Catching and handling a SifakaError
    try:
        # Some operation
        process_data(input_data)
    except SifakaError as e:
        print(f"Error: {e.message}")
        print(f"Metadata: {e.metadata}")
    ```

    Attributes:
        message (str): Human-readable error message
        metadata (Dict[str, Any]): Additional error context and details
    """

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a SifakaError with a message and optional metadata.

        This constructor initializes a SifakaError with a human-readable message
        and optional metadata dictionary. The message is used as the exception
        message, and the metadata provides additional context for error handling.

        Args:
            message (str): Human-readable error message
            metadata (Optional[Dict[str, Any]]): Additional error context and details

        Example:
            ```python
            # Create a basic error
            error = SifakaError("Validation failed")

            # Create an error with metadata
            error = SifakaError(
                "Validation failed",
                metadata={"field": "name", "value": "invalid value"}
            )
            ```
        """
        self.message = message
        self.metadata = metadata or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Get string representation of the error.

        This method returns a string representation of the error, including
        the message and metadata if available. This is used when the error
        is printed or converted to a string.

        Returns:
            str: String representation of the error

        Example:
            ```python
            error = SifakaError("Validation failed", metadata={"field": "name"})
            print(error)  # Outputs: "Validation failed (metadata: {'field': 'name'})"
            ```
        """
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


class InitializationError(SifakaError):
    """Error raised when component initialization fails.

    This error is raised when a component fails to initialize properly,
    such as when required resources cannot be loaded or configured.
    """

    pass


# Component-specific error classes


class ChainError(SifakaError):
    """Error raised by chain components.

    This error is raised when a chain component encounters an error,
    such as during orchestration, execution, or result processing.

    ## Architecture
    ChainError is a specialized SifakaError for chain-related errors.
    It inherits all functionality from SifakaError and serves as the
    base class for more specific chain-related errors.

    ## Examples
    ```python
    # Raising a ChainError
    raise ChainError(
        "Chain execution failed",
        metadata={"chain_id": "text_generation", "step": "model_call"}
    )

    # Catching chain-specific errors
    try:
        result = chain.run(prompt)
    except ChainError as e:
        print(f"Chain error: {e.message}")
        print(f"Chain error metadata: {e.metadata}")
    except SifakaError as e:
        print(f"Other Sifaka error: {e.message}")
    ```

    Attributes:
        message (str): Human-readable error message
        metadata (Dict[str, Any]): Additional error context and details
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
            error_metadata = handle_error(
                e,
                "DataProcessor",
                log_level="warning"
            )

        # Handle an error without traceback
        try:
            # Some operation
            result = process_data(input_data)
        except Exception as e:
            error_metadata = handle_error(
                e,
                "DataProcessor",
                include_traceback=False
            )

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
    component name, error type, and error message. It provides a consistent
    way to log errors across the codebase.

    Args:
        error (Exception): The exception to log
        component_name (str): Name of the component where the error occurred
        log_level (str): Log level to use (default: "error")
        additional_message (Optional[str]): Additional message to include in the log

    Raises:
        Exception: If an error occurs during logging (rare)

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


class ErrorResult(BaseModel):
    """Result of an error handling operation.

    This model provides a standardized structure for error results,
    including error type, message, component name, and metadata.
    It is used to represent errors in a structured way that can be
    returned from operations instead of raising exceptions.

    ## Architecture
    ErrorResult is a Pydantic model that represents errors in a structured format.
    It is used by the safe execution functions to return errors as values rather
    than raising exceptions, allowing for more flexible error handling patterns.

    ## Examples
    ```python
    # Creating an ErrorResult
    error = ErrorResult(
        error_type="ValidationError",
        error_message="Invalid input",
        component_name="TextValidator",
        metadata={"field": "text", "max_length": 100}
    )

    # Using ErrorResult in a function
    def process_data(data):
        if not validate(data):
            return ErrorResult(
                error_type="ValidationError",
                error_message="Invalid data",
                component_name="DataProcessor",
                metadata={"data": data}
            )
        # Process data
        return result

    # Handling ErrorResult
    result = process_data(input_data)
    if isinstance(result, ErrorResult):
        print(f"Error: {result.error_message}")
        print(f"Component: {result.component_name}")
        print(f"Error type: {result.error_type}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"Result: {result}")
    ```

    Attributes:
        error_type (str): Type of the error (e.g., "ValidationError")
        error_message (str): Human-readable error message
        component_name (str): Name of the component where the error occurred
        metadata (Dict[str, Any]): Additional error context and details
    """

    error_type: str
    error_message: str
    component_name: str
    metadata: Dict[str, Any] = {}


# Generic component error handling


def handle_component_error(
    error: Exception,
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
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


# Error handler factory


def create_error_handler(
    component_type: str, error_class: Type[SifakaError]
) -> Callable[[Exception, str, str, bool, Optional[Dict[str, Any]]], ErrorResult]:
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
    ) -> ErrorResult:
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


# Generic error result creation function
def create_error_result(
    error: Exception,
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> ErrorResult:
    """Create a standardized error result for any component type.

    This function creates a standardized error result for any component type,
    using the appropriate error handler based on the component type.

    Args:
        error: The exception that occurred
        component_name: Name of the component where the error occurred
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion
        log_level: Log level to use (default: "error")
        include_traceback: Whether to include traceback in metadata
        additional_metadata: Additional metadata to include

    Returns:
        Standardized error result
    """
    return handle_component_error(
        error=error,
        component_name=component_name,
        component_type=component_type,
        error_class=error_class,
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )


# Factory function for creating component-specific error result functions
def create_error_result_factory(component_type: str, error_class: Type[SifakaError]) -> Callable:
    """Create an error result factory for a specific component type.

    This factory function creates an error result function for a specific component type,
    using the generic create_error_result function with the appropriate component type and error class.

    Args:
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion

    Returns:
        An error result function for the specified component type
    """

    def factory(
        error: Exception,
        component_name: str,
        log_level: str = "error",
        include_traceback: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> ErrorResult:
        """Create a standardized error result for a specific component type."""
        return create_error_result(
            error=error,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

    return factory


# Create component-specific error result functions
create_chain_error_result = create_error_result_factory("Chain", ChainError)
create_model_error_result = create_error_result_factory("Model", ModelError)
create_rule_error_result = create_error_result_factory("Rule", RuleError)
create_critic_error_result = create_error_result_factory("Critic", CriticError)
create_classifier_error_result = create_error_result_factory("Classifier", ClassifierError)
create_retrieval_error_result = create_error_result_factory("Retrieval", RetrievalError)


# Function to safely execute component operations with standardized error handling
def safely_execute_component_operation(
    operation: Callable[[], T],
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    default_result: Optional[Union[T, ErrorResult]] = None,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Union[T, ErrorResult]:
    """
    Safely execute a component operation with standardized error handling.

    This function executes an operation and handles any errors that occur,
    providing standardized error handling and logging. It returns either
    the operation result or an ErrorResult object.

    Args:
        operation: The operation to execute
        component_name: Name of the component executing the operation
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for wrapping generic exceptions
        default_result: Value or ErrorResult to return if operation fails
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Either the operation result or an ErrorResult object

    Examples:
        ```python
        from sifaka.utils.errors import safely_execute_component_operation, ChainError

        # Execute a chain operation safely
        result = safely_execute_component_operation(
            lambda: chain.run(prompt),
            component_name="MyChain",
            component_type="Chain",
            error_class=ChainError
        )

        # Check if result is an error
        if isinstance(result, ErrorResult):
            print(f"Chain error: {result.error_message}")
        else:
            print(f"Chain result: {result}")
        ```
    """
    try:
        # Try to execute the operation
        return try_component_operation(
            operation=operation,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )
    except Exception as e:
        # If operation fails, create an error result
        if default_result is not None:
            return default_result

        return create_error_result(
            error=e,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )


# Factory function for creating component-specific safe execution functions
def create_safe_execution_factory(component_type: str, error_class: Type[SifakaError]) -> Callable:
    """
    Create a safe execution factory for a specific component type.

    This factory function creates a safe execution function for a specific component type,
    using the generic safely_execute_component_operation function with the appropriate
    component type and error class.

    Args:
        component_type: Type of the component (e.g., "Chain", "Model")
        error_class: SifakaError subclass to use for conversion

    Returns:
        A safe execution function for the specified component type
    """

    def factory(
        operation: Callable[[], T],
        component_name: str,
        default_result: Optional[Union[T, ErrorResult]] = None,
        log_level: str = "error",
        include_traceback: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[T, ErrorResult]:
        """Safely execute an operation for a specific component type."""
        return safely_execute_component_operation(
            operation=operation,
            component_name=component_name,
            component_type=component_type,
            error_class=error_class,
            default_result=default_result,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

    return factory


# Create component-specific safe execution functions
safely_execute_chain = create_safe_execution_factory("Chain", ChainError)
safely_execute_model = create_safe_execution_factory("Model", ModelError)
safely_execute_rule = create_safe_execution_factory("Rule", RuleError)
safely_execute_critic = create_safe_execution_factory("Critic", CriticError)
safely_execute_classifier = create_safe_execution_factory("Classifier", ClassifierError)
safely_execute_retrieval = create_safe_execution_factory("Retrieval", RetrievalError)
