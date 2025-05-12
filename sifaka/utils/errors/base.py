"""
Base Error Classes

This module defines the base error classes for the Sifaka framework.
These classes form the foundation of the error hierarchy and provide
standardized error handling functionality.

## Classes
- **SifakaError**: Base class for all Sifaka exceptions
- **ValidationError**: Raised when validation fails
- **ConfigurationError**: Raised when configuration is invalid
- **ProcessingError**: Raised when processing fails
- **ResourceError**: Raised when a resource is unavailable
- **TimeoutError**: Raised when an operation times out
- **InputError**: Raised when input is invalid
- **StateError**: Raised when state is invalid
- **DependencyError**: Raised when a dependency is missing or invalid
- **InitializationError**: Raised when initialization fails
- **ComponentError**: Base class for component-specific errors
"""

from typing import Any, Dict, Optional


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


class ComponentError(SifakaError):
    """Base class for component-specific errors.

    This class provides a standardized structure for component-specific errors,
    including component name, component type, and error type. It serves as the
    base class for all component-specific error classes.

    Attributes:
        message (str): Human-readable error message
        component_name (Optional[str]): Name of the component that raised the error
        component_type (str): Type of the component that raised the error
        error_type (str): Type of error
        metadata (Dict[str, Any]): Additional error context and details
    """

    def __init__(
        self,
        message: str,
        component_name: Optional[str] = None,
        component_type: str = "component",
        error_type: str = "component_error",
        **kwargs: Any,
    ):
        """Initialize a ComponentError with a message and component information.

        Args:
            message (str): Human-readable error message
            component_name (Optional[str]): Name of the component that raised the error
            component_type (str): Type of the component that raised the error
            error_type (str): Type of error
            **kwargs: Additional error metadata
        """
        metadata = {
            "component_name": component_name,
            "component_type": component_type,
            "error_type": error_type,
            **kwargs,
        }
        super().__init__(message, metadata=metadata)
