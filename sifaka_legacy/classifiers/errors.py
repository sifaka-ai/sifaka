"""
Classifier Error Handling

This module provides error handling utilities for the classifiers component.
It defines standard error classes and utility functions for safely executing
classifier operations.

## Usage Examples

```python
# Import error handling utilities
from sifaka.classifiers import errors

# Define an operation that might fail
def risky_operation():
    # Some code that might raise an exception
    return result

# Safely execute the operation
result = errors.safely_execute(
    operation=risky_operation,
    component_name="my_classifier",
    component_type="ClassifierImplementation",
    error_class=errors.ClassifierError,
)
```
"""

from typing import Any, Callable, Optional, Type, TypeVar

from sifaka.utils.errors.base import ComponentError
from sifaka.utils.errors.safe_execution import safely_execute_component

# Type variables
T = TypeVar("T")  # Return type


class ClassifierError(ComponentError):
    """Base error class for classifier errors."""

    def __init__(
        self,
        message: str,
        component_name: Optional[Optional[str]] = None,
        component_type: str = "classifier",
        error_type: str = "classifier_error",
        **kwargs: Any,
    ):
        """
        Initialize the error.

        Args:
            message: Error message
            component_name: Name of the component that raised the error
            component_type: Type of the component that raised the error
            error_type: Type of error
            **kwargs: Additional error metadata
        """
        super().__init__(
            message=message,
            component_name=component_name,
            component_type=component_type,
            error_type=error_type,
            **kwargs,
        )


class ImplementationError(ClassifierError):
    """Error raised when a classifier implementation fails."""

    def __init__(
        self,
        message: str,
        component_name: Optional[Optional[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the error.

        Args:
            message: Error message
            component_name: Name of the component that raised the error
            **kwargs: Additional error metadata
        """
        super().__init__(
            message=message,
            component_name=component_name,
            component_type="ClassifierImplementation",
            error_type="implementation_error",
            **kwargs,
        )


class ConfigurationError(ClassifierError):
    """Error raised when a classifier configuration is invalid."""

    def __init__(
        self,
        message: str,
        component_name: Optional[Optional[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the error.

        Args:
            message: Error message
            component_name: Name of the component that raised the error
            **kwargs: Additional error metadata
        """
        super().__init__(
            message=message,
            component_name=component_name,
            component_type="ClassifierConfiguration",
            error_type="configuration_error",
            **kwargs,
        )


def safely_execute(
    operation: Callable[[], T],
    component_name: Optional[Optional[str]] = None,
    component_type: str = "classifier",
    error_class: Type[Exception] = ClassifierError,
    **kwargs: Any,
) -> T:
    """
    Safely execute a classifier operation.

    This function wraps the operation in a try-except block and handles
    exceptions in a standardized way.

    Args:
        operation: The operation to execute
        component_name: Name of the component executing the operation
        component_type: Type of the component executing the operation
        error_class: Error class to use for exceptions
        **kwargs: Additional error metadata

    Returns:
        The result of the operation

    Raises:
        error_class: If the operation fails
    """
    return safely_execute_component(
        operation=operation,
        component_name=component_name,
        component_type=component_type,
        error_class=error_class,
        **kwargs,
    )
