"""
Safe Execution Functions and Factories

This module provides functions for safely executing operations with standardized
error handling. These functions allow operations to be executed in a way that
handles errors consistently and returns either the operation result or an
error result.

## Functions
- **try_component_operation**: Try to execute a component operation
- **safely_execute_component_operation**: Safely execute a component operation
- **create_safe_execution_factory**: Create a component-specific safe execution factory
- Component-specific safe execution functions:
  - **safely_execute_chain**
  - **safely_execute_model**
  - **safely_execute_rule**
  - **safely_execute_critic**
  - **safely_execute_classifier**
  - **safely_execute_retrieval**
- Functions consolidated from error_patterns.py:
  - **safely_execute_component**
"""

import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

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
from .handling import handle_error
from .results import ErrorResult, create_error_result

# Configure logger
logger = get_logger(__name__)

# Type variable for return type
T = TypeVar("T")


def try_component_operation(
    operation: Callable[[], T],
    component_name: str,
    component_type: str,
    error_class: Type[SifakaError],
    default_value: Optional[Optional[T]] = None,
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
    """
    try:
        # Execute the operation
        return operation()
    except Exception as e:
        # Handle the error
        error_metadata = handle_error(
            e, component_name, log_level, include_traceback, additional_metadata
        )

        # Return default value if provided
        if default_value is not None:
            return default_value

        # Wrap in component-specific error if not already a SifakaError
        if not isinstance(e, SifakaError):
            error_message = f"{component_type} error in {component_name}: {str(e)}"
            raise error_class(error_message, metadata=error_metadata) from e

        # Re-raise the original exception
        raise


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
        from sifaka.utils.errors.safe_execution import safely_execute_component_operation, ChainError

        # Execute a chain operation safely
        result = safely_execute_component_operation(
            lambda: (chain.run(prompt),
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


def safely_execute_component(
    operation: Callable[[], T],
    component_name: Optional[Optional[str]] = None,
    component_type: str = "component",
    error_class: Type[Exception] = Exception,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> T:
    """
    Safely execute a component operation with standardized error handling.

    This function executes a component operation and handles any errors that occur,
    providing standardized error handling and logging.

    Args:
        operation: The operation to execute
        component_name: Name of the component executing the operation
        component_type: Type of the component executing the operation
        error_class: Error class to use for exceptions
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error
        **kwargs: Additional error metadata

    Returns:
        Result of the operation

    Raises:
        error_class: If the operation fails
    """
    try:
        # Record start time
        start_time = (time.time()

        # Execute operation
        result = operation()

        # Record execution time
        execution_time = (time.time() - start_time
        (logger and logger.debug(
            f"{(component_type.capitalize()} operation completed in {execution_time:.4f}s",
            extra={
                "execution_time": execution_time,
                "component": component_name,
                "component_type": component_type,
            },
        )

        return result
    except Exception as e:
        # Handle error
        error_metadata = handle_error(
            e,
            component_name or component_type,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata={
                "component_type": component_type,
                **(additional_metadata or {}),
                **kwargs,
            },
        )

        # Raise error with appropriate class
        if isinstance(e, error_class):
            raise
        else:
            raise error_class(
                f"{(component_type.capitalize()} operation failed: {str(e)}",
                **kwargs,
            ) from e
