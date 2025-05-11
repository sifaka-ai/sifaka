"""
Error handling patterns for Sifaka components.

This module provides standardized error handling patterns for Sifaka components,
including safe execution functions for critics, rules, and other components.
"""

import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union, Type, cast

from sifaka.utils.errors import (
    CriticError,
    RuleError,
    SifakaError,
    handle_error,
)
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)

# Type variables
T = TypeVar("T")
R = TypeVar("R")

__all__ = [
    "safely_execute_critic",
    "create_critic_error_result",
    "safely_execute_rule",
    "create_rule_error_result",
    "safely_execute_component",
]


def safely_execute_critic(
    operation: Callable[[], T],
    component_name: str,
    default_value: Optional[T] = None,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Safely execute a critic operation with standardized error handling.

    This function executes a critic operation and handles any errors that occur,
    providing standardized error handling and logging.

    Args:
        operation: The operation to execute
        component_name: Name of the critic executing the operation
        default_value: Value to return if operation fails
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Result of the operation or default value if it fails

    Raises:
        CriticError: If the operation fails and no default value is provided
    """
    try:
        # Record start time
        start_time = time.time()

        # Execute operation
        result = operation()

        # Record execution time
        execution_time = time.time() - start_time
        logger.debug(
            f"Critic operation completed in {execution_time:.4f}s",
            extra={"execution_time": execution_time, "component": component_name},
        )

        return result
    except Exception as e:
        # Handle error
        error_metadata = handle_error(
            e,
            component_name,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

        # If default value is provided, return it
        if default_value is not None:
            return default_value

        # Otherwise, raise CriticError
        if isinstance(e, CriticError):
            raise
        else:
            raise CriticError(f"Critic operation failed: {str(e)}", metadata=error_metadata) from e


def create_critic_error_result(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized error result for a critic operation.

    This function creates a standardized error result for a critic operation,
    including error details, metadata, and logging.

    Args:
        error: The exception that occurred
        component_name: Name of the critic where the error occurred
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Standardized error result dictionary
    """
    # Handle error
    error_metadata = handle_error(
        error,
        component_name,
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Create error result
    return {
        "success": False,
        "error": str(error),
        "error_type": type(error).__name__,
        "component": component_name,
        "metadata": error_metadata,
    }


def safely_execute_rule(
    operation: Callable[[], T],
    component_name: str,
    default_value: Optional[T] = None,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Safely execute a rule operation with standardized error handling.

    This function executes a rule operation and handles any errors that occur,
    providing standardized error handling and logging.

    Args:
        operation: The operation to execute
        component_name: Name of the rule executing the operation
        default_value: Value to return if operation fails
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Result of the operation or default value if it fails

    Raises:
        RuleError: If the operation fails and no default value is provided
    """
    try:
        # Record start time
        start_time = time.time()

        # Execute operation
        result = operation()

        # Record execution time
        execution_time = time.time() - start_time
        logger.debug(
            f"Rule operation completed in {execution_time:.4f}s",
            extra={"execution_time": execution_time, "component": component_name},
        )

        return result
    except Exception as e:
        # Handle error
        error_metadata = handle_error(
            e,
            component_name,
            log_level=log_level,
            include_traceback=include_traceback,
            additional_metadata=additional_metadata,
        )

        # If default value is provided, return it
        if default_value is not None:
            return default_value

        # Otherwise, raise RuleError
        if isinstance(e, RuleError):
            raise
        else:
            raise RuleError(f"Rule operation failed: {str(e)}", metadata=error_metadata) from e


def create_rule_error_result(
    error: Exception,
    component_name: str,
    log_level: str = "error",
    include_traceback: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized error result for a rule operation.

    This function creates a standardized error result for a rule operation,
    including error details, metadata, and logging.

    Args:
        error: The exception that occurred
        component_name: Name of the rule where the error occurred
        log_level: Log level to use for errors
        include_traceback: Whether to include traceback in error metadata
        additional_metadata: Additional metadata to include in error

    Returns:
        Standardized error result dictionary
    """
    # Handle error
    error_metadata = handle_error(
        error,
        component_name,
        log_level=log_level,
        include_traceback=include_traceback,
        additional_metadata=additional_metadata,
    )

    # Create error result
    return {
        "success": False,
        "error": str(error),
        "error_type": type(error).__name__,
        "component": component_name,
        "metadata": error_metadata,
    }


def safely_execute_component(
    operation: Callable[[], T],
    component_name: Optional[str] = None,
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
        start_time = time.time()

        # Execute operation
        result = operation()

        # Record execution time
        execution_time = time.time() - start_time
        logger.debug(
            f"{component_type.capitalize()} operation completed in {execution_time:.4f}s",
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
                f"{component_type.capitalize()} operation failed: {str(e)}",
                **kwargs,
            ) from e
