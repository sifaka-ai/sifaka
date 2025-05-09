"""
Utility functions for chains in Sifaka.

This module provides utility functions for chains in the Sifaka framework,
including error handling, result creation, and chain helpers.
"""

from typing import Any, Dict, List, Optional, TypeVar, Callable, cast, Union, Generic

from pydantic import BaseModel

from .result import ChainResult
from sifaka.utils.errors import ChainError, try_operation
from sifaka.utils.error_patterns import handle_chain_error, ErrorResult

# Type variable for return type
T = TypeVar("T")


def create_chain_result(
    output: T,
    rule_results: Optional[List[Any]] = None,
    critique_details: Optional[Dict[str, Any]] = None,
    attempts: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> ChainResult[T]:
    """
    Create a chain result with standardized structure.

    Args:
        output: The generated output
        rule_results: Results of rule validations
        critique_details: Details about the critique if available
        attempts: Number of attempts made
        metadata: Additional metadata

    Returns:
        Standardized chain result
    """
    return ChainResult(
        output=output,
        rule_results=rule_results or [],
        critique_details=critique_details,
        attempts=attempts,
        metadata=metadata or {},
    )


def create_error_result(
    error: Exception,
    chain_name: str,
    log_level: str = "error",
) -> ErrorResult:
    """
    Create a result for a chain error.

    This function creates a standardized result for a chain error,
    including error type, message, and metadata.

    Args:
        error: The exception that occurred
        chain_name: Name of the chain where the error occurred
        log_level: Log level to use (default: "error")

    Returns:
        Error result with error information
    """
    # Handle the error
    return handle_chain_error(
        error=error,
        chain_name=chain_name,
        log_level=log_level,
    )


def try_chain_operation(
    operation_func: Callable[[], T],
    chain_name: str,
    log_level: str = "error",
    default_result: Optional[T] = None,
) -> T:
    """
    Execute a chain operation with standardized error handling.

    This function executes a chain operation and handles any errors
    that occur, providing standardized error handling and logging.

    Args:
        operation_func: The operation function to execute
        chain_name: Name of the chain executing the operation
        log_level: Log level to use for errors
        default_result: Default result to return if operation fails

    Returns:
        Result of the operation or default result if it fails

    Examples:
        ```python
        from sifaka.chain.utils import try_chain_operation, create_chain_result

        def run_chain(prompt: str) -> ChainResult[str]:
            # Chain logic
            output = model.generate(prompt)
            return create_chain_result(
                output=output,
                rule_results=[],
                attempts=1,
            )

        # Use try_chain_operation to handle errors
        result = try_chain_operation(
            lambda: run_chain(prompt),
            chain_name="my_chain",
        )
        ```
    """
    # Use try_operation with default result
    return try_operation(
        operation=operation_func,
        component_name=f"Chain:{chain_name}",
        default_value=default_result,
        log_level=log_level,
    )


__all__ = [
    "create_chain_result",
    "create_error_result",
    "try_chain_operation",
]
