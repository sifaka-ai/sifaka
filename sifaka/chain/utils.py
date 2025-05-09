"""
Chain Utilities Module

A collection of utility functions for Sifaka's chain system.

## Overview
This module provides utility functions for chains in the Sifaka framework,
including error handling, result creation, and chain helpers. These utilities
help standardize common operations and error handling patterns across the
chain system.

## Components
1. **create_chain_result**: Creates standardized chain results
2. **create_error_result**: Creates standardized error results
3. **try_chain_operation**: Executes chain operations with error handling

## Usage Examples
```python
from sifaka.chain.utils import create_chain_result, create_error_result, try_chain_operation

# Create a chain result
result = create_chain_result(
    output="Generated text output",
    rule_results=[
        {"rule_name": "length_rule", "passed": True, "details": {"length": 100}}
    ],
    critique_details={
        "feedback": "Good text, but could be more concise",
        "suggestions": ["Remove redundant phrases", "Use active voice"]
    },
    attempts=2,
    metadata={"model": "gpt-3.5-turbo", "temperature": 0.7}
)

# Create an error result
error_result = create_error_result(
    error=ValueError("Invalid input"),
    chain_name="my_chain",
    log_level="error"
)

# Execute a chain operation with error handling
def run_chain(prompt: str) -> ChainResult[str]:
    # Chain logic
    output = model.generate(prompt)
    return create_chain_result(
        output=output,
        rule_results=[],
        attempts=1,
    )

result = try_chain_operation(
    lambda: run_chain(prompt),
    chain_name="my_chain",
    log_level="error",
    default_result=None
)
```

## Error Handling
- ChainError: Raised when chain execution fails
- ValueError: Raised when input validation fails
- TypeError: Raised when type validation fails

## Configuration
- create_chain_result:
  - output: The generated output
  - rule_results: Results of rule validations
  - critique_details: Details about the critique
  - attempts: Number of attempts made
  - metadata: Additional metadata

- create_error_result:
  - error: The exception that occurred
  - chain_name: Name of the chain
  - log_level: Log level to use

- try_chain_operation:
  - operation_func: The operation function
  - chain_name: Name of the chain
  - log_level: Log level to use
  - default_result: Default result if operation fails
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

    Detailed description of what the function does, including:
    - Creates a standardized ChainResult object
    - Ensures consistent structure across all chain results
    - Handles optional parameters with defaults

    Args:
        output (T): The generated output
        rule_results (Optional[List[Any]]): Results of rule validations
        critique_details (Optional[Dict[str, Any]]): Details about the critique if available
        attempts (int): Number of attempts made
        metadata (Optional[Dict[str, Any]]): Additional metadata

    Returns:
        ChainResult[T]: Standardized chain result

    Raises:
        ValueError: When input validation fails
        TypeError: When type validation fails

    Example:
        ```python
        # Create a basic chain result
        result = create_chain_result(
            output="Generated text output",
            rule_results=[
                {"rule_name": "length_rule", "passed": True, "details": {"length": 100}}
            ],
            attempts=1
        )

        # Create a result with critique details
        result_with_critique = create_chain_result(
            output="Generated text output",
            rule_results=[],
            critique_details={
                "feedback": "Good text, but could be more concise",
                "suggestions": ["Remove redundant phrases", "Use active voice"]
            },
            attempts=2,
            metadata={"model": "gpt-3.5-turbo", "temperature": 0.7}
        )
        ```
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

    Detailed description of what the function does, including:
    - Creates a standardized result for a chain error
    - Includes error type, message, and metadata
    - Ensures consistent error handling across the chain system

    Args:
        error (Exception): The exception that occurred
        chain_name (str): Name of the chain
        log_level (str): Log level to use

    Returns:
        ErrorResult: Standardized error result

    Raises:
        ValueError: When input validation fails
        TypeError: When type validation fails

    Example:
        ```python
        # Create an error result
        error_result = create_error_result(
            error=ValueError("Invalid input"),
            chain_name="my_chain",
            log_level="error"
        )
        ```
    """
    return handle_chain_error(error, chain_name, log_level)


def try_chain_operation(
    operation_func: Callable[[], T],
    chain_name: str,
    log_level: str = "error",
    default_result: Optional[T] = None,
) -> T:
    """
    Execute a chain operation with error handling.

    Detailed description of what the function does, including:
    - Executes a chain operation with standardized error handling
    - Returns default result if operation fails
    - Logs errors with specified log level

    Args:
        operation_func (Callable[[], T]): The operation function to execute
        chain_name (str): Name of the chain
        log_level (str): Log level to use
        default_result (Optional[T]): Default result if operation fails

    Returns:
        T: Result of the operation or default result if operation fails

    Raises:
        ChainError: When chain execution fails
        ValueError: When input validation fails
        TypeError: When type validation fails

    Example:
        ```python
        # Execute a chain operation with error handling
        def run_chain(prompt: str) -> ChainResult[str]:
            # Chain logic
            output = model.generate(prompt)
            return create_chain_result(
                output=output,
                rule_results=[],
                attempts=1,
            )

        result = try_chain_operation(
            lambda: run_chain(prompt),
            chain_name="my_chain",
            log_level="error",
            default_result=None
        )
        ```
    """
    return try_operation(operation_func, chain_name, log_level, default_result)


__all__ = [
    "create_chain_result",
    "create_error_result",
    "try_chain_operation",
]
