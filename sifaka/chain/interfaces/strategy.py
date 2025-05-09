"""
Strategy Interface Module

Protocol interfaces for Sifaka's strategy system.

## Overview
This module defines the core protocol interfaces for strategies in the Sifaka
framework. These interfaces establish a common contract for strategy behavior,
enabling better modularity, extensibility, and interoperability between
different strategy implementations.

## Components
1. **RetryStrategyProtocol**: Base retry strategy interface
   - Operation execution
   - Retry condition handling
   - Error management
   - Resource handling

## Usage Examples
```python
from typing import Any, Optional
from sifaka.chain.interfaces.strategy import RetryStrategyProtocol

class SimpleRetryStrategy(RetryStrategyProtocol):
    def __init__(self, max_attempts: int = 3):
        self._max_attempts = max_attempts

    def execute(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while attempt < self._max_attempts:
            try:
                result = operation(*args, **kwargs)
                if not self.should_retry(attempt, result):
                    return result
            except Exception as e:
                if not self.should_retry(attempt, None, e):
                    raise RuntimeError(f"Operation failed after {attempt + 1} attempts") from e
            attempt += 1
        raise RuntimeError(f"Operation failed after {self._max_attempts} attempts")

    def should_retry(self, attempt: int, result: Any, error: Optional[Exception] = None) -> bool:
        if attempt >= self._max_attempts - 1:
            return False
        return error is not None or result is None

# Use the strategy
strategy = SimpleRetryStrategy(max_attempts=3)
try:
    result = strategy.execute(some_operation, arg1, arg2, kwarg1="value")
except RuntimeError as e:
    print(f"Operation failed: {e}")
```

## Error Handling
- RuntimeError: Raised when operation fails after all retries
- Exception: Base class for operation-specific errors
- ValueError: Raised for invalid retry conditions

## Configuration
- max_attempts: Maximum number of retry attempts
- retry_conditions: Conditions for retrying operations
- error_handling: How to handle different error types
"""

from abc import abstractmethod
from typing import Any, Optional, Protocol, TypeVar, runtime_checkable

# Type variables
T = TypeVar("T")


@runtime_checkable
class RetryStrategyProtocol(Protocol):
    """
    Interface for retry strategies.

    Detailed description of what the class does, including:
    - Defines the contract for components that manage retry logic
    - Ensures consistent retry behavior across different implementations
    - Handles operation execution with retry logic
    - Manages retry conditions and error handling

    Example:
        ```python
        class SimpleRetryStrategy(RetryStrategyProtocol):
            def __init__(self, max_attempts: int = 3):
                self._max_attempts = max_attempts

            def execute(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
                attempt = 0
                while attempt < self._max_attempts:
                    try:
                        result = operation(*args, **kwargs)
                        if not self.should_retry(attempt, result):
                            return result
                    except Exception as e:
                        if not self.should_retry(attempt, None, e):
                            raise RuntimeError("Operation failed") from e
                    attempt += 1
                raise RuntimeError("Max attempts reached")

            def should_retry(self, attempt: int, result: Any, error: Optional[Exception] = None) -> bool:
                return attempt < self._max_attempts - 1 and (error is not None or result is None)
        ```
    """

    @abstractmethod
    def execute(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Execute an operation with retries.

        Detailed description of what the method does, including:
        - Executes an operation with retry logic
        - Handles operation failures and retries
        - Manages retry attempts and conditions
        - Returns operation result or raises error

        Args:
            operation: The operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation

        Raises:
            RuntimeError: If the operation fails after all retries

        Example:
            ```python
            # Execute an operation with retries
            try:
                result = strategy.execute(
                    operation=some_function,
                    arg1="value1",
                    arg2="value2"
                )
                print(f"Operation result: {result}")
            except RuntimeError as e:
                print(f"Operation failed: {e}")
            ```
        """
        pass

    @abstractmethod
    def should_retry(self, attempt: int, result: Any, error: Optional[Exception] = None) -> bool:
        """
        Check if an operation should be retried.

        Detailed description of what the method does, including:
        - Determines whether an operation should be retried
        - Evaluates retry conditions based on attempt, result, and error
        - Implements retry decision logic
        - Returns boolean indicating retry decision

        Args:
            attempt: The current attempt number
            result: The result of the operation
            error: The error that occurred, if any

        Returns:
            True if the operation should be retried, False otherwise

        Example:
            ```python
            # Check if operation should be retried
            should_retry = strategy.should_retry(
                attempt=1,
                result=None,
                error=ValueError("Invalid input")
            )
            print(f"Should retry: {should_retry}")
            ```
        """
        pass
