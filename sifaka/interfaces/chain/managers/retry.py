"""
Retry strategy interface for Sifaka.

This module defines the interface for retry strategies in the Sifaka framework.
These interfaces establish a common contract for retry strategy behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **RetryStrategy**: Interface for retry strategies

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class RetryStrategy(Protocol):
    """
    Interface for retry strategies.

    This interface defines the contract for components that manage retry logic.
    It ensures that retry strategies can execute operations with retries and
    handle retry conditions.

    ## Lifecycle

    1. **Initialization**: Set up retry strategy resources
    2. **Execution**: Execute operations with retries
    3. **Condition Handling**: Handle retry conditions
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an execute method to execute operations with retries
    - Handle retry conditions appropriately
    """

    @abstractmethod
    def execute(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Execute an operation with retries.

        Args:
            operation: The operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation

        Raises:
            RuntimeError: If the operation fails after all retries
        """

    @abstractmethod
    def should_retry(
        self, attempt: int, result: Any, error: Optional[Exception] = None
    ) -> bool:
        """
        Check if an operation should be retried.

        Args:
            attempt: The current attempt number
            result: The result of the operation
            error: The error that occurred, if any

        Returns:
            True if the operation should be retried, False otherwise
        """
