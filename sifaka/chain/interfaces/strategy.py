"""
Strategy protocol interfaces for Sifaka.

This module defines the interfaces for strategies in the Sifaka framework.
These interfaces establish a common contract for strategy behavior, enabling better
modularity and extensibility.
"""

from abc import abstractmethod
from typing import Any, Optional, Protocol, TypeVar, runtime_checkable

# Type variables
T = TypeVar("T")


@runtime_checkable
class RetryStrategyProtocol(Protocol):
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
        pass

    @abstractmethod
    def should_retry(self, attempt: int, result: Any, error: Optional[Exception] = None) -> bool:
        """
        Check if an operation should be retried.

        Args:
            attempt: The current attempt number
            result: The result of the operation
            error: The error that occurred, if any

        Returns:
            True if the operation should be retried, False otherwise
        """
        pass
