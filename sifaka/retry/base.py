"""
Base retry functionality for Sifaka.

This module provides the base retry functionality for handling transient errors.
"""

import time
import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

from sifaka.errors import RetryError


# Type for the function to retry
T = TypeVar("T")
RetryFunction = Callable[..., T]

# Logger for retry operations
logger = logging.getLogger("sifaka.retry")


class RetryStrategy(ABC):
    """Abstract base class for retry strategies.

    This class defines the interface for retry strategies.
    """

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get the delay before the next retry attempt.

        Args:
            attempt: The current attempt number (1-based).

        Returns:
            The delay in seconds before the next retry attempt.
        """
        pass

    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine whether to retry after an exception.

        Args:
            attempt: The current attempt number (1-based).
            exception: The exception that was raised.

        Returns:
            True if the operation should be retried, False otherwise.
        """
        pass


def retry(
    strategy: RetryStrategy,
    exceptions: Optional[Union[Type[Exception], List[Type[Exception]]]] = None,
    max_attempts: int = 3,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[RetryFunction[T]], RetryFunction[T]]:
    """Decorator for retrying a function on failure.

    Args:
        strategy: The retry strategy to use.
        exceptions: The exception types to retry on. If None, all exceptions are retried.
        max_attempts: The maximum number of attempts to make.
        on_retry: A callback function to call before each retry attempt.
            The callback is passed the attempt number and the exception.

    Returns:
        A decorator function.
    """
    if exceptions is None:
        exceptions = Exception

    if not isinstance(exceptions, list):
        exceptions = [exceptions]

    def decorator(func: RetryFunction[T]) -> RetryFunction[T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    last_exception = e

                    if attempt == max_attempts or not strategy.should_retry(attempt, e):
                        break

                    delay = strategy.get_delay(attempt)

                    if on_retry:
                        on_retry(attempt, e)

                    logger.debug(
                        f"Retry attempt {attempt}/{max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s due to {type(e).__name__}: {str(e)}"
                    )

                    time.sleep(delay)

            if last_exception is not None:
                raise RetryError(
                    f"Failed after {max_attempts} attempts: {str(last_exception)}"
                ) from last_exception

            # This should never happen, but is needed for type checking
            raise RetryError("Failed with no exception")

        return wrapper

    return decorator
