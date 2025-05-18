"""
Retry strategies for Sifaka.

This module provides various retry strategies for handling transient errors.
"""

import random
from typing import List, Optional, Type, Union

from sifaka.retry.base import RetryStrategy


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff retry strategy.

    This strategy increases the delay exponentially with each retry attempt.

    Attributes:
        base_delay: The base delay in seconds.
        max_delay: The maximum delay in seconds.
        jitter: Whether to add jitter to the delay.
        retryable_exceptions: The exception types to retry on.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Union[Type[Exception], List[Type[Exception]]]] = None,
    ):
        """Initialize the exponential backoff strategy.

        Args:
            base_delay: The base delay in seconds.
            max_delay: The maximum delay in seconds.
            jitter: Whether to add jitter to the delay.
            retryable_exceptions: The exception types to retry on.
                If None, all exceptions are retried.
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

        if retryable_exceptions is None:
            self.retryable_exceptions = [Exception]
        elif not isinstance(retryable_exceptions, list):
            self.retryable_exceptions = [retryable_exceptions]
        else:
            self.retryable_exceptions = retryable_exceptions

    def get_delay(self, attempt: int) -> float:
        """Get the delay before the next retry attempt.

        Args:
            attempt: The current attempt number (1-based).

        Returns:
            The delay in seconds before the next retry attempt.
        """
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)

        if self.jitter:
            # Add jitter to avoid thundering herd problem
            delay = delay * (0.5 + random.random())

        return float(delay)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine whether to retry after an exception.

        Args:
            attempt: The current attempt number (1-based).
            exception: The exception that was raised.

        Returns:
            True if the operation should be retried, False otherwise.
        """
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)


class FixedInterval(RetryStrategy):
    """Fixed interval retry strategy.

    This strategy uses a fixed delay between retry attempts.

    Attributes:
        delay: The delay in seconds.
        jitter: Whether to add jitter to the delay.
        retryable_exceptions: The exception types to retry on.
    """

    def __init__(
        self,
        delay: float = 1.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Union[Type[Exception], List[Type[Exception]]]] = None,
    ):
        """Initialize the fixed interval strategy.

        Args:
            delay: The delay in seconds.
            jitter: Whether to add jitter to the delay.
            retryable_exceptions: The exception types to retry on.
                If None, all exceptions are retried.
        """
        self.delay = delay
        self.jitter = jitter

        if retryable_exceptions is None:
            self.retryable_exceptions = [Exception]
        elif not isinstance(retryable_exceptions, list):
            self.retryable_exceptions = [retryable_exceptions]
        else:
            self.retryable_exceptions = retryable_exceptions

    def get_delay(self, attempt: int) -> float:
        """Get the delay before the next retry attempt.

        Args:
            attempt: The current attempt number (1-based).

        Returns:
            The delay in seconds before the next retry attempt.
        """
        if self.jitter:
            # Add jitter to avoid thundering herd problem
            return float(self.delay * (0.5 + random.random()))

        return float(self.delay)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine whether to retry after an exception.

        Args:
            attempt: The current attempt number (1-based).
            exception: The exception that was raised.

        Returns:
            True if the operation should be retried, False otherwise.
        """
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)


class LinearBackoff(RetryStrategy):
    """Linear backoff retry strategy.

    This strategy increases the delay linearly with each retry attempt.

    Attributes:
        base_delay: The base delay in seconds.
        increment: The increment in seconds for each retry.
        max_delay: The maximum delay in seconds.
        jitter: Whether to add jitter to the delay.
        retryable_exceptions: The exception types to retry on.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        increment: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Union[Type[Exception], List[Type[Exception]]]] = None,
    ):
        """Initialize the linear backoff strategy.

        Args:
            base_delay: The base delay in seconds.
            increment: The increment in seconds for each retry.
            max_delay: The maximum delay in seconds.
            jitter: Whether to add jitter to the delay.
            retryable_exceptions: The exception types to retry on.
                If None, all exceptions are retried.
        """
        self.base_delay = base_delay
        self.increment = increment
        self.max_delay = max_delay
        self.jitter = jitter

        if retryable_exceptions is None:
            self.retryable_exceptions = [Exception]
        elif not isinstance(retryable_exceptions, list):
            self.retryable_exceptions = [retryable_exceptions]
        else:
            self.retryable_exceptions = retryable_exceptions

    def get_delay(self, attempt: int) -> float:
        """Get the delay before the next retry attempt.

        Args:
            attempt: The current attempt number (1-based).

        Returns:
            The delay in seconds before the next retry attempt.
        """
        delay = min(self.base_delay + self.increment * (attempt - 1), self.max_delay)

        if self.jitter:
            # Add jitter to avoid thundering herd problem
            delay = delay * (0.5 + random.random())

        return float(delay)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine whether to retry after an exception.

        Args:
            attempt: The current attempt number (1-based).
            exception: The exception that was raised.

        Returns:
            True if the operation should be retried, False otherwise.
        """
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
