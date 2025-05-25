"""Retry mechanisms with exponential backoff for Sifaka.

This module provides configurable retry mechanisms for handling transient failures
in external services. It supports various backoff strategies and retry conditions.

Example:
    ```python
    from sifaka.utils.retry import RetryConfig, retry_with_backoff

    # Configure retry behavior
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True
    )

    # Use as decorator
    @retry_with_backoff(config)
    def unreliable_function():
        # Your code here
        pass

    # Or use directly
    result = retry_with_backoff(config)(lambda: api_call())()
    ```
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Optional, Type, Union
import logging

from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry mechanisms."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_factor: float = 2.0  # Multiplier for exponential backoff
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True  # Add random jitter to prevent thundering herd
    jitter_range: float = 0.1  # Jitter range as fraction of delay

    # Exception handling
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)

    # Conditional retry
    retry_condition: Optional[Callable[[Exception], bool]] = None

    # Logging
    log_attempts: bool = True
    log_level: int = logging.WARNING


@dataclass
class RetryStats:
    """Statistics for retry attempts."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay: float = 0.0
    last_exception: Optional[Exception] = None
    attempt_times: List[float] = field(default_factory=list)


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""

    def __init__(self, original_exception: Exception, stats: RetryStats, config: RetryConfig):
        self.original_exception = original_exception
        self.stats = stats
        self.config = config

        message = (
            f"All {config.max_attempts} retry attempts failed. "
            f"Last exception: {type(original_exception).__name__}: {original_exception}"
        )
        super().__init__(message)


class RetryManager:
    """Manages retry logic with configurable strategies."""

    def __init__(self, config: RetryConfig):
        """Initialize retry manager.

        Args:
            config: Retry configuration.
        """
        self.config = config
        self.stats = RetryStats()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-based).

        Returns:
            Delay in seconds.
        """
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_factor**attempt)
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            # Fibonacci sequence for delays
            if attempt == 0:
                delay = self.config.base_delay
            elif attempt == 1:
                delay = self.config.base_delay
            else:
                # Calculate fibonacci number for delay
                a, b = 1, 1
                for _ in range(attempt - 1):
                    a, b = b, a + b
                delay = self.config.base_delay * b
        else:
            delay = self.config.base_delay  # type: ignore[unreachable]

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)

        return delay

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if we should retry given the exception and attempt number.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number (0-based).

        Returns:
            True if we should retry, False otherwise.
        """
        # Check if we've exceeded max attempts
        if attempt >= self.config.max_attempts:
            return False

        # Check non-retryable exceptions first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False

        # Check retryable exceptions
        is_retryable = False
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                is_retryable = True
                break

        if not is_retryable:
            return False

        # Check custom retry condition
        if self.config.retry_condition:
            return self.config.retry_condition(exception)

        return True

    def _log_attempt(
        self, attempt: int, exception: Exception, delay: Optional[float] = None
    ) -> None:
        """Log retry attempt.

        Args:
            attempt: Current attempt number (0-based).
            exception: The exception that occurred.
            delay: Delay before next attempt (if any).
        """
        if not self.config.log_attempts:
            return

        message = f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {type(exception).__name__}: {exception}"
        if delay is not None:
            message += f". Retrying in {delay:.2f}s"
        else:
            message += ". No more retries"

        logger.log(self.config.log_level, message)

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with retry logic.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function result.

        Raises:
            RetryError: If all attempts fail.
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            self.stats.total_attempts += 1
            attempt_start = time.time()

            try:
                result = func(*args, **kwargs)
                self.stats.successful_attempts += 1
                self.stats.attempt_times.append(time.time() - attempt_start)
                return result

            except Exception as e:
                last_exception = e
                self.stats.failed_attempts += 1
                self.stats.last_exception = e
                self.stats.attempt_times.append(time.time() - attempt_start)

                if not self._should_retry(e, attempt):
                    break

                # Calculate delay for next attempt
                if attempt < self.config.max_attempts - 1:  # Don't delay after last attempt
                    delay = self._calculate_delay(attempt)
                    self._log_attempt(attempt, e, delay)

                    time.sleep(delay)
                    self.stats.total_delay += delay
                else:
                    self._log_attempt(attempt, e)

        # All attempts failed
        raise RetryError(last_exception or Exception("Unknown error"), self.stats, self.config)

    async def _execute_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute async function with retry logic.

        Args:
            func: Async function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function result.

        Raises:
            RetryError: If all attempts fail.
        """
        last_exception = None

        for attempt in range(self.config.max_attempts):
            self.stats.total_attempts += 1
            attempt_start = time.time()

            try:
                result = await func(*args, **kwargs)
                self.stats.successful_attempts += 1
                self.stats.attempt_times.append(time.time() - attempt_start)
                return result

            except Exception as e:
                last_exception = e
                self.stats.failed_attempts += 1
                self.stats.last_exception = e
                self.stats.attempt_times.append(time.time() - attempt_start)

                if not self._should_retry(e, attempt):
                    break

                # Calculate delay for next attempt
                if attempt < self.config.max_attempts - 1:  # Don't delay after last attempt
                    delay = self._calculate_delay(attempt)
                    self._log_attempt(attempt, e, delay)

                    await asyncio.sleep(delay)
                    self.stats.total_delay += delay
                else:
                    self._log_attempt(attempt, e)

        # All attempts failed
        raise RetryError(last_exception or Exception("Unknown error"), self.stats, self.config)


def retry_with_backoff(config: RetryConfig) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for adding retry logic to functions.

    Args:
        config: Retry configuration.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = RetryManager(config)
            return manager.execute(func, *args, **kwargs)

        @wraps(func)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = RetryManager(config)
            return await manager._execute_async(func, *args, **kwargs)

        return _async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


# Predefined configurations for common scenarios
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_attempts=3, base_delay=1.0, max_delay=30.0, backoff_factor=2.0, jitter=True
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5, base_delay=0.5, max_delay=60.0, backoff_factor=2.0, jitter=True
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=2, base_delay=2.0, max_delay=10.0, backoff_factor=1.5, jitter=False
)

API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=True,
    retryable_exceptions=[ConnectionError, TimeoutError],
    non_retryable_exceptions=[ValueError, TypeError],
)
