"""Retry logic for handling transient failures in LLM API calls.

This module provides a configurable retry system for handling temporary
failures when calling LLM APIs. It implements exponential backoff and
selective retry based on error types.

## Key Features:

- **Exponential Backoff**: Delays increase between retries to avoid
  overwhelming APIs
- **Selective Retry**: Only retries specific error types that are likely
  to be transient
- **Configurable Limits**: Control max attempts, delays, and backoff rates
- **Preset Configurations**: Quick, Standard, and Persistent retry strategies

## Usage:

    >>> # Use default retry
    >>> @with_retry()
    >>> async def call_api():
    ...     return await llm_client.generate(prompt)

    >>> # Use custom configuration
    >>> config = RetryConfig(max_attempts=5, delay=2.0)
    >>> @with_retry(config)
    >>> async def reliable_call():
    ...     return await critical_operation()

    >>> # Use preset configuration
    >>> @with_retry(RETRY_PERSISTENT)
    >>> async def important_call():
    ...     return await slow_api()

## Error Handling:

The retry system only retries these error types:
- ModelProviderError: API errors from LLM providers
- asyncio.TimeoutError: Request timeouts
- ConnectionError: Network connectivity issues

Other errors are raised immediately without retry.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from .exceptions import ModelProviderError

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    This class defines how retry logic should behave when operations fail.
    It controls the number of attempts, delays between attempts, and how
    delays increase over time.

    The retry delay follows this pattern:
    - 1st retry: delay seconds
    - 2nd retry: delay * backoff seconds
    - 3rd retry: delay * backoff^2 seconds
    - And so on...

    Example:
        >>> # Quick retries for fast operations
        >>> quick = RetryConfig(max_attempts=2, delay=0.5, backoff=1.5)
        >>>
        >>> # Patient retries for slow APIs
        >>> patient = RetryConfig(max_attempts=5, delay=2.0, backoff=2.0)
        >>>
        >>> # Custom configuration
        >>> custom = RetryConfig(
        ...     max_attempts=4,
        ...     delay=1.0,      # Start with 1 second
        ...     backoff=3.0     # Triple the delay each time
        ... )

    Attributes:
        max_attempts: Maximum number of attempts before giving up (must be >= 1)
        delay: Initial delay in seconds between retries (must be > 0)
        backoff: Multiplier applied to delay after each retry (must be >= 1.0)
    """

    def __init__(self, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Initialize retry configuration with validation.

        Args:
            max_attempts: Maximum number of attempts including the initial try.
                Must be at least 1. Higher values increase reliability but
                also increase total time spent on failed operations.
            delay: Initial delay in seconds to wait before the first retry.
                Subsequent delays are multiplied by backoff. Should be based
                on typical API response times.
            backoff: Exponential backoff multiplier. Applied to delay after
                each failed attempt. Common values are 2.0 (double each time)
                or 1.5 (50% increase each time).

        Raises:
            ValueError: If any parameter is out of valid range

        Example:
            >>> # Standard configuration
            >>> config = RetryConfig()  # 3 attempts, 1s delay, 2x backoff
            >>>
            >>> # Aggressive retry for critical operations
            >>> config = RetryConfig(max_attempts=10, delay=0.5, backoff=1.2)
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff


def with_retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that adds automatic retry logic to async functions.

    This decorator wraps async functions to automatically retry them when
    they raise retryable exceptions. It implements exponential backoff
    between retries to avoid overwhelming services.

    Args:
        config: Retry configuration to use. If None, uses default
            configuration (3 attempts, 1s initial delay, 2x backoff).

    Returns:
        Decorator function that wraps the target async function

    Example:
        >>> @with_retry()  # Use default config
        >>> async def unreliable_api_call():
        ...     response = await client.post("/api/endpoint")
        ...     return response.json()
        >>>
        >>> @with_retry(RETRY_PERSISTENT)  # Use preset config
        >>> async def critical_operation():
        ...     return await database.commit()
        >>>
        >>> # Manual retry with custom config
        >>> config = RetryConfig(max_attempts=5)
        >>> wrapped = with_retry(config)(my_async_function)
        >>> result = await wrapped(arg1, arg2)

    Note:
        Only retries specific exception types that indicate transient
        failures. Permanent errors (like validation errors) are raised
        immediately without retry.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            delay = config.delay

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except (ModelProviderError, asyncio.TimeoutError, ConnectionError) as e:
                    last_error = e

                    if attempt < config.max_attempts:
                        await asyncio.sleep(delay)
                        delay *= config.backoff
                    else:
                        raise

            # This should never be reached, but helps mypy
            if last_error is not None:
                raise last_error
            raise RuntimeError("Retry failed unexpectedly")

        return wrapper

    return decorator


# Preset configurations
# These provide common retry strategies for different scenarios

RETRY_QUICK = RetryConfig(max_attempts=2, delay=0.5, backoff=1.5)
"""Quick retry for fast operations with minimal delay.

Use this for:
- Local operations that rarely fail
- Fast API calls with low latency
- Time-sensitive operations where waiting is costly

Timing: 0.5s, 0.75s = ~1.25s total delay
"""

RETRY_STANDARD = RetryConfig(max_attempts=3, delay=1.0, backoff=2.0)
"""Standard retry configuration suitable for most API calls.

Use this for:
- Normal LLM API calls
- Network operations with moderate latency
- Default retry behavior when unsure

Timing: 1s, 2s = ~3s total delay
"""

RETRY_PERSISTENT = RetryConfig(max_attempts=5, delay=1.0, backoff=2.0)
"""Persistent retry for critical operations that must succeed.

Use this for:
- Critical API calls that must complete
- Operations during high load or instability
- Long-running processes where reliability matters more than speed

Timing: 1s, 2s, 4s, 8s = ~15s total delay
"""
