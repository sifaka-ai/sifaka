"""Retry logic with exponential backoff for Sifaka."""

import asyncio
import random
from typing import TypeVar, Callable, Optional, Type, Tuple, Any
from functools import wraps
import logging

from .exceptions import ModelProviderError, TimeoutError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (
            ModelProviderError,
            asyncio.TimeoutError,
            ConnectionError,
        )


def calculate_backoff(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """Calculate exponential backoff with optional jitter."""
    delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
    
    if jitter:
        # Add randomness to prevent thundering herd
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> T:
    """Retry an async function with exponential backoff."""
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts:
                logger.error(
                    f"Failed after {config.max_attempts} attempts: {str(e)}"
                )
                raise
            
            delay = calculate_backoff(
                attempt,
                config.initial_delay,
                config.max_delay,
                config.exponential_base,
                config.jitter
            )
            
            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed: {str(e)}. "
                f"Retrying in {delay:.2f}s..."
            )
            
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error")


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to async functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


class RetryableMixin:
    """Mixin to add retry capability to classes."""
    
    def __init__(self, *args, retry_config: Optional[RetryConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_config = retry_config or RetryConfig()
    
    async def _retry_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Retry an async method with the configured settings."""
        return await retry_async(func, *args, config=self.retry_config, **kwargs)