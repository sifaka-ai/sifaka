"""Simplified retry logic for Sifaka."""

import asyncio
from typing import TypeVar, Callable, Optional
from functools import wraps

from .exceptions import ModelProviderError

T = TypeVar('T')


class RetryConfig:
    """Simple retry configuration."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0
    ):
        """Initialize retry config.
        
        Args:
            max_attempts: Maximum number of attempts
            delay: Initial delay between retries
            backoff: Multiplier for delay after each retry
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry logic to async functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
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
            
            raise last_error
        
        return wrapper
    return decorator


# Preset configurations
RETRY_QUICK = RetryConfig(max_attempts=2, delay=0.5, backoff=1.5)
RETRY_STANDARD = RetryConfig(max_attempts=3, delay=1.0, backoff=2.0)
RETRY_PERSISTENT = RetryConfig(max_attempts=5, delay=1.0, backoff=2.0)