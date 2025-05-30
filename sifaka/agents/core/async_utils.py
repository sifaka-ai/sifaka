"""Async utilities for PydanticAI chain implementation.

This module provides proper async utilities that replace the problematic
sync/async patterns in the original implementation.
"""

import asyncio
import concurrent.futures
from functools import wraps
from typing import Any, Callable, TypeVar

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def ensure_async_compatibility(func: Callable[..., T]) -> Callable[..., T]:
    """Ensure a function can be called in both sync and async contexts.

    This is a cleaner replacement for the problematic async_to_sync decorator.
    Instead of trying to bridge sync/async, we encourage pure async usage.

    Args:
        func: The async function to wrap.

    Returns:
        A function that can handle both sync and async contexts properly.
    """
    if not asyncio.iscoroutinefunction(func):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Check if we're in an async context
            asyncio.get_running_loop()
            # If we get here, we're in an async context
            # Return the coroutine directly - let the caller await it
            return func(*args, **kwargs)
        except RuntimeError:
            # No running event loop, we can safely use asyncio.run()
            return asyncio.run(func(*args, **kwargs))

    return wrapper


async def run_in_thread_pool(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a sync function in a thread pool to avoid blocking the event loop.

    Args:
        func: The sync function to run.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function call.
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


async def gather_with_error_handling(*awaitables, return_exceptions: bool = True) -> list[Any]:
    """Gather multiple awaitables with proper error handling and logging.

    Args:
        *awaitables: The awaitables to gather.
        return_exceptions: Whether to return exceptions instead of raising them.

    Returns:
        List of results, with exceptions if return_exceptions is True.
    """
    try:
        return await asyncio.gather(*awaitables, return_exceptions=return_exceptions)
    except Exception as e:
        logger.error(f"Error in gather operation: {e}")
        if return_exceptions:
            return [e] * len(awaitables)
        raise
