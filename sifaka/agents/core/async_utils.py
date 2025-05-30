"""Async utilities for PydanticAI chain implementation.

This module provides simple async utilities for running sync functions
in thread pools to avoid blocking the event loop.
"""

import asyncio
import concurrent.futures
from typing import Any, Callable, TypeVar

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


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
