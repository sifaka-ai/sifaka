"""
Token counting functionality for the ModelProviderCore class.

This module provides functions for counting tokens using a ModelProviderCore
instance, including token counter management and statistics tracking.
"""

from typing import TYPE_CHECKING

from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from .provider import ModelProviderCore

logger = get_logger(__name__)


def count_tokens_impl(provider: 'ModelProviderCore', text: str) -> int:
    """
    Implement token counting logic.

    This function implements the token counting logic for a ModelProviderCore
    instance, delegating to the token counter manager and tracing manager.

    Args:
        provider: The model provider instance
        text: The text to count tokens for

    Returns:
        The number of tokens in the text

    Raises:
        TypeError: If text is not a string
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Get token counter manager from state
    token_counter_manager = provider._state_manager.get("token_counter_manager")
    token_count = token_counter_manager.count_tokens(text)

    # Get tracing manager from state
    tracing_manager = provider._state_manager.get("tracing_manager")
    tracing_manager.trace_event(
        "token_count",
        {
            "text_length": len(text),
            "token_count": token_count,
        },
    )

    return token_count


def get_token_counter_manager(provider: 'ModelProviderCore'):
    """
    Get the token counter manager from the provider's state.

    This function retrieves the token counter manager from a ModelProviderCore
    instance's state, ensuring that the provider is initialized first.

    Args:
        provider: The model provider instance

    Returns:
        The token counter manager

    Raises:
        RuntimeError: If the token counter manager is not found
    """
    # Ensure component is initialized
    if not provider._state_manager.get("initialized", False):
        provider.warm_up()

    # Get token counter manager from state
    token_counter_manager = provider._state_manager.get("token_counter_manager")
    if not token_counter_manager:
        raise RuntimeError("Token counter manager not found")

    return token_counter_manager


def get_token_count_stats(provider: 'ModelProviderCore'):
    """
    Get token count statistics.

    This function retrieves token count statistics for a ModelProviderCore
    instance, including total tokens counted and average processing time.

    Args:
        provider: The model provider instance

    Returns:
        A dictionary containing token count statistics
    """
    count_stats = provider._state_manager.get("token_count_stats", {})
    total_tokens = count_stats.get("total_tokens_counted", 0)
    count_operations = count_stats.get("count_operations", 0)
    total_time_ms = count_stats.get("total_processing_time_ms", 0)

    avg_time_ms = total_time_ms / count_operations if count_operations > 0 else 0
    avg_tokens = total_tokens / count_operations if count_operations > 0 else 0

    return {
        "total_tokens_counted": total_tokens,
        "count_operations": count_operations,
        "avg_processing_time_ms": avg_time_ms,
        "avg_tokens_per_operation": avg_tokens,
    }
