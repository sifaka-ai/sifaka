"""
Utility functions for the ModelProviderCore class.

This module provides utility functions for a ModelProviderCore instance,
including statistics tracking and helper methods.
"""

from typing import Any, Dict, TYPE_CHECKING
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from .provider import ModelProviderCore
logger = get_logger(__name__)


def update_statistics(
    provider: "ModelProviderCore", result: str, processing_time_ms: float
) -> None:
    """
    Update statistics after generation.

    This function updates the generation statistics in a ModelProviderCore
    instance's state manager, tracking metrics like generation count and
    total processing time.

    Args:
        provider: The model provider instance
        result: The generated text (used for potential future metrics like token count)
        processing_time_ms: The processing time in milliseconds
    """
    generation_stats = provider._state_manager.get("generation_stats", {})
    generation_stats["generation_count"] = generation_stats.get("generation_count", 0) + 1
    generation_stats["total_generation_time_ms"] = (
        generation_stats.get("total_generation_time_ms", 0) + processing_time_ms
    )
    provider._state_manager.update("generation_stats", generation_stats)


def update_token_count_statistics(
    provider: "ModelProviderCore", token_count: int, processing_time_ms: float
) -> None:
    """
    Update statistics after token counting.

    This function updates the token count statistics in a ModelProviderCore
    instance's state manager, tracking metrics like total tokens counted and
    processing time.

    Args:
        provider: The model provider instance
        token_count: The number of tokens counted
        processing_time_ms: The processing time in milliseconds
    """
    count_stats = provider._state_manager.get("token_count_stats", {})
    count_stats["total_tokens_counted"] = count_stats.get("total_tokens_counted", 0) + token_count
    count_stats["count_operations"] = count_stats.get("count_operations", 0) + 1
    count_stats["total_processing_time_ms"] = (
        count_stats.get("total_processing_time_ms", 0) + processing_time_ms
    )
    provider._state_manager.update("token_count_stats", count_stats)


def get_generation_stats(provider: "ModelProviderCore") -> Any:
    """
    Get generation statistics.

    This function retrieves generation statistics for a ModelProviderCore
    instance, including generation count and average processing time.

    Args:
        provider: The model provider instance

    Returns:
        A dictionary containing generation statistics
    """
    generation_stats = provider._state_manager.get("generation_stats", {})
    generation_count = generation_stats.get("generation_count", 0) if generation_stats else 0
    total_time_ms = generation_stats.get("total_generation_time_ms", 0) if generation_stats else 0
    avg_time_ms = total_time_ms / generation_count if generation_count > 0 else 0
    return {
        "generation_count": generation_count,
        "avg_generation_time_ms": avg_time_ms,
        "total_generation_time_ms": total_time_ms,
    }


def get_component_info(provider: "ModelProviderCore") -> Any:
    """
    Get component information.

    This function retrieves information about a ModelProviderCore instance,
    including model name, configuration, and initialization status.

    Args:
        provider: The model provider instance

    Returns:
        A dictionary containing component information
    """
    return {
        "name": provider.name,
        "model_name": provider.model_name,
        "config": (
            provider.config.model_dump()
            if hasattr(provider.config, "model_dump")
            else str(provider.config)
        ),
        "initialized": provider._state_manager.get("initialized", False),
        "creation_time": provider._state_manager.get_metadata("creation_time"),
        "warm_up_time": provider._state_manager.get_metadata("warm_up_time"),
    }
