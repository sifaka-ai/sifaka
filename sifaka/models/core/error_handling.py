"""
Error handling utilities for the ModelProviderCore class.

This module provides functions for handling errors in a ModelProviderCore
instance, including error recording and safe execution patterns.
"""

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from sifaka.utils.logging import get_logger

if TYPE_CHECKING:
    from .provider import ModelProviderCore
logger = get_logger(__name__)


def record_error(provider: "ModelProviderCore", error: Exception) -> None:
    """
    Record an error in the state manager.

    This function records an error in a ModelProviderCore instance's state
    manager, including error type, message, and traceback.

    Args:
        provider: The model provider instance
        error: The exception to record
    """
    from sifaka.utils.common import record_error as record_error_common

    record_error_common(
        error=error,
        state_manager=provider._state_manager,
    )


def safely_execute(
    provider: "ModelProviderCore",
    operation: Callable[[], Any],
    error_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Safely execute an operation with error handling.

    This function safely executes an operation with standardized error
    handling, recording any errors that occur.

    Args:
        provider: The model provider instance
        operation: The operation to execute
        error_context: Optional context information for error recording

    Returns:
        The result of the operation

    Raises:
        Exception: If the operation raises an exception
    """
    from sifaka.utils.errors.safe_execution import safely_execute_component_operation

    from sifaka.utils.errors.base import SifakaError

    return safely_execute_component_operation(
        operation=operation,
        component_name=provider.name,
        component_type=provider.__class__.__name__,
        error_class=SifakaError,
        additional_metadata=error_context or {},
    )


def get_error_stats(provider: "ModelProviderCore") -> Any:
    """
    Get error statistics.

    This function retrieves error statistics for a ModelProviderCore
    instance, including error count by type and total error count.

    Args:
        provider: The model provider instance

    Returns:
        A dictionary containing error statistics
    """
    error_stats = provider._state_manager.get("error_stats", {})
    error_count = error_stats.get("error_count", 0) if error_stats else 0
    error_types = error_stats.get("error_types", {}) if error_stats else {}
    return {"error_count": error_count, "error_types": error_types, "has_errors": error_count > 0}


def clear_error_stats(provider: "ModelProviderCore") -> None:
    """
    Clear error statistics.

    This function clears error statistics for a ModelProviderCore instance.

    Args:
        provider: The model provider instance
    """
    provider._state_manager.update(
        "error_stats", {"error_count": 0, "error_types": {}, "last_error": None}
    )
