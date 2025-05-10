"""
Common Utility Functions for Sifaka

This module provides common utility functions used across different components
in the Sifaka framework. It centralizes frequently used patterns to reduce
code duplication and ensure consistency.

## Categories of Utilities

1. **State Access Patterns**: Common patterns for accessing and updating state
2. **Error Handling Patterns**: Common patterns for handling errors
3. **Result Creation Patterns**: Common patterns for creating results
4. **Validation Patterns**: Common patterns for validating inputs

## Usage Examples

```python
from sifaka.utils.common import (
    get_cached_result, update_cache, update_statistics,
    safely_execute, create_standard_result
)

# State access patterns
result = get_cached_result(state_manager, cache_key)
update_cache(state_manager, cache_key, result, max_size=100)
update_statistics(state_manager, execution_time, success=True)

# Error handling patterns
result = safely_execute(
    lambda: process_data(input_data),
    component_name="DataProcessor",
    component_type="Processor"
)

# Result creation patterns
result = create_standard_result(
    output="Generated text",
    metadata={"model": "gpt-4"},
    success=True
)
```
"""

from typing import Any, Dict, Optional, TypeVar, Callable
import time
import logging

from sifaka.utils.state import StateManager

# Type variable for return type
T = TypeVar("T")

# Get logger
logger = logging.getLogger(__name__)


# ===== State Access Patterns =====


def initialize_component_state(
    state_manager: StateManager, component_type: str, name: str, description: Optional[str] = None
) -> None:
    """
    Initialize standard component state.

    Args:
        state_manager: The state manager to initialize
        component_type: Type of the component (e.g., "Classifier", "Chain")
        name: Name of the component
        description: Optional description of the component
    """
    state_manager.update("initialized", False)
    state_manager.update("cache", {})
    state_manager.update("result_cache", {})
    state_manager.update("execution_count", 0)
    state_manager.update("success_count", 0)
    state_manager.update("error_count", 0)
    state_manager.update("total_execution_time_ms", 0)
    state_manager.update("errors", [])
    state_manager.update("cache_hits", 0)
    state_manager.set_metadata("component_type", component_type)
    state_manager.set_metadata("name", name)

    if description:
        state_manager.set_metadata("description", description)


def get_cached_result(
    state_manager: StateManager, cache_key: str, cache_name: str = "result_cache"
) -> Optional[Any]:
    """
    Get a cached result from the state manager.

    Args:
        state_manager: The state manager to use
        cache_key: The key to look up in the cache
        cache_name: The name of the cache in the state

    Returns:
        The cached result or None if not found
    """
    # Get the cache from state
    cache = state_manager.get(cache_name, {})

    # Check if we have a cached result
    if cache_key in cache:
        # Track cache hit
        cache_hits = state_manager.get("cache_hits", 0)
        state_manager.update("cache_hits", cache_hits + 1)
        state_manager.set_metadata("last_cache_hit", time.time())
        return cache[cache_key]

    return None


def update_cache(
    state_manager: StateManager,
    cache_key: str,
    result: Any,
    cache_name: str = "result_cache",
    max_size: int = 100,
) -> None:
    """
    Update the cache with a new result.

    Args:
        state_manager: The state manager to use
        cache_key: The key to store the result under
        result: The result to cache
        cache_name: The name of the cache in the state
        max_size: Maximum size of the cache
    """
    # Get the cache from state
    cache = state_manager.get(cache_name, {})

    # Clear cache if it's full
    if len(cache) >= max_size:
        cache.clear()

    # Update cache with new result
    cache[cache_key] = result
    state_manager.update(cache_name, cache)
    state_manager.set_metadata("last_cache_update", time.time())


def update_statistics(
    state_manager: StateManager,
    execution_time: float,
    success: bool = True,
    error: Optional[Exception] = None,
) -> None:
    """
    Update component statistics in the state manager.

    Args:
        state_manager: The state manager to use
        execution_time: Execution time in seconds
        success: Whether the operation was successful
        error: The error that occurred, if any
    """
    # Update execution count
    execution_count = state_manager.get("execution_count", 0)
    state_manager.update("execution_count", execution_count + 1)

    # Update success/error counts
    if success:
        success_count = state_manager.get("success_count", 0)
        state_manager.update("success_count", success_count + 1)
    else:
        error_count = state_manager.get("error_count", 0)
        state_manager.update("error_count", error_count + 1)

        # Track errors
        if error:
            errors = state_manager.get("errors", [])
            errors.append({"error": str(error), "type": type(error).__name__, "time": time.time()})
            state_manager.update("errors", errors)

    # Update average execution time
    total_time = state_manager.get("total_execution_time_ms", 0)
    state_manager.update("total_execution_time_ms", total_time + (execution_time * 1000))

    # Calculate and store average time
    avg_time = (total_time + (execution_time * 1000)) / execution_count
    state_manager.set_metadata("avg_execution_time_ms", avg_time)


def clear_component_statistics(state_manager: StateManager) -> None:
    """
    Clear component statistics in the state manager.

    Args:
        state_manager: The state manager to use
    """
    # Reset statistics
    state_manager.update("statistics", {})
    state_manager.update("cache_hits", 0)
    state_manager.update("execution_count", 0)
    state_manager.update("success_count", 0)
    state_manager.update("error_count", 0)
    state_manager.update("total_execution_time_ms", 0)
    state_manager.update("errors", [])


# ===== Error Handling Patterns =====


def record_error(
    state_manager: StateManager, error: Exception, include_traceback: bool = True
) -> Dict[str, Any]:
    """
    Record an error in the state manager.

    Args:
        state_manager: The state manager to use
        error: The exception to record
        include_traceback: Whether to include traceback in the error info

    Returns:
        Error information dictionary
    """
    import traceback

    # Create error info
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": time.time(),
    }

    # Add traceback if requested
    if include_traceback:
        error_info["traceback"] = traceback.format_exc()

    # Get existing errors
    errors = state_manager.get("errors", [])

    # Add new error
    errors.append(error_info)

    # Update errors in state
    state_manager.update("errors", errors)

    # Update error count
    error_count = state_manager.get("error_count", 0)
    state_manager.update("error_count", error_count + 1)

    # Set last error metadata
    state_manager.set_metadata("last_error", error_info)

    return error_info


def safely_execute(
    operation: Callable[[], T],
    component_name: str,
    state_manager: Optional[StateManager] = None,
    component_type: Optional[str] = None,
    default_value: Optional[T] = None,
    log_level: str = "error",
) -> T:
    """
    Execute an operation with standardized error handling.

    Args:
        operation: The operation to execute
        component_name: Name of the component executing the operation
        state_manager: Optional state manager to record errors
        component_type: Type of the component (e.g., "Chain", "Model")
        default_value: Value to return if operation fails
        log_level: Log level to use for errors

    Returns:
        Result of the operation or default value if it fails
    """
    try:
        return operation()
    except Exception as e:
        # Log the error
        if log_level == "error":
            logger.error(f"Error in {component_name}: {e}")
        elif log_level == "warning":
            logger.warning(f"Warning in {component_name}: {e}")
        elif log_level == "info":
            logger.info(f"Info in {component_name}: {e}")
        elif log_level == "debug":
            logger.debug(f"Debug in {component_name}: {e}")

        # Record error in state if state manager provided
        if state_manager:
            record_error(state_manager, e)

        # Return default value if provided
        if default_value is not None:
            return default_value

        # Re-raise the exception if no default value
        raise


# ===== Result Creation Patterns =====


def create_standard_result(
    output: Any,
    metadata: Optional[Dict[str, Any]] = None,
    success: bool = True,
    message: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a standardized result dictionary.

    Args:
        output: The output of the operation
        metadata: Additional metadata for the result
        success: Whether the operation was successful
        message: Optional message about the result
        processing_time_ms: Processing time in milliseconds

    Returns:
        A standardized result dictionary
    """
    result = {
        "output": output,
        "success": success,
        "metadata": metadata or {},
    }

    if message:
        result["message"] = message

    if processing_time_ms is not None:
        result["processing_time_ms"] = processing_time_ms
        result["metadata"]["processing_time_ms"] = processing_time_ms

    return result


# Export all functions
__all__ = [
    "initialize_component_state",
    "get_cached_result",
    "update_cache",
    "update_statistics",
    "clear_component_statistics",
    "record_error",
    "safely_execute",
    "create_standard_result",
]
