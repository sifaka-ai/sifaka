"""
Common Utility Functions for Sifaka

This module provides common utility functions used across different components
in the Sifaka framework. It centralizes frequently used patterns to reduce
code duplication and ensure consistency.

## Overview
The common utilities module provides standardized implementations of frequently
used patterns across the Sifaka framework. These utilities help ensure consistent
behavior, reduce code duplication, and simplify component implementation.

## Components
The module is organized into several categories of utilities:

1. **State Access Patterns**: Functions for accessing and updating component state
   - initialize_component_state: Initialize standard component state
   - get_cached_result: Retrieve cached results with tracking
   - update_cache: Update cache with size management
   - update_statistics: Track component performance metrics
   - clear_component_statistics: Reset component statistics

2. **Error Handling Patterns**: Functions for standardized error handling
   - record_error: Record errors in component state
   - safely_execute: Execute operations with standardized error handling

3. **Result Creation Patterns**: Functions for creating standardized results
   - create_standard_result: Create results with consistent structure

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

## Error Handling
The utilities provide standardized error handling patterns that ensure:
- Consistent error recording in component state
- Proper error logging with appropriate context
- Optional default values for graceful failure
- Traceback capture for debugging

## Configuration
The utilities support various configuration options:
- Cache size management
- Error logging levels
- Statistics tracking options
- Result formatting options
"""

from typing import Any, Dict, Optional, TypeVar, Callable
import time

from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

# Type variable for return type
T = TypeVar("T")

# Get logger
logger = get_logger(__name__)


# ===== State Access Patterns =====


def initialize_component_state(
    state_manager: StateManager,
    component_type: str,
    name: str,
    description: Optional[Optional[str]] = None,
) -> None:
    """
    Initialize standard component state.

    This function initializes a state manager with standard state fields and metadata
    that are common across all components in the Sifaka framework. It sets up state
    for tracking initialization status, caching, statistics, and errors.

    ## Architecture
    This function follows a standardized pattern for component state initialization,
    ensuring that all components have a consistent state structure. This makes it
    easier to implement components and ensures that utilities that operate on
    component state can rely on a consistent structure.

    Args:
        state_manager: The state manager to initialize
        component_type: Type of the component (e.g., "Classifier", "Chain")
        name: Name of the component
        description: Optional description of the component

    Example:
        ```python
        from sifaka.utils.state import create_manager_state
        from sifaka.utils.common import initialize_component_state

        # Create and initialize state manager
        state_manager = create_manager_state()
        initialize_component_state(
            state_manager,
            component_type="Classifier",
            name="toxicity_classifier",
            description="Classifies text for toxicity"
        )

        # State is now initialized with standard fields
        print(state_manager.get("initialized"))  # False
        print(state_manager.get("cache"))  # {}
        print(state_manager.get_metadata("component_type"))  # "Classifier"
        ```
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
    error: Optional[Optional[Exception]] = None,
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
    if execution_count > 0:
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
    state_manager: Optional[Optional[StateManager]] = None,
    component_type: Optional[Optional[str]] = None,
    default_value: Optional[Optional[T]] = None,
    log_level: str = "error",
) -> T:
    """
    Execute an operation with standardized error handling.

    This function provides a standardized way to execute operations with proper
    error handling, logging, and state tracking. It catches exceptions, logs them
    with appropriate context, records them in component state if a state manager
    is provided, and either returns a default value or re-raises the exception.

    ## Architecture
    This function implements a standardized error handling pattern that ensures
    consistent behavior across the framework. It separates the error handling
    logic from the business logic, making components cleaner and more focused.

    ## Error Handling
    The function handles errors by:
    1. Catching any exceptions that occur during operation execution
    2. Logging the error with the specified log level and component context
    3. Recording the error in component state if a state manager is provided
    4. Either returning a default value or re-raising the exception

    Args:
        operation: The operation to execute
        component_name: Name of the component executing the operation
        state_manager: Optional state manager to record errors
        component_type: Type of the component (e.g., "Chain", "Model")
        default_value: Value to return if operation fails
        log_level: Log level to use for errors ("error", "warning", "info", "debug")

    Returns:
        Result of the operation or default value if it fails

    Raises:
        Exception: Re-raises the original exception if no default value is provided

    Example:
        ```python
        from sifaka.utils.common import safely_execute

        # Execute with default value
        result = safely_execute(
            lambda: process_data(input_data),
            component_name="DataProcessor",
            default_value={"success": False, "error": "Processing failed"}
        )

        # Execute with state tracking
        result = safely_execute(
            lambda: process_data(input_data),
            component_name="DataProcessor",
            state_manager=self._state_manager,
            component_type="Processor",
            log_level="warning"
        )
        ```
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
    message: Optional[Optional[str]] = None,
    processing_time_ms: Optional[Optional[float]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized result dictionary.

    This function creates a standardized result dictionary with a consistent structure
    that can be used across all components in the Sifaka framework. It ensures that
    results have a consistent format, making it easier to process and interpret results
    from different components.

    ## Architecture
    This function implements a standardized result creation pattern that ensures
    consistent result structure across the framework. The standard result format
    includes the output, success status, optional message, and metadata.

    Args:
        output: The output of the operation
        metadata: Additional metadata for the result
        success: Whether the operation was successful
        message: Optional message about the result
        processing_time_ms: Processing time in milliseconds

    Returns:
        A standardized result dictionary with the following structure:
        ```
        {
            "output": Any,              # The operation output
            "success": bool,            # Whether the operation succeeded
            "message": str,             # Optional message (if provided)
            "processing_time_ms": float, # Processing time (if provided)
            "metadata": {               # Additional metadata
                "processing_time_ms": float, # Processing time (if provided)
                # Additional metadata fields
            )
        }
        ```

    Example:
        ```python
        from sifaka.utils.common import create_standard_result

        # Create a successful result
        result = create_standard_result(
            output="Generated text",
            metadata={"model": "gpt-4", "temperature": 0.7},
            success=True,
            message="Text generated successfully",
            processing_time_ms=150.5
        )

        # Create a failure result
        result = create_standard_result(
            output=None,
            metadata={"error_type": "ValidationError"},
            success=False,
            message="Failed to generate text: Invalid input"
        )
        ```
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
