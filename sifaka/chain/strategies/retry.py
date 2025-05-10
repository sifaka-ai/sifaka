"""
Retry strategy for Sifaka chains.

This module provides the retry strategy for Sifaka chains,
enabling configurable retry behavior for failed operations.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
import time

from pydantic import BaseModel, PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class RetryStrategy(BaseComponent):
    """
    Retry strategy for Sifaka chains.

    This class provides configurable retry behavior for failed operations,
    with support for different retry policies and backoff strategies.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        max_backoff: float = 60.0,
        name: str = "retry_strategy",
        description: str = "Retry strategy for Sifaka chains",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the retry strategy.

        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor to increase delay between retries
            max_backoff: Maximum delay between retries in seconds
            name: Name of the strategy
            description: Description of the strategy
            config: Additional configuration
        """
        super().__init__()

        self._state.update("max_retries", max_retries)
        self._state.update("backoff_factor", backoff_factor)
        self._state.update("max_backoff", max_backoff)
        self._state.update("name", name)
        self._state.update("description", description)
        self._state.update("config", config or {})
        self._state.update("initialized", True)
        self._state.update("execution_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "retry_strategy")
        self._state.set_metadata("creation_time", time.time())

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Determine if an operation should be retried.

        Args:
            attempt: Current attempt number
            error: The error that occurred

        Returns:
            True if the operation should be retried, False otherwise
        """
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Record start time
        start_time = time.time()

        try:
            # Get configuration from state
            max_retries = self._state.get("max_retries")

            # Check if we should retry
            should_retry = attempt < max_retries

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

            return should_retry

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Retry strategy error: {str(e)}")
            raise

    def get_delay(self, attempt: int) -> float:
        """
        Calculate the delay before the next retry attempt.

        Args:
            attempt: Current attempt number

        Returns:
            Delay in seconds before next retry
        """
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Record start time
        start_time = time.time()

        try:
            # Get configuration from state
            backoff_factor = self._state.get("backoff_factor")
            max_backoff = self._state.get("max_backoff")

            # Calculate delay with exponential backoff
            delay = min(backoff_factor**attempt, max_backoff)

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

            return delay

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Retry strategy error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retry strategy usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "execution_count": self._state.get("execution_count", 0),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "error_count": self._state.get_metadata("error_count", 0),
            "max_retries": self._state.get("max_retries"),
            "backoff_factor": self._state.get("backoff_factor"),
            "max_backoff": self._state.get("max_backoff"),
        }

    def clear_cache(self) -> None:
        """Clear the retry strategy cache."""
        self._state.update("result_cache", {})
        logger.debug("Retry strategy cache cleared")
