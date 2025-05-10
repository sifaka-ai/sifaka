"""
Result formatter for Sifaka chains.

This module provides the result formatter for Sifaka chains,
enabling consistent formatting of chain outputs and validation results.
"""

from typing import Any, Dict, List, Optional, TypeVar
import time

from pydantic import PrivateAttr

from sifaka.core.base import BaseComponent, BaseResult, ComponentResultEnum
from sifaka.chain.result import ChainResult
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ResultFormatter(BaseComponent):
    """
    Result formatter for Sifaka chains.

    This class provides consistent formatting of chain outputs and validation results,
    with support for different output types and validation states. It uses the
    ChainResult class to create standardized result objects.
    """

    # State management
    _state_manager = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        name: str = "result_formatter",
        description: str = "Result formatter for Sifaka chains",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the result formatter.

        Args:
            name: Name of the formatter
            description: Description of the formatter
            config: Additional configuration
        """
        super().__init__(name=name, description=description, config=config)

        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("result_cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "result_formatter")
        self._state_manager.set_metadata("creation_time", time.time())

    def format_result(
        self, output: OutputType, validation_results: List[BaseResult]
    ) -> ChainResult[OutputType]:
        """
        Format the chain output and validation results.

        Args:
            output: The chain output to format
            validation_results: List of validation results

        Returns:
            Formatted ChainResult containing the output and validation state
        """
        # Track execution count
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)

        # Record start time
        start_time = time.time()

        try:
            # Determine status based on validation results
            status = (
                ComponentResultEnum.SUCCESS
                if all(r.passed for r in validation_results)
                else ComponentResultEnum.FAILURE
            )

            # Create result
            result = ChainResult(
                output=output,
                rule_results=validation_results,
                status=status,
                passed=status == ComponentResultEnum.SUCCESS,
                message="Chain execution completed",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            count = self._state_manager.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state_manager.set_metadata("max_execution_time", exec_time)

            return result

        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            logger.error(f"Result formatting error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about result formatter usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "execution_count": self._state_manager.get("execution_count", 0),
            "avg_execution_time": self._state_manager.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state_manager.get_metadata("max_execution_time", 0),
            "error_count": self._state_manager.get_metadata("error_count", 0),
        }

    def clear_cache(self) -> None:
        """Clear the result formatter cache."""
        self._state_manager.update("result_cache", {})
        logger.debug("Result formatter cache cleared")
