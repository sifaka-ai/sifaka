"""
Result formatter for Sifaka chains.

This module provides the result formatter for Sifaka chains,
enabling consistent formatting of chain outputs and validation results.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
import time

from pydantic import BaseModel, PrivateAttr

from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ResultFormatter(BaseComponent):
    """
    Result formatter for Sifaka chains.

    This class provides consistent formatting of chain outputs and validation results,
    with support for different output types and validation states.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

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
        super().__init__()

        self._state.update("name", name)
        self._state.update("description", description)
        self._state.update("config", config or {})
        self._state.update("initialized", True)
        self._state.update("execution_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "result_formatter")
        self._state.set_metadata("creation_time", time.time())

    def format_result(
        self, output: OutputType, validation_results: List[BaseResult]
    ) -> BaseResult[OutputType]:
        """
        Format the chain output and validation results.

        Args:
            output: The chain output to format
            validation_results: List of validation results

        Returns:
            Formatted result containing the output and validation state
        """
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Record start time
        start_time = time.time()

        try:
            # Create result
            result = BaseResult(
                output=output,
                validation_results=validation_results,
                status=(
                    ComponentResultEnum.SUCCESS
                    if all(r.passed for r in validation_results)
                    else ComponentResultEnum.FAILURE
                ),
            )

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

            return result

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Result formatting error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about result formatter usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "execution_count": self._state.get("execution_count", 0),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "error_count": self._state.get_metadata("error_count", 0),
        }

    def clear_cache(self) -> None:
        """Clear the result formatter cache."""
        self._state.update("result_cache", {})
        logger.debug("Result formatter cache cleared")
