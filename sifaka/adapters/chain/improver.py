"""
Improver Adapter Module

This module provides the ImproverAdapter class for adapting existing critics
to the Improver interface from the chain system.
"""

import asyncio
import time
from typing import Any, List, Optional
from sifaka.interfaces.chain.components import Improver
from sifaka.interfaces.chain.models import ValidationResult
from sifaka.utils.errors.component import ImproverError
from sifaka.utils.errors.safe_execution import safely_execute_component_operation
from sifaka.utils.state import create_adapter_state


class ImproverAdapter(Improver):
    """
    Adapter for existing critics.

    This adapter implements the Improver interface for existing critics,
    using the standardized state management pattern.

    ## Architecture
    The ImproverAdapter follows the adapter pattern to wrap existing critics:
    - Implements the Improver interface from chain.interfaces
    - Uses standardized state management with _state_manager
    - Delegates to the wrapped critic
    - Handles different critic interfaces (improve, refine, process, run)
    - Provides consistent error handling and statistics tracking

    ## Lifecycle
    1. **Initialization**: Adapter is created with a critic
    2. **State Setup**: State manager is initialized with adapter state
    3. **Operation**: Adapter delegates to the critic
    4. **Cleanup**: Resources are released when no longer needed

    ## Error Handling
    - ImproverError: Raised when improvement fails
    - Tracks error statistics in state manager
    - Provides detailed error messages with component information

    Attributes:
        _improver (Any): The wrapped critic
        _name (str): The name of the adapter
        _description (str): The description of the adapter
        _state_manager (StateManager): The state manager for the adapter
    """

    def __init__(
        self, improver: Any, name: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        """
        Initialize the improver adapter.

        Args:
            improver: The critic to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        self._improver = improver
        self._name = name or f"{type(improver).__name__}Adapter"
        self._description = description or f"Adapter for {type(improver).__name__}"
        self._state_manager = create_adapter_state()
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        self._state_manager.update("adaptee", self._improver)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", "improver_adapter")
        self._state_manager.set_metadata("adaptee_type", type(self._improver).__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    def description(self) -> str:
        """Get adapter description."""
        return self._description

    def improve(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve an output based on validation results.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        start_time = time.time()
        try:

            def improve_operation() -> Any:
                if hasattr(self._improver, "improve"):
                    return self._improver.improve(output, validation_results)
                elif hasattr(self._improver, "refine"):
                    return self._improver.refine(output, validation_results)
                elif hasattr(self._improver, "process"):
                    return self._improver.process(output, validation_results)
                elif hasattr(self._improver, "run"):
                    return self._improver.run(output, validation_results)
                else:
                    raise ImproverError(f"Unsupported improver: {type(self._improver).__name__}")

            result = safely_execute_component_operation(
                operation=improve_operation,
                component_name=self.name,
                component_type="Improver",
                error_class=ImproverError,
            )

            end_time = time.time()
            execution_time = end_time - start_time
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = (avg_time * improvement_count + execution_time) / (improvement_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)
            max_time = self._state_manager.get_metadata("max_execution_time", 0)

            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            if isinstance(e, ImproverError):
                raise e
            raise ImproverError(f"Improvement failed: {str(e)}")

    async def improve_async(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve an output asynchronously.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        start_time = time.time()
        try:
            if hasattr(self._improver, "improve_async"):
                result = await self._improver.improve_async(output, validation_results)
            elif hasattr(self._improver, "refine_async"):
                result = await self._improver.refine_async(output, validation_results)
            elif hasattr(self._improver, "process_async"):
                result = await self._improver.process_async(output, validation_results)
            elif hasattr(self._improver, "run_async"):
                result = await self._improver.run_async(output, validation_results)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.improve, output, validation_results)
                return result

            end_time = time.time()
            execution_time = end_time - start_time
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = (avg_time * improvement_count + execution_time) / (improvement_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)
            max_time = self._state_manager.get_metadata("max_execution_time", 0)

            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            if isinstance(e, ImproverError):
                raise e
            raise ImproverError(f"Async improvement failed: {str(e)}")
