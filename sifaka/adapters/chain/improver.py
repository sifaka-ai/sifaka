"""
Improver Adapter Module

This module provides the ImproverAdapter class for adapting existing critics
to the Improver interface from the chain system.
"""

import time
from typing import Any, List, Optional, Union, cast
from sifaka.interfaces.chain.components import Improver
from sifaka.interfaces.chain.models import ValidationResult
from sifaka.utils.errors.component import ImproverError
from sifaka.utils.errors.results import ErrorResult
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

            def improve_operation() -> str:
                if hasattr(self._improver, "improve"):
                    return str(self._improver.improve(output, validation_results))
                elif hasattr(self._improver, "refine"):
                    return str(self._improver.refine(output, validation_results))
                elif hasattr(self._improver, "process"):
                    return str(self._improver.process(output, validation_results))
                elif hasattr(self._improver, "run"):
                    return str(self._improver.run(output, validation_results))
                else:
                    raise ImproverError(f"Unsupported improver: {type(self._improver).__name__}")

            result_or_error = safely_execute_component_operation(
                operation=improve_operation,
                component_name=self.name,
                component_type="Improver",
                error_class=ImproverError,
            )

            # Handle the case where result might be an ErrorResult
            if isinstance(result_or_error, ErrorResult):
                raise ImproverError(f"Improvement failed: {result_or_error.error_message}")

            result = str(result_or_error)

            # Handle the case where result might be an ErrorResult
            if isinstance(result, ErrorResult):
                raise ImproverError(f"Improvement failed: {result.error_message}")

            # Convert result to string if it's not already
            improved_text = str(result)

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
                cache[cache_key] = improved_text
                self._state_manager.update("cache", cache)

            return improved_text

        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            if isinstance(e, ImproverError):
                raise e
            raise ImproverError(f"Improvement failed: {str(e)}")
