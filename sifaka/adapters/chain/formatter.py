"""
Formatter Adapter Module

This module provides the FormatterAdapter class for adapting existing formatters
to the Formatter interface from the chain system.
"""

import asyncio
import time
from typing import Any, List

from sifaka.interfaces.chain.components.formatter import ChainFormatter as Formatter
from sifaka.interfaces.chain.models import ValidationResult
from sifaka.utils.errors.component import FormatterError
from sifaka.utils.errors.safe_execution import safely_execute_component_operation
from sifaka.utils.state import create_adapter_state


class FormatterAdapter(Formatter):
    """
    Adapter for existing formatters.

    This adapter implements the Formatter interface for existing formatters,
    using the standardized state management pattern.

    ## Architecture
    The FormatterAdapter follows the adapter pattern to wrap existing formatters:
    - Implements the Formatter interface from chain.interfaces
    - Uses standardized state management with _state_manager
    - Delegates to the wrapped formatter
    - Handles different formatter interfaces (format, process, run)
    - Provides a default implementation when no formatter is provided
    - Provides consistent error handling and statistics tracking

    ## Lifecycle
    1. **Initialization**: Adapter is created with a formatter (or None)
    2. **State Setup**: State manager is initialized with adapter state
    3. **Operation**: Adapter delegates to the formatter or uses default
    4. **Cleanup**: Resources are released when no longer needed

    ## Error Handling
    - FormatterError: Raised when formatting fails
    - Tracks error statistics in state manager
    - Provides detailed error messages with component information

    Attributes:
        _formatter (Any): The wrapped formatter (or None)
        _name (str): The name of the adapter
        _description (str): The description of the adapter
        _state_manager (StateManager): The state manager for the adapter
    """

    def __init__(self, formatter: Any, name: str = None, description: str = None):
        """
        Initialize the formatter adapter.

        Args:
            formatter: The formatter to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        # Store formatter
        self._formatter = formatter

        # Set name and description
        self._name = name or f"{type(formatter).__name__ if formatter else 'Default'}Adapter"
        self._description = (
            description
            or f"Adapter for {type(formatter).__name__ if formatter else 'default formatter'}"
        )

        # Create state manager
        self._state_manager = create_adapter_state()

        # Initialize state
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        # Update state with initial values
        self._state_manager.update("adaptee", self._formatter)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "formatter_adapter")
        self._state_manager.set_metadata(
            "adaptee_type", type(self._formatter).__name__ if self._formatter else "None"
        )
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    def description(self) -> str:
        """Get adapter description."""
        return self._description

    def format(self, output: str, validation_results: List[ValidationResult]) -> Any:
        """
        Format a result.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # If no formatter is provided, return a default result
        if self._formatter is None:
            from sifaka.core.results import ChainResult

            return ChainResult(
                output=output,
                validation_results=validation_results,
                prompt="",
                execution_time=0.0,
                attempt_count=1,
                metadata={},
            )

        # Record start time
        start_time = time.time()

        try:

            def format_operation():
                # Check for different formatter interfaces
                if hasattr(self._formatter, "format"):
                    return self._formatter.format(output, validation_results)
                elif hasattr(self._formatter, "format_result"):
                    return self._formatter.format_result(output, validation_results)
                elif hasattr(self._formatter, "process"):
                    return self._formatter.process(output, validation_results)
                elif hasattr(self._formatter, "run"):
                    return self._formatter.run(output, validation_results)
                else:
                    raise FormatterError(f"Unsupported formatter: {type(self._formatter).__name__}")

            # Execute operation safely
            result = safely_execute_component_operation(
                operation=format_operation,
                component_name=self.name,
                component_type="Formatter",
                error_class=FormatterError,
            )

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update format count
            format_count = self._state_manager.get_metadata("format_count", 0)
            self._state_manager.set_metadata("format_count", format_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * format_count) + execution_time) / (format_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as FormatterError
            if isinstance(e, FormatterError):
                raise e
            raise FormatterError(f"Formatting failed: {str(e)}")

    async def format_async(self, output: str, validation_results: List[ValidationResult]) -> Any:
        """
        Format a result asynchronously.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # If no formatter is provided, return a default result
        if self._formatter is None:
            from sifaka.core.results import ChainResult

            return ChainResult(
                output=output,
                validation_results=validation_results,
                prompt="",
                execution_time=0.0,
                attempt_count=1,
                metadata={},
            )

        # Record start time
        start_time = time.time()

        try:
            # Check if formatter has async methods
            if hasattr(self._formatter, "format_async"):
                result = await self._formatter.format_async(output, validation_results)
            elif hasattr(self._formatter, "format_result_async"):
                result = await self._formatter.format_result_async(output, validation_results)
            elif hasattr(self._formatter, "process_async"):
                result = await self._formatter.process_async(output, validation_results)
            elif hasattr(self._formatter, "run_async"):
                result = await self._formatter.run_async(output, validation_results)
            else:
                # Fall back to running synchronous method in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.format, output, validation_results)
                # Return early since statistics are updated in the synchronous method
                return result

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update format count
            format_count = self._state_manager.get_metadata("format_count", 0)
            self._state_manager.set_metadata("format_count", format_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * format_count) + execution_time) / (format_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as FormatterError
            if isinstance(e, FormatterError):
                raise e
            raise FormatterError(f"Async formatting failed: {str(e)}")
