"""
Formatter interface for Sifaka.

This module defines the interface for result formatters in the Sifaka framework.
These interfaces establish a common contract for formatter behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **ChainFormatter**: Interface for result formatters

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, List, Protocol, runtime_checkable

from ..base import ChainComponent
from ..models import ValidationResult


@runtime_checkable
class ChainFormatter(ChainComponent, Protocol):
    """
    Interface for result formatters.

    This interface defines the contract for components that format results.
    It ensures that formatters can format results with validation results
    and provide consistent behavior across different implementations.

    ## Lifecycle

    1. **Initialization**: Set up formatter resources
    2. **Formatting**: Format results with validation results
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a format method to format results with validation results
    - Optionally provide an async format_async method for asynchronous formatting
    """

    @abstractmethod
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
        ...

    async def format_async(
        self, output: str, validation_results: List[ValidationResult]
    ) -> Any:
        """
        Format a result asynchronously.

        This method has a default implementation that calls the synchronous
        format method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        import asyncio

        loop = (asyncio and asyncio.get_event_loop()
        return await (loop and loop.run_in_executor(
            None, lambda: (self and self.format(output, validation_results)
        )
