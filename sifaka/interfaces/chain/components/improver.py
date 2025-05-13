"""
Improver interface for Sifaka.

This module defines the interface for output improvers in the Sifaka framework.
These interfaces establish a common contract for improver behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Improver**: Interface for output improvers

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import List, Protocol, runtime_checkable

from ..base import ChainComponent
from ..models import ValidationResult


@runtime_checkable
class Improver(ChainComponent, Protocol):
    """
    Interface for output improvers.

    This interface defines the contract for components that improve outputs.
    It ensures that improvers can improve outputs based on validation results
    and provide consistent behavior across different implementations.

    ## Lifecycle

    1. **Initialization**: Set up improver resources
    2. **Improvement**: Improve outputs based on validation results
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an improve method to improve outputs based on validation results
    - Optionally provide an async improve_async method for asynchronous improvement
    """

    @abstractmethod
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
        ...

    async def improve_async(
        self, output: str, validation_results: List[ValidationResult]
    ) -> str:
        """
        Improve an output asynchronously.

        This method has a default implementation that calls the synchronous
        improve method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        import asyncio

        loop = asyncio.get_event_loop() if asyncio else ""
        return await loop.run_in_executor(
            None, lambda: self.improve(output, validation_results) if self else "" if loop else ""
        )
