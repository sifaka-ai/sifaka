"""
Result formatter interface for Sifaka.

This module defines the interface for result formatters in the Sifaka framework.
These interfaces establish a common contract for result formatter behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **ResultFormatter**: Interface for result formatters

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

# Type variables
ResultType = TypeVar("ResultType", covariant=True)


@runtime_checkable
class ResultFormatter(Protocol[ResultType]):
    """
    Interface for result formatters.

    This interface defines the contract for components that format results.
    It ensures that result formatters can format results for different outputs.

    ## Lifecycle

    1. **Initialization**: Set up result formatting resources
    2. **Formatting**: Format results for different outputs
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a format method to format results
    """

    @abstractmethod
    def format(self, result: Any) -> ResultType:
        """
        Format a result.

        Args:
            result: The result to format

        Returns:
            A formatted result

        Raises:
            ValueError: If the result is invalid
        """
