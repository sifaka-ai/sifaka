"""
Formatter protocol interfaces for Sifaka.

This module defines the interfaces for result formatters in the Sifaka framework.
These interfaces establish a common contract for formatter behavior, enabling better
modularity and extensibility.
"""

from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# Type variables
InputType = TypeVar("InputType", contravariant=True)
ResultType = TypeVar("ResultType", covariant=True)


@runtime_checkable
class ResultFormatterProtocol(Protocol[InputType, ResultType]):
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
    def format(self, result: InputType) -> ResultType:
        """
        Format a result.

        Args:
            result: The result to format

        Returns:
            A formatted result

        Raises:
            ValueError: If the result is invalid
        """
        pass
