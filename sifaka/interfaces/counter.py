"""
Token counter interfaces for Sifaka.

This module defines the interfaces for token counters in the Sifaka framework.
These interfaces establish a common contract for token counter behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **TokenCounterProtocol**: Base interface for all token counters

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.

## Error Handling

The interfaces define error handling patterns:
- ValueError for invalid inputs
- RuntimeError for execution failures
- TypeError for type mismatches
- Detailed error tracking and reporting
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounterProtocol(Protocol):
    """
    Protocol interface for token counters.

    This interface defines the contract for components that count tokens in text.
    It ensures that token counters can count tokens in text and estimate token
    usage for prompts.

    ## Lifecycle

    1. **Initialization**: Set up token counting resources
    2. **Token Counting**: Count tokens in text
    3. **Token Estimation**: Estimate token usage for prompts
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a count_tokens method to count tokens in text
    - Handle different text formats and encodings
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If the text is invalid
        """
        pass
