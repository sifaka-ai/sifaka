"""
Protocol interfaces for token counters.

This module defines the protocol interfaces for token counters,
establishing a common contract for token counter behavior.
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
