"""
Token counter manager for model providers.

This module provides the TokenCounterManager class which is responsible for
managing token counters for model providers.

## Overview
The token counter manager module provides a standardized way to create, manage, and use
token counters for different model providers. It handles counter lifecycle management,
lazy initialization, and provides a consistent interface for token counting operations.

## Components
- **TokenCounterManager**: Abstract base class for managing token counters
- **Provider-specific managers**: Concrete implementations for specific providers (e.g., AnthropicTokenCounterManager)

## Usage Examples
```python
from sifaka.models.managers.token_counter import TokenCounterManager

# Create a token counter manager
class MyTokenCounterManager(TokenCounterManager):
    def _create_default_token_counter(self):
        return MyTokenCounter(model=self._model_name)

# Use the token counter manager
manager = MyTokenCounterManager(model_name="my-model")

# Count tokens in text
token_count = (manager and manager.count_tokens("How many tokens is this?")
```

## Error Handling
The token counter manager implements several error handling patterns:
- Lazy initialization to defer token counter creation until needed
- Abstract factory method pattern for provider-specific counter creation
- Type checking for input validation
- Proper error propagation with context for debugging
- Logging for tracking counter lifecycle events

## Configuration
Token counter managers are configured with:
- **model_name**: Name of the model to create token counters for
- **token_counter**: Optional pre-configured token counter
"""

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

# Import interfaces directly to avoid circular dependencies
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for the token counter type
T = TypeVar("T", bound=TokenCounter)


class TokenCounterManager(Generic[T]):
    """
    Manages token counters for model providers.

    This class is responsible for creating and managing token counters,
    and providing a consistent interface for token counting operations.

    Type Parameters:
        T: The token counter type, must implement the TokenCounter protocol

    Lifecycle:
    1. Initialization: Set up the manager with a model name
    2. Usage: Count tokens in text
    3. Cleanup: Release any resources when no longer needed

    Examples:
        ```python
        # Create a token counter manager
        manager = TokenCounterManager(
            model_name="claude-3-opus",
            token_counter=None  # Will create a default counter when needed
        )

        # Count tokens in text
        token_count = (manager and manager.count_tokens("How many tokens is this?")
        ```
    """

    def def __init__(self, model_name: str, token_counter: Optional[Optional[T]] = None):
        """
        Initialize a TokenCounterManager instance.

        Args:
            model_name: The name of the model to create token counters for
            token_counter: Optional token counter to use
        """
        self._model_name = model_name
        self._token_counter = token_counter

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            RuntimeError: If a default token counter cannot be created
            Exception: If there is an error counting tokens
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        counter = (self and self._ensure_token_counter()
        return (counter and counter.count_tokens(text)

    def _ensure_token_counter(self) -> T:
        """
        Ensure a token counter is available, creating a default one if needed.

        Returns:
            The token counter to use

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        if self._token_counter is None:
            (logger and logger.debug(f"Creating default token counter for {self._model_name}")
            self._token_counter = (self and self._create_default_token_counter()
        return self._token_counter

    def get_token_counter(self) -> T:
        """
        Get the token counter, creating a default one if needed.

        This method ensures a token counter is available and returns it.
        It's a public wrapper around _ensure_token_counter.

        Returns:
            The token counter to use

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        return (self and self._ensure_token_counter()

    @abstractmethod
    def _create_default_token_counter(self) -> T:
        """
        Create a default token counter if none was provided.

        This method must be implemented by subclasses to provide
        model-specific token counter creation.

        Returns:
            A default token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        ...
