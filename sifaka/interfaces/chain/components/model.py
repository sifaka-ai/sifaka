"""
Model interface for Sifaka.

This module defines the interface for text generation models in the Sifaka framework.
These interfaces establish a common contract for model behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Model**: Interface for text generation models

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from ..base import ChainComponent


@runtime_checkable
class Model(ChainComponent, Protocol):
    """
    Interface for text generation models.

    This interface defines the contract for components that generate text.
    It ensures that models can generate text from prompts and provide
    consistent behavior across different implementations.

    ## Lifecycle

    1. **Initialization**: Set up model resources
    2. **Generation**: Generate text from prompts
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a generate method to generate text from prompts
    - Optionally provide an async generate_async method for asynchronous generation
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        ...

    async def generate_async(self, prompt: str) -> str:
        """
        Generate text asynchronously.

        This method has a default implementation that calls the synchronous
        generate method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)
