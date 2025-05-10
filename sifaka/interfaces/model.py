"""
Model interfaces for Sifaka.

This module defines the interfaces for model providers in the Sifaka framework.
These interfaces establish a common contract for model provider behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **ModelProviderProtocol**: Base interface for all model providers
   - **AsyncModelProviderProtocol**: Interface for asynchronous model providers

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.

## State Management

The interfaces support standardized state management:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state manager methods
- Clear separation of configuration and state

## Error Handling

The interfaces define error handling patterns:
- ValueError for invalid inputs
- RuntimeError for execution failures
- TypeError for type mismatches
- Detailed error tracking and reporting

## Execution Tracking

The interfaces support execution tracking:
- Execution count tracking
- Execution time tracking
- Success/failure tracking
- Performance statistics
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.core.interfaces import Configurable, Identifiable

# Type variables
ConfigType = TypeVar("ConfigType")
ModelConfigType = TypeVar("ModelConfigType")


@runtime_checkable
class ModelProviderProtocol(Identifiable, Configurable[ModelConfigType], Protocol):
    """
    Protocol interface for model providers.

    This interface defines the contract for components that provide access to
    language models. It ensures that model providers can generate text, count
    tokens, and manage model configuration.

    ## Lifecycle

    1. **Initialization**: Set up model resources and configuration
    2. **Text Generation**: Generate text from prompts
    3. **Token Counting**: Count tokens in text
    4. **Configuration Management**: Manage model configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a generate method to generate text from prompts
    - Provide a count_tokens method to count tokens in text
    - Provide a model_name property to access the model name
    - Provide a config property to access the model configuration
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The model name
        """
        pass

    @property
    @abstractmethod
    def config(self) -> ModelConfigType:
        """
        Get the model configuration.

        Returns:
            The model configuration
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the model.

        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional keyword arguments for generation

        Returns:
            The generated text

        Raises:
            ValueError: If the prompt is invalid
            RuntimeError: If text generation fails
        """
        pass

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


@runtime_checkable
class AsyncModelProviderProtocol(Protocol):
    """
    Protocol interface for asynchronous model providers.

    This interface defines the contract for components that provide asynchronous
    access to language models. It ensures that model providers can generate text
    asynchronously, count tokens, and manage model configuration.

    ## Lifecycle

    1. **Initialization**: Set up model resources and configuration
    2. **Text Generation**: Generate text from prompts asynchronously
    3. **Token Counting**: Count tokens in text asynchronously
    4. **Configuration Management**: Manage model configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an async generate method to generate text from prompts
    - Provide an async count_tokens method to count tokens in text
    - Provide a model_name property to access the model name
    - Provide a config property to access the model configuration
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The model name
        """
        pass

    @property
    @abstractmethod
    def config(self) -> Any:
        """
        Get the model configuration.

        Returns:
            The model configuration
        """
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the model asynchronously.

        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional keyword arguments for generation

        Returns:
            The generated text

        Raises:
            ValueError: If the prompt is invalid
            RuntimeError: If text generation fails
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text asynchronously.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If the text is invalid
        """
        pass
