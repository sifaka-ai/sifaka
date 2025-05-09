"""
Protocol interfaces for model providers.

This module defines the protocol interfaces for model providers,
establishing a common contract for model provider behavior.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.interfaces.core import Configurable, Identifiable

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
