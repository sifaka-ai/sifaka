"""
Model provider interfaces for Sifaka.

This module defines the interfaces for model providers in the Sifaka framework.
These interfaces establish a common contract for model provider behavior,
enabling better modularity and extensibility.

## Interface Hierarchy

1. **ModelProvider**: Base interface for all model providers
   - **APIClient**: Interface for API clients
   - **TokenCounter**: Interface for token counters
   - **GenerationService**: Interface for text generation services

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.interfaces.core import Component, Configurable, Identifiable, Stateful

# Type variables
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")
ModelConfigType = TypeVar("ModelConfigType")
PromptType = TypeVar("PromptType")
ResponseType = TypeVar("ResponseType")


@runtime_checkable
class APIClient(Protocol):
    """
    Interface for API clients that handle direct communication with LLM services.

    This interface defines the contract for components that send prompts to
    language model services and return the responses. It ensures that API clients
    can send prompts, handle errors, and manage authentication.

    ## Lifecycle

    1. **Initialization**: Set up client libraries and authentication
    2. **Request Preparation**: Format prompts and parameters for the API
    3. **Request Execution**: Send requests to the API
    4. **Response Handling**: Process and return responses
    5. **Error Handling**: Manage API errors and retries
    6. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a send_prompt method to send prompts to the API
    - Handle API errors and retries appropriately
    - Manage authentication and API keys securely
    """

    @abstractmethod
    def send_prompt(self, prompt: Any, config: Dict[str, Any]) -> Any:
        """
        Send a prompt to the API and return the response.

        Args:
            prompt: The prompt to send
            config: Configuration parameters for the request

        Returns:
            The API response

        Raises:
            ValueError: If the prompt or config is invalid
            RuntimeError: If the API request fails
        """
        pass


@runtime_checkable
class TokenCounter(Protocol):
    """
    Interface for token counters that count tokens in text.

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
        Count the number of tokens in text.

        Args:
            text: The text to count tokens in

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If the text is invalid
        """
        pass


@runtime_checkable
class GenerationService(Protocol):
    """
    Interface for text generation services.

    This interface defines the contract for components that generate text using
    language models. It ensures that generation services can generate text,
    handle errors, and manage generation parameters.

    ## Lifecycle

    1. **Initialization**: Set up generation resources
    2. **Text Generation**: Generate text from prompts
    3. **Error Handling**: Manage generation errors and retries
    4. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a generate method to generate text from prompts
    - Handle generation errors and retries appropriately
    - Manage generation parameters
    """

    @abstractmethod
    def generate(self, prompt: Any, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional generation parameters

        Returns:
            The generated text

        Raises:
            ValueError: If the prompt or parameters are invalid
            RuntimeError: If text generation fails
        """
        pass


@runtime_checkable
class ModelProvider(Identifiable, Configurable[ModelConfigType], Protocol):
    """
    Interface for model providers.

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
    - Provide a config property to access the model configuration
    - Provide an update_config method to update the model configuration
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

    @abstractmethod
    def generate(self, prompt: Any, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional generation parameters

        Returns:
            The generated text

        Raises:
            ValueError: If the prompt or parameters are invalid
            RuntimeError: If text generation fails
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: The text to count tokens in

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If the text is invalid
        """
        pass


@runtime_checkable
class AsyncModelProvider(Protocol):
    """
    Interface for asynchronous model providers.

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
    - Provide a config property to access the model configuration
    - Provide an update_config method to update the model configuration
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

    @abstractmethod
    async def generate(self, prompt: Any, **kwargs: Any) -> str:
        """
        Generate text from a prompt asynchronously.

        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional generation parameters

        Returns:
            The generated text

        Raises:
            ValueError: If the prompt or parameters are invalid
            RuntimeError: If text generation fails
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text asynchronously.

        Args:
            text: The text to count tokens in

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If the text is invalid
        """
        pass
