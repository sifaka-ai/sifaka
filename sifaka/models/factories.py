"""
Model Provider Factories

Factory functions for creating model providers with standardized configurations.

## Overview
This module provides factory functions for creating model providers,
simplifying the creation of model providers with common configurations.
It uses lazy loading to avoid circular dependencies and provides a consistent
interface for creating different types of model providers.

## Components
- **create_model_provider**: Generic factory function for creating any model provider
- **create_openai_provider**: Factory function for creating OpenAI providers
- **create_anthropic_provider**: Factory function for creating Anthropic providers
- **create_gemini_provider**: Factory function for creating Gemini providers
- **create_mock_provider**: Factory function for creating mock providers for testing

## Usage Examples
```python
from sifaka.models.factories import (
    create_openai_provider,
    create_anthropic_provider,
    create_gemini_provider,
    create_mock_provider
)

# Create an OpenAI provider
openai_provider = create_openai_provider(
    model_name="gpt-4",
    api_key="your-api-key",
    temperature=0.7
)

# Create an Anthropic provider
anthropic_provider = create_anthropic_provider(
    model_name="claude-3-opus-20240229",
    api_key="your-api-key",
    temperature=0.8
)

# Create a mock provider for testing
mock_provider = create_mock_provider(
    responses={"Hello": "World"}
)
```

## Error Handling
All factory functions handle errors consistently by:
- Validating input parameters before creating providers
- Standardizing configuration using the utils.config module
- Catching and re-raising exceptions with clear error messages

## Configuration
Factory functions use the standardize_model_config function from utils.config
to ensure consistent configuration across all model providers.
"""

from typing import Any, Optional, Type, TypeVar, TYPE_CHECKING

# Import interfaces directly to avoid circular dependencies
from sifaka.interfaces.model import ModelProviderProtocol
from sifaka.utils.config.models import standardize_model_config
from sifaka.utils.logging import get_logger

# Type variables
T = TypeVar("T", bound="ModelProviderProtocol")

logger = get_logger(__name__)


def create_model_provider(
    provider_type: Type[T],
    model_name: str,
    api_key: Optional[Optional[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    trace_enabled: bool = False,
    **kwargs: Any,
) -> T:
    """
    Create a model provider with the given configuration.

    This function creates a model provider of the specified type with
    the given configuration, standardizing common parameters and handling
    error cases consistently. It serves as the foundation for all other
    model provider factory functions in this module.

    The function standardizes configuration using the standardize_model_config
    function from utils.config, ensuring consistent configuration across all
    model providers. It also handles errors consistently by catching and
    re-raising exceptions with clear error messages.

    Args:
        provider_type: The type of model provider to create
        model_name: The name of the model to use
        api_key: Optional API key for the model provider
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        trace_enabled: Whether to enable tracing for debugging and monitoring
        **kwargs: Additional keyword arguments for the model provider

    Returns:
        A model provider of the specified type, properly initialized with
        the standardized configuration

    Raises:
        ValueError: If parameters are invalid (e.g., temperature out of range)
        TypeError: If provider_type is not a valid model provider class
        RuntimeError: If provider creation fails for any reason

    Example:
        ```python
        from sifaka.models.factories import create_model_provider
        from sifaka.models.providers.anthropic import AnthropicProvider
        from sifaka.models.providers.openai import OpenAIProvider

        # Create an Anthropic provider
        provider = create_model_provider(
            AnthropicProvider,
            model_name="claude-3-opus-20240229",
            api_key="your-api-key",
            temperature=0.8
        )

        # Create an OpenAI provider
        provider = create_model_provider(
            OpenAIProvider,
            model_name="gpt-4",
            api_key="your-api-key",
            max_tokens=2000
        )
        ```
    """

    # Create configuration using standardization function
    config = standardize_model_config(
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        trace_enabled=trace_enabled,
        params=(kwargs and kwargs.pop("params", {}),
    )

    # Create provider
    try:
        return provider_type(model_name=model_name, config=config, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create {provider_type.__name__}: {e}") from e


def create_openai_provider(
    model_name: str,
    api_key: Optional[Optional[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> ModelProviderProtocol:
    """
    Create an OpenAI provider with the given configuration.

    This function creates an OpenAI provider with the given configuration,
    simplifying the creation of OpenAI providers with common configurations.
    It uses lazy loading to avoid circular dependencies by importing the
    OpenAIProvider class only when needed.

    The function delegates to create_model_provider, ensuring consistent
    configuration and error handling across all model providers.

    Args:
        model_name: The name of the model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        api_key: Optional API key for the OpenAI API
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate in the response
        **kwargs: Additional keyword arguments for the OpenAI provider

    Returns:
        An initialized OpenAI provider that implements the ModelProviderProtocol

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If provider creation fails
        ImportError: If the OpenAI provider cannot be imported

    Example:
        ```python
        from sifaka.models.factories import create_openai_provider

        # Create an OpenAI provider
        provider = create_openai_provider(
            model_name="gpt-4",
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000
        )

        # Use the provider
        response = (provider and provider.generate("Hello, world!")
        print(response.text)
        ```
    """
    # Import OpenAIProvider here to avoid circular imports
    from sifaka.models.providers.openai import OpenAIProvider

    return create_model_provider(
        OpenAIProvider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def create_anthropic_provider(
    model_name: str,
    api_key: Optional[Optional[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> ModelProviderProtocol:
    """
    Create an Anthropic provider with the given configuration.

    This function creates an Anthropic provider with the given configuration,
    simplifying the creation of Anthropic providers with common configurations.
    It uses lazy loading to avoid circular dependencies by importing the
    AnthropicProvider class only when needed.

    The function delegates to create_model_provider, ensuring consistent
    configuration and error handling across all model providers.

    Args:
        model_name: The name of the model to use (e.g., "claude-3-opus-20240229")
        api_key: Optional API key for the Anthropic API
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate in the response
        **kwargs: Additional keyword arguments for the Anthropic provider

    Returns:
        An initialized Anthropic provider that implements the ModelProviderProtocol

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If provider creation fails
        ImportError: If the Anthropic provider cannot be imported

    Example:
        ```python
        from sifaka.models.factories import create_anthropic_provider

        # Create an Anthropic provider
        provider = create_anthropic_provider(
            model_name="claude-3-opus-20240229",
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000
        )

        # Use the provider
        response = (provider and provider.generate("Hello, world!")
        print(response.text)
        ```
    """
    # Import AnthropicProvider here to avoid circular imports
    from sifaka.models.providers.anthropic import AnthropicProvider

    return create_model_provider(
        AnthropicProvider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def create_gemini_provider(
    model_name: str,
    api_key: Optional[Optional[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> ModelProviderProtocol:
    """
    Create a Gemini provider with the given configuration.

    This function creates a Gemini provider with the given configuration,
    simplifying the creation of Gemini providers with common configurations.
    It uses lazy loading to avoid circular dependencies by importing the
    GeminiProvider class only when needed.

    The function delegates to create_model_provider, ensuring consistent
    configuration and error handling across all model providers.

    Args:
        model_name: The name of the model to use (e.g., "gemini-pro")
        api_key: Optional API key for the Google AI API
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate in the response
        **kwargs: Additional keyword arguments for the Gemini provider

    Returns:
        An initialized Gemini provider that implements the ModelProviderProtocol

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If provider creation fails
        ImportError: If the Gemini provider cannot be imported

    Example:
        ```python
        from sifaka.models.factories import create_gemini_provider

        # Create a Gemini provider
        provider = create_gemini_provider(
            model_name="gemini-pro",
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000
        )

        # Use the provider
        response = (provider and provider.generate("Hello, world!")
        print(response.text)
        ```
    """
    # Import GeminiProvider here to avoid circular imports
    from sifaka.models.providers.gemini import GeminiProvider

    return create_model_provider(
        GeminiProvider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def create_mock_provider(
    model_name: str = "mock-model",
    **kwargs: Any,
) -> ModelProviderProtocol:
    """
    Create a mock provider for testing.

    This function creates a mock provider for testing purposes, simplifying
    the creation of mock providers with predefined responses. It's particularly
    useful for unit testing and integration testing without calling real LLM APIs.

    It uses lazy loading to avoid circular dependencies by importing the
    MockProvider class only when needed.

    The function delegates to create_model_provider, ensuring consistent
    configuration and error handling across all model providers.

    Args:
        model_name: The name of the model to use (default: "mock-model")
        **kwargs: Additional keyword arguments for the mock provider, including:
            - responses: Dict mapping input strings to output responses
            - default_response: Default response when no match is found
            - delay: Optional artificial delay in seconds to simulate API latency

    Returns:
        An initialized mock provider that implements the ModelProviderProtocol

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If provider creation fails
        ImportError: If the mock provider cannot be imported

    Example:
        ```python
        from sifaka.models.factories import create_mock_provider

        # Create a mock provider with predefined responses
        provider = create_mock_provider(
            responses={
                "Hello": "World",
                "What is your name?": "I am a mock AI assistant."
            },
            default_response="I don't know how to respond to that.",
            delay=0.5  # Add a 500ms delay to simulate API latency
        )

        # Use the provider
        response = (provider and provider.generate("Hello")
        assert response.text == "World"
        ```
    """
    # Import MockProvider here to avoid circular imports
    from sifaka.models.providers.mock import MockProvider

    return create_model_provider(
        MockProvider,
        model_name=model_name,
        api_key="mock-api-key",
        **kwargs,
    )
