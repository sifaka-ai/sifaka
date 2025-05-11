"""
Factory functions for model providers.

This module provides factory functions for creating model providers,
simplifying the creation of model providers with common configurations.
"""

from typing import Any, Optional, Type, TypeVar

from sifaka.models.core import ModelProviderCore
from sifaka.utils.config import standardize_model_config
from sifaka.models.providers.openai import OpenAIProvider
from sifaka.models.providers.anthropic import AnthropicProvider
from sifaka.models.providers.gemini import GeminiProvider
from sifaka.models.providers.mock import MockProvider

# Type variables
T = TypeVar("T", bound=ModelProviderCore)


def create_model_provider(
    provider_type: Type[T],
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    trace_enabled: bool = False,
    **kwargs: Any,
) -> T:
    """
    Create a model provider with the given configuration.

    This function creates a model provider of the specified type with
    the given configuration, simplifying the creation of model providers
    with common configurations.

    Args:
        provider_type: The type of model provider to create
        model_name: The name of the model to use
        api_key: Optional API key for the model provider
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        trace_enabled: Whether to enable tracing
        **kwargs: Additional keyword arguments for the model provider

    Returns:
        A model provider of the specified type

    Raises:
        ValueError: If parameters are invalid
        TypeError: If provider_type is not a valid model provider class
        RuntimeError: If provider creation fails

    Examples:
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
        params=kwargs.pop("params", {}),
    )

    # Create provider
    try:
        return provider_type(model_name=model_name, config=config, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create {provider_type.__name__}: {e}") from e


def create_openai_provider(
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> OpenAIProvider:
    """
    Create an OpenAI provider with the given configuration.

    This function creates an OpenAI provider with the given configuration,
    simplifying the creation of OpenAI providers with common configurations.

    Args:
        model_name: The name of the model to use
        api_key: Optional API key for the model provider
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional keyword arguments for the model provider

    Returns:
        An OpenAI provider

    Examples:
        ```python
        from sifaka.models.factories import create_openai_provider

        # Create an OpenAI provider
        provider = create_openai_provider(
            model_name="gpt-4",
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000
        )
        ```
    """
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
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> AnthropicProvider:
    """
    Create an Anthropic provider with the given configuration.

    This function creates an Anthropic provider with the given configuration,
    simplifying the creation of Anthropic providers with common configurations.

    Args:
        model_name: The name of the model to use
        api_key: Optional API key for the model provider
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional keyword arguments for the model provider

    Returns:
        An Anthropic provider

    Examples:
        ```python
        from sifaka.models.factories import create_anthropic_provider

        # Create an Anthropic provider
        provider = create_anthropic_provider(
            model_name="claude-3-opus-20240229",
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000
        )
        ```
    """
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
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> GeminiProvider:
    """
    Create a Gemini provider with the given configuration.

    This function creates a Gemini provider with the given configuration,
    simplifying the creation of Gemini providers with common configurations.

    Args:
        model_name: The name of the model to use
        api_key: Optional API key for the model provider
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional keyword arguments for the model provider

    Returns:
        A Gemini provider

    Examples:
        ```python
        from sifaka.models.factories import create_gemini_provider

        # Create a Gemini provider
        provider = create_gemini_provider(
            model_name="gemini-pro",
            api_key="your-api-key",
            temperature=0.7,
            max_tokens=1000
        )
        ```
    """
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
) -> MockProvider:
    """
    Create a mock provider for testing.

    This function creates a mock provider for testing, simplifying
    the creation of mock providers with common configurations.

    Args:
        model_name: The name of the model to use
        **kwargs: Additional keyword arguments for the model provider

    Returns:
        A mock provider

    Examples:
        ```python
        from sifaka.models.factories import create_mock_provider

        # Create a mock provider
        provider = create_mock_provider(
            model_name="mock-model",
            responses={"Hello": "World"}
        )
        ```
    """
    return create_model_provider(
        MockProvider,
        model_name=model_name,
        api_key="mock-api-key",
        **kwargs,
    )
