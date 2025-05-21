"""
Model factory for Sifaka.

This module provides factory functions for creating model instances based on provider
and model name. It supports all implemented model providers (OpenAI, Anthropic, Gemini)
and includes error handling for missing dependencies.
"""

import logging
import os
from typing import Any

from sifaka.core.interfaces import Model

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Exception raised when a model provider or model is not found."""

    pass


class ModelConfigurationError(ModelError):
    """Exception raised when there is a configuration error with a model."""

    pass


def create_model(provider_or_combined: str, model_name: str = "", **options: Any) -> Model:
    """
    Create a model instance based on provider and model name.

    This function creates a model instance for the specified provider and model name.
    The provider can be specified in two ways:
    1. As separate provider and model_name parameters: create_model("openai", "gpt-4")
    2. As a combined string with provider:model format: create_model("openai:gpt-4")

    If the combined format is used, the model_name parameter should be empty.

    Args:
        provider_or_combined: The provider name (e.g., "openai", "anthropic") or a combined
            provider:model string (e.g., "openai:gpt-4").
        model_name: The model name (e.g., "gpt-4", "claude-3"). Not needed if
            using the combined provider:model format.
        **options: Additional options to pass to the model constructor, such as:
            - api_key: API key for the provider
            - temperature: Controls randomness (0.0 to 1.0)
            - max_tokens: Maximum number of tokens to generate

    Returns:
        A model instance implementing the Model interface.

    Raises:
        ModelNotFoundError: If the provider or model is not found.
        ModelConfigurationError: If the required package for the provider is not installed.
        ModelError: If there is an error initializing the model.
    """
    # Parse provider and model name if combined format is used
    if ":" in provider_or_combined and not model_name:
        provider, model_name = provider_or_combined.split(":", 1)
    else:
        provider = provider_or_combined

    # Normalize provider name
    provider = provider.lower()

    # Create model based on provider
    if provider == "openai":
        return create_openai_model(model_name, **options)
    elif provider == "anthropic":
        return create_anthropic_model(model_name, **options)
    elif provider == "gemini":
        return create_gemini_model(model_name, **options)
    elif provider == "mock":
        return create_mock_model(model_name, **options)
    else:
        raise ModelNotFoundError(f"Provider '{provider}' not found")


def create_openai_model(model_name: str, **options: Any) -> Model:
    """
    Create an OpenAI model instance.

    Args:
        model_name: The name of the OpenAI model to use.
        **options: Additional options to pass to the OpenAI model constructor.

    Returns:
        An OpenAI model instance.

    Raises:
        ModelConfigurationError: If the OpenAI package is not installed.
        ModelError: If there is an error initializing the model.
    """
    try:
        from sifaka.models.openai_model import OpenAIModel

        # Get API key from options or environment variable
        api_key = options.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")

        # Create model
        return OpenAIModel(model_name=model_name, api_key=api_key, **options)
    except ImportError as e:
        logger.error(f"Error importing OpenAI model: {e}")
        raise ModelConfigurationError(
            "OpenAI package not installed. Install it with 'pip install openai tiktoken'."
        )
    except Exception as e:
        logger.error(f"Error creating OpenAI model: {e}")
        raise ModelError(f"Error creating OpenAI model: {str(e)}")


def create_anthropic_model(model_name: str, **options: Any) -> Model:
    """
    Create an Anthropic model instance.

    Args:
        model_name: The name of the Anthropic model to use.
        **options: Additional options to pass to the Anthropic model constructor.

    Returns:
        An Anthropic model instance.

    Raises:
        ModelConfigurationError: If the Anthropic package is not installed.
        ModelError: If there is an error initializing the model.
    """
    try:
        from sifaka.models.anthropic_model import AnthropicModel

        # Get API key from options or environment variable
        api_key = options.pop("api_key", None) or os.environ.get("ANTHROPIC_API_KEY")

        # Create model
        return AnthropicModel(model_name=model_name, api_key=api_key, **options)
    except ImportError as e:
        logger.error(f"Error importing Anthropic model: {e}")
        raise ModelConfigurationError(
            "Anthropic package not installed. Install it with 'pip install anthropic'."
        )
    except Exception as e:
        logger.error(f"Error creating Anthropic model: {e}")
        raise ModelError(f"Error creating Anthropic model: {str(e)}")


def create_gemini_model(model_name: str, **options: Any) -> Model:
    """
    Create a Google Gemini model instance.

    Args:
        model_name: The name of the Gemini model to use.
        **options: Additional options to pass to the Gemini model constructor.

    Returns:
        A Gemini model instance.

    Raises:
        ModelConfigurationError: If the Google Generative AI package is not installed.
        ModelError: If there is an error initializing the model.
    """
    try:
        from sifaka.models.gemini_model import GeminiModel

        # Get API key from options or environment variable
        api_key = options.pop("api_key", None) or os.environ.get("GOOGLE_API_KEY")

        # Create model
        return GeminiModel(model_name=model_name, api_key=api_key, **options)
    except ImportError as e:
        logger.error(f"Error importing Gemini model: {e}")
        raise ModelConfigurationError(
            "Google Generative AI package not installed. Install it with 'pip install google-generativeai'."
        )
    except Exception as e:
        logger.error(f"Error creating Gemini model: {e}")
        raise ModelError(f"Error creating Gemini model: {str(e)}")


def create_mock_model(model_name: str, **options: Any) -> Model:
    """
    Create a mock model for testing.

    Args:
        model_name: The name of the mock model.
        **options: Additional options to pass to the mock model.

    Returns:
        A mock model instance.
    """
    from sifaka.models.mock_model import MockModel

    return MockModel(model_name=model_name, **options)
