"""
Base model interface for Sifaka.

This module defines the protocol that all model implementations must follow.
It also provides a factory function for creating model instances.
"""

from typing import Protocol, Any, Dict, Optional, Callable

# Import the registry
from sifaka.registry import register_model, get_model_factory


class Model(Protocol):
    """Protocol defining the interface for model providers."""

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the model.

        Returns:
            The generated text.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.
        """
        ...


def create_model(provider: str, model_name: str, **options: Any) -> Model:
    """Create a model instance based on provider and model name.

    This function uses the registry to find the appropriate model factory
    for the specified provider.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").
        model_name: The model name (e.g., "gpt-4", "claude-3").
        **options: Additional options to pass to the model constructor.

    Returns:
        A model instance.

    Raises:
        ModelNotFoundError: If the provider or model is not found.
        ConfigurationError: If the required package for the provider is not installed.
        ModelError: If there is an error initializing the model.
    """
    from sifaka.errors import ModelNotFoundError, ConfigurationError
    from sifaka.factories import create_model as factory_create_model

    # Use the factory function from sifaka.factories
    try:
        return factory_create_model(provider, model_name, **options)
    except Exception as e:
        # Fall back to direct imports if factory fails
        provider = provider.lower()

        if provider == "openai":
            try:
                from sifaka.models.openai import OpenAIModel, OPENAI_AVAILABLE

                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI package not available")
                return OpenAIModel(model_name=model_name, **options)
            except ImportError:
                raise ConfigurationError(
                    "OpenAI package not installed. Install it with 'pip install openai tiktoken'."
                )
        elif provider == "anthropic":
            try:
                from sifaka.models.anthropic import AnthropicModel

                return AnthropicModel(model_name=model_name, **options)
            except ImportError:
                raise ConfigurationError(
                    "Anthropic package not installed or Anthropic model not yet implemented."
                )
        elif provider == "gemini":
            try:
                from sifaka.models.gemini import GeminiModel

                return GeminiModel(model_name=model_name, **options)
            except ImportError:
                raise ConfigurationError(
                    "Google Gemini package not installed or Gemini model not yet implemented."
                )
        elif provider == "mock":
            # Create a simple mock model for testing
            class MockModel:
                def __init__(self, model_name: str, **kwargs: Any):
                    self.model_name = model_name
                    self.kwargs = kwargs

                def generate(self, prompt: str, **_: Any) -> str:
                    return f"Mock response from {self.model_name} for: {prompt}"

                def count_tokens(self, text: str) -> int:
                    return len(text.split())

            return MockModel(model_name, **options)
        else:
            raise ModelNotFoundError(f"Provider '{provider}' not found")
