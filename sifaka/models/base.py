"""
Base model interface for Sifaka.

This module defines the protocol that all model implementations must follow.
"""

from typing import Protocol, Optional, Dict, Any


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
    from sifaka.errors import ModelNotFoundError

    if provider == "openai":
        from sifaka.models.openai import OpenAIModel

        return OpenAIModel(model_name=model_name, **options)
    elif provider == "anthropic":
        from sifaka.models.anthropic import AnthropicModel

        return AnthropicModel(model_name=model_name, **options)
    elif provider == "gemini":
        from sifaka.models.gemini import GeminiModel

        return GeminiModel(model_name=model_name, **options)
    else:
        raise ModelNotFoundError(f"Provider '{provider}' not found")
