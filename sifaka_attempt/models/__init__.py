"""
Model providers for the Sifaka library.

This package provides model provider classes that generate text using various
language models.
"""

from typing import Protocol, Dict, Any, Union, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    """
    Protocol for model providers that generate text.

    Model providers implement this protocol to generate text using
    various language models.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        ...


def create_model(model_type: str, **kwargs: Any) -> ModelProvider:
    """
    Create a model provider of the given type.

    This factory function creates a model provider of the given type
    with the given parameters.

    Args:
        model_type: The type of model provider to create (e.g., "openai", "anthropic")
        **kwargs: Additional parameters for the model provider

    Returns:
        A model provider instance

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(**kwargs)
    elif model_type == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
