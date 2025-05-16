"""
Anthropic model implementation for Sifaka.

This module provides an implementation of the Model protocol for Anthropic models.
"""

import os
from typing import Optional, Dict, Any, List

try:
    import anthropic
    from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from sifaka.errors import ModelError, ModelAPIError, ConfigurationError
from sifaka.registry import register_model


class AnthropicModel:
    """Anthropic model implementation.

    This class implements the Model protocol for Anthropic models.

    Attributes:
        model_name: The name of the Anthropic model to use.
        api_key: The Anthropic API key to use. If not provided, it will be read from the
            ANTHROPIC_API_KEY environment variable.
        client: The Anthropic client instance.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **options: Any):
        """Initialize the Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use.
            api_key: The Anthropic API key to use. If not provided, it will be read from the
                ANTHROPIC_API_KEY environment variable.
            **options: Additional options to pass to the Anthropic client.

        Raises:
            ConfigurationError: If the Anthropic package is not installed.
            ModelError: If the API key is not provided and not available in the environment.
        """
        if not ANTHROPIC_AVAILABLE:
            raise ConfigurationError(
                "Anthropic package not installed. Install it with 'pip install anthropic'."
            )

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.options = options

        if not self.api_key:
            raise ModelError(
                "Anthropic API key not provided. Either pass it as an argument or "
                "set the ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the Anthropic API.
                Supported options include:
                - temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                  lower values (e.g., 0.2) make it more deterministic.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - top_k: Controls diversity by limiting to top k tokens.
                - stop_sequences: Sequences where the API will stop generating further tokens.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error communicating with the Anthropic API.
        """
        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Convert max_tokens to max_tokens_to_sample if present
        if "max_tokens" in merged_options:
            merged_options["max_tokens_to_sample"] = merged_options.pop("max_tokens")

        # Convert stop to stop_sequences if present
        if "stop" in merged_options:
            merged_options["stop_sequences"] = merged_options.pop("stop")

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **merged_options,
            )
            return response.content[0].text
        except RateLimitError as e:
            raise ModelAPIError(f"Anthropic rate limit exceeded: {str(e)}")
        except APIConnectionError as e:
            raise ModelAPIError(f"Error connecting to Anthropic API: {str(e)}")
        except APIError as e:
            raise ModelAPIError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            raise ModelAPIError(f"Unexpected error when calling Anthropic API: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.

        Raises:
            ModelError: If there is an error counting tokens.
        """
        try:
            # Use Anthropic's token counting function
            return self.client.count_tokens(text)
        except Exception as e:
            raise ModelError(f"Error counting tokens: {str(e)}")


@register_model("anthropic")
def create_anthropic_model(model_name: str, **options: Any) -> AnthropicModel:
    """Create an Anthropic model instance.

    This factory function creates an Anthropic model instance with the specified
    model name and options. It is registered with the registry system for
    dependency injection.

    Args:
        model_name: The name of the Anthropic model to use.
        **options: Additional options to pass to the Anthropic model constructor.

    Returns:
        An Anthropic model instance.

    Raises:
        ConfigurationError: If the Anthropic package is not installed.
        ModelError: If the API key is not provided and not available in the environment.
    """
    return AnthropicModel(model_name=model_name, **options)
