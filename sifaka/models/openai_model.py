"""
OpenAI model implementation for Sifaka.

This module provides an implementation of the Model interface for OpenAI models.
"""

import os
from typing import Optional, Dict, Any

try:
    import tiktoken
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from sifaka.interfaces import Model
from sifaka.registry import register_model


class ModelError(Exception):
    """Error raised when a model operation fails."""
    pass


class ConfigurationError(Exception):
    """Error raised when a configuration is invalid."""
    pass


class OpenAIModel:
    """OpenAI model implementation.

    This class implements the Model interface for OpenAI models.

    Attributes:
        model_name: The name of the OpenAI model to use.
        api_key: The OpenAI API key to use.
        organization: The OpenAI organization ID to use.
        options: Additional options to pass to the OpenAI client.
        client: The OpenAI client instance.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **options: Any,
    ):
        """Initialize the OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use.
            api_key: The OpenAI API key to use. If not provided, it will be read from the
                OPENAI_API_KEY environment variable.
            organization: The OpenAI organization ID to use.
            **options: Additional options to pass to the OpenAI client.

        Raises:
            ConfigurationError: If the OpenAI package is not installed.
            ModelError: If the API key is not provided and not available in the environment.
        """
        if not OPENAI_AVAILABLE:
            raise ConfigurationError(
                "OpenAI package not installed. Install it with 'pip install openai tiktoken'."
            )

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization = organization
        self.options = options

        if not self.api_key:
            raise ModelError(
                "OpenAI API key not provided. Either pass it as an argument or "
                "set the OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
        )

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the OpenAI API.
                Supported options include:
                - temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                  lower values (e.g., 0.2) make it more deterministic.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - frequency_penalty: Reduces repetition of token sequences.
                - presence_penalty: Reduces repetition of topics.
                - stop: Sequences where the API will stop generating further tokens.
                - system_message: A system message to include at the beginning of the conversation.
                - use_completion_api: Force using the completion API instead of chat API.

        Returns:
            The generated text.

        Raises:
            ModelError: If there is an error communicating with the OpenAI API.
        """
        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Determine if we should use the completion API
        use_completion_api = merged_options.pop("use_completion_api", False)

        # Extract system message if provided
        system_message = merged_options.pop("system_message", None)

        try:
            # Use the appropriate API based on the model and options
            if not use_completion_api and self._is_chat_model():
                return self._generate_chat(prompt, system_message, **merged_options)
            else:
                return self._generate_completion(prompt, **merged_options)
        except RateLimitError as e:
            raise ModelError(f"OpenAI rate limit exceeded: {str(e)}")
        except APIConnectionError as e:
            raise ModelError(f"Error connecting to OpenAI API: {str(e)}")
        except APIError as e:
            raise ModelError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ModelError(f"Unexpected error when calling OpenAI API: {str(e)}")

    def _generate_chat(
        self, prompt: str, system_message: Optional[str] = None, **options: Any
    ) -> str:
        """Generate text using the chat completions API.

        Args:
            prompt: The prompt to generate text from.
            system_message: Optional system message to include.
            **options: Additional options to pass to the OpenAI API.

        Returns:
            The generated text.
        """
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Add system message if provided
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        # Send request to OpenAI
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **options
        )

        return response.choices[0].message.content or ""

    def _generate_completion(self, prompt: str, **options: Any) -> str:
        """Generate text using the completions API.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the OpenAI API.

        Returns:
            The generated text.
        """
        # Send request to OpenAI
        response = self.client.completions.create(model=self.model_name, prompt=prompt, **options)

        return response.choices[0].text.strip()

    def _is_chat_model(self) -> bool:
        """Determine if the model is a chat model.

        Returns:
            True if the model is a chat model, False otherwise.
        """
        # Most OpenAI models starting with "gpt" are chat models
        return "gpt" in self.model_name.lower()

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
            # Get the encoding for the model
            encoding = self._get_encoding()

            # Count tokens
            return len(encoding.encode(text))
        except Exception as e:
            raise ModelError(f"Error counting tokens: {str(e)}")

    def _get_encoding(self):
        """Get the encoding for the model.

        Returns:
            The encoding for the model.

        Raises:
            ModelError: If the encoding for the model cannot be determined.
        """
        try:
            # Try to get the encoding for the specific model
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fall back to cl100k_base, which is used by gpt-4 and gpt-3.5-turbo
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                raise ModelError(f"Error getting encoding: {str(e)}")


@register_model("openai")
def create_openai_model(model_name: str, **options: Any) -> Model:
    """Create an OpenAI model.

    Args:
        model_name: The name of the OpenAI model to use.
        **options: Additional options to pass to the OpenAI model constructor.

    Returns:
        An OpenAI model instance.

    Raises:
        ConfigurationError: If the OpenAI package is not installed.
        ModelError: If the API key is not provided and not available in the environment.
    """
    return OpenAIModel(model_name, **options)
