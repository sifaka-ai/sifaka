"""
OpenAI model provider for the Sifaka library.

This module provides an OpenAI model provider implementation.
"""

from typing import Dict, Any, Optional, Union, List


class OpenAIProvider:
    """
    OpenAI model provider implementation.

    This class implements the ModelProvider protocol for OpenAI models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI model provider.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to the OpenAI API
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Initialize the OpenAI client lazily to avoid errors if OpenAI is not installed
        self._client = None

    @property
    def client(self):
        """Get the OpenAI client, initializing it if needed."""
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional generation parameters that override the defaults

        Returns:
            Generated text
        """
        # Merge kwargs with defaults
        params = {
            "model": self.model,
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        params.update(self.kwargs)
        params.update(kwargs)

        # Create messages
        messages = [{"role": "user", "content": prompt}]

        # Generate
        response = self.client.chat.completions.create(messages=messages, **params)

        # Extract text
        return response.choices[0].message.content
