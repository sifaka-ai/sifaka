"""
Anthropic model provider for the Sifaka library.

This module provides an Anthropic model provider implementation.
"""

from typing import Dict, Any, Optional, Union, List


class AnthropicProvider:
    """
    Anthropic model provider implementation.

    This class implements the ModelProvider protocol for Anthropic models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Anthropic model provider.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to the Anthropic API
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Initialize the Anthropic client lazily to avoid errors if Anthropic is not installed
        self._client = None

    @property
    def client(self):
        """Get the Anthropic client, initializing it if needed."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
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

        # Generate
        response = self.client.messages.create(
            messages=[{"role": "user", "content": prompt}], **params
        )

        # Extract text
        return response.content[0].text
