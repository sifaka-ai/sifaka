"""
Anthropic model provider for Sifaka.
"""

from typing import Dict, Any, Optional
from pydantic import Field, PrivateAttr
from .base import ModelProvider


class AnthropicProvider(ModelProvider):
    """
    Anthropic model provider for Sifaka.

    Attributes:
        model_name (str): The name of the Anthropic model to use
        api_key (Optional[str]): The Anthropic API key (if None, uses environment variable)
        temperature (float): The temperature to use for generation
        max_tokens (int): The maximum number of tokens to generate
        name (str): The name of the provider
        additional_kwargs (Dict[str, Any]): Additional arguments to pass to the Anthropic API
    """

    model_name: str = "claude-3-opus-20240229"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Private attributes
    _anthropic: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize an Anthropic provider.

        Args:
            model_name (str): The name of the Anthropic model to use
            api_key (Optional[str]): The Anthropic API key (if None, uses environment variable)
            temperature (float): The temperature to use for generation
            max_tokens (int): The maximum number of tokens to generate
            name (Optional[str]): The name of the provider
            **kwargs: Additional arguments to pass to the Anthropic API
        """
        # Set up initial values
        init_data = {
            "model_name": model_name,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "additional_kwargs": kwargs,
        }

        # Set name
        if name is not None:
            init_data["name"] = name
        else:
            init_data["name"] = f"anthropic_{model_name}"

        # Initialize the model
        super().__init__(**init_data)

        # Lazy import to avoid dependency issues
        try:
            import anthropic

            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Please install it with `pip install anthropic`"
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the Anthropic API.

        Args:
            prompt (str): The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the Anthropic API

        Returns:
            str: The generated text
        """
        # Merge kwargs with defaults
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_kwargs,
            **kwargs,
        }

        try:
            client = self._anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                messages=[{"role": "user", "content": prompt}], **params
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Error generating text with Anthropic: {e}")
