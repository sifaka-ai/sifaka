"""
OpenAI model provider for Sifaka.
"""

from typing import Dict, Any, Optional, ClassVar
from pydantic import Field, PrivateAttr
from .base import ModelProvider


class OpenAIProvider(ModelProvider):
    """
    OpenAI model provider for Sifaka.

    Attributes:
        model_name (str): The name of the OpenAI model to use
        api_key (Optional[str]): The OpenAI API key (if None, uses environment variable)
        temperature (float): The temperature to use for generation
        max_tokens (int): The maximum number of tokens to generate
        name (str): The name of the provider
        additional_kwargs (Dict[str, Any]): Additional arguments to pass to the OpenAI API
    """

    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Private attributes
    _openai: ClassVar[Any] = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize an OpenAI provider.

        Args:
            model_name (str): The name of the OpenAI model to use
            api_key (Optional[str]): The OpenAI API key (if None, uses environment variable)
            temperature (float): The temperature to use for generation
            max_tokens (int): The maximum number of tokens to generate
            name (Optional[str]): The name of the provider
            **kwargs: Additional arguments to pass to the OpenAI API
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
            init_data["name"] = f"openai_{model_name}"

        # Initialize the model
        super().__init__(**init_data)

        # Lazy import to avoid dependency issues
        try:
            import openai

            self._openai = openai

            if api_key:
                self._openai.api_key = api_key
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install it with `pip install openai`"
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the OpenAI API.

        Args:
            prompt (str): The prompt to send to the LLM
            **kwargs: Additional arguments to pass to the OpenAI API

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

        # Handle different OpenAI API versions
        try:
            # New API (openai>=1.0.0)
            client = self._openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], **params
            )
            return response.choices[0].message.content
        except (AttributeError, TypeError):
            try:
                # Legacy API (openai<1.0.0)
                response = self._openai.ChatCompletion.create(
                    messages=[{"role": "user", "content": prompt}], **params
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"Error generating text with OpenAI: {e}")
