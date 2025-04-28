"""
OpenAI model provider implementation.
"""

from typing import Optional, Any

import openai
import tiktoken
from openai import OpenAI

from sifaka.models.base import APIClient, ModelConfig, ModelProvider, TokenCounter
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient(APIClient):
    """OpenAI API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=api_key)
        logger.debug("Initialized OpenAI client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to OpenAI and return the response."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise


class OpenAITokenCounter(TokenCounter):
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """Initialize the token counter for a specific model."""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            logger.debug(f"Initialized token counter for model {model}")
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in the text using the model's encoding."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise


class OpenAIProvider:
    """Provider for OpenAI language models."""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        config: Optional[ModelConfig] = None,
    ) -> None:
        """Initialize the OpenAI provider."""
        self._model_name = model_name
        self._config = config or ModelConfig()

        # Initialize OpenAI client
        try:
            self._client = OpenAI(api_key=self._config.api_key)
            logger.info(f"Initialized OpenAI provider with model {model_name}")
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {e}")

    def invoke(self, prompt: str) -> Any:
        """Invoke the model with a prompt and return structured output."""
        return self.generate(prompt)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._config
