"""
OpenAI model provider implementation.
"""

from typing import Optional, Dict, Any
import openai
from openai import OpenAI
import tiktoken
from sifaka.models.base import ModelProvider, ModelConfig, APIClient, TokenCounter
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


class OpenAIProvider(ModelProvider):
    """
    OpenAI model provider implementation.

    This provider supports GPT-3.5 and GPT-4 models with configurable parameters
    and built-in token counting.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """Initialize the OpenAI provider."""
        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )
        logger.info(f"Initialized OpenAI provider with model {model_name}")

    def _create_default_client(self) -> APIClient:
        """Create a default OpenAI client."""
        return OpenAIClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter for the current model."""
        return OpenAITokenCounter(model=self.model_name)
