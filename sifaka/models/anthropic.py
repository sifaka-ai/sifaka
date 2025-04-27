"""
Anthropic model provider implementation.
"""

from typing import Optional, Dict, Any
import anthropic
from anthropic import Anthropic
import tiktoken
from langchain_anthropic import ChatAnthropic
from sifaka.models.base import ModelProvider, ModelConfig, APIClient, TokenCounter
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicClient(APIClient):
    """Anthropic API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=api_key)
        logger.debug("Initialized Anthropic client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to Anthropic and return the response."""
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            return response.content[0].text
        except anthropic.AnthropicError as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise


class AnthropicTokenCounter(TokenCounter):
    """Token counter using tiktoken for Anthropic models."""

    def __init__(self, model: str = "claude-3-opus-20240229") -> None:
        """Initialize the token counter for a specific model."""
        try:
            # Anthropic uses cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug(f"Initialized token counter for model {model}")
        except Exception as e:
            logger.error(f"Error initializing token counter: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in the text using the model's encoding."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise


class AnthropicProvider(ModelProvider):
    """
    Anthropic model provider implementation.

    This provider supports Claude models with configurable parameters
    and built-in token counting.
    """

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """Initialize the Anthropic provider."""
        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )
        logger.info(f"Initialized Anthropic provider with model {model_name}")

    def _create_default_client(self) -> APIClient:
        """Create a default Anthropic client."""
        return AnthropicClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter for the current model."""
        return AnthropicTokenCounter(model=self.model_name)

    def get_langchain_llm(self) -> ChatAnthropic:
        """Get a LangChain ChatAnthropic instance for this provider."""
        return ChatAnthropic(
            model_name=self.model_name,
            anthropic_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
