"""
Google Gemini model provider implementation.

This module provides the GeminiProvider class which implements the ModelProviderCore
interface for Google Gemini models.
"""

from typing import Optional

import google.generativeai as genai
import tiktoken

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiClient(APIClient):
    """Gemini API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Gemini client."""
        genai.configure(api_key=api_key)
        self.model = None  # Lazy initialization
        logger.debug("Initialized Gemini client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to Gemini and return the response."""
        try:
            if self.model is None:
                self.model = genai.GenerativeModel(
                    model_name="gemini-pro",
                    generation_config={
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_tokens,
                    },
                )

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise


class GeminiTokenCounter(TokenCounter):
    """Token counter using tiktoken for Gemini models."""

    def __init__(self, model: str = "gemini-pro") -> None:
        """Initialize the token counter for a specific model."""
        try:
            # Gemini uses similar tokenization to GPT-3.5
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


class GeminiProvider(ModelProviderCore):
    """
    Google Gemini model provider implementation.

    This provider supports Gemini Pro models with configurable parameters
    and built-in token counting.
    """

    def __init__(
        self,
        model_name: str = "gemini-pro",
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the Gemini provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
        """
        # Verify Google Generative AI package is installed
        try:
            # Just importing the package to verify it's installed
            # We already imported it at the module level
            pass
        except ImportError:
            raise ImportError(
                "Google Generative AI package is required. Install with: pip install google-generativeai"
            )

        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )

    def _create_default_client(self) -> APIClient:
        """
        Create a default Gemini client.

        Returns:
            A default Gemini API client
        """
        return GeminiClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter for the current model.

        Returns:
            A default token counter for Gemini models
        """
        return GeminiTokenCounter(model=self.model_name)
