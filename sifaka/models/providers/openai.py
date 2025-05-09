"""
OpenAI Model Provider

This module provides the OpenAIProvider class which implements the ModelProviderCore
interface for OpenAI models, enabling integration with OpenAI's API services.

## Overview
The OpenAI provider connects to OpenAI's API for text generation, offering access
to models like GPT-4, GPT-3.5-Turbo, and others. It handles authentication, API
communication, token counting, and response processing.

## Components
- **OpenAIProvider**: Main provider class for OpenAI models
- **OpenAIClient**: API client implementation for OpenAI
- **OpenAITokenCounter**: Token counter implementation for OpenAI models
"""

from typing import Optional, Any, Dict, ClassVar

import openai
import tiktoken
from openai import OpenAI

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
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


class OpenAIProvider(ModelProviderCore):
    """
    OpenAI model provider implementation.

    This provider supports OpenAI models with configurable parameters
    and built-in token counting. It handles communication with OpenAI's API,
    token counting, and response processing.

    ## Architecture
    OpenAIProvider extends ModelProviderCore and follows Sifaka's component-based
    architecture. It delegates API communication to OpenAIClient and token counting
    to OpenAITokenCounter.

    ## Lifecycle
    1. **Initialization**: Provider is created with model name and configuration
    2. **Client Creation**: API client is created on first use
    3. **Token Counter Creation**: Token counter is created on first use
    4. **Generation**: Text is generated using the model
    5. **Token Counting**: Tokens are counted for input text
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "gpt-4-turbo"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
        """
        # Verify OpenAI package is installed
        try:
            # Just importing the package to verify it's installed
            # We already imported it at the module level
            pass
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        # Initialize parent class first
        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )

    def _create_default_client(self) -> APIClient:
        """Create a default OpenAI client."""
        return OpenAIClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter for the current model."""
        return OpenAITokenCounter(model=self.model_name)

    def invoke(self, prompt: str) -> Any:
        """
        Invoke the model with a prompt and return structured output.

        This is a convenience method that delegates to generate.

        Args:
            prompt: The prompt to generate from

        Returns:
            The generated text
        """
        return self.generate(prompt)
