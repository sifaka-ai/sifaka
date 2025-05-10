"""
OpenAI model provider for Sifaka.

This module provides integration with OpenAI's language models,
enabling text generation and completion capabilities.

## Overview
The OpenAI provider connects to OpenAI's API for text generation, offering access
to models like GPT-4, GPT-3.5-Turbo, and others. It handles authentication,
API communication, token counting, response processing, and execution tracking.

## Components
- **OpenAIProvider**: Main provider class for OpenAI models
- **OpenAIClient**: API client implementation for OpenAI
- **OpenAITokenCounter**: Token counter implementation for OpenAI models

## Usage Examples
```python
from sifaka.models.providers.openai import OpenAIProvider
from sifaka.models.config import ModelConfig

# Create a provider with default configuration
provider = OpenAIProvider(model_name="gpt-4")

# Create a provider with custom configuration
config = ModelConfig(
    temperature=0.8,
    max_tokens=2000,
    api_key="your-api-key"
)
provider = OpenAIProvider(model_name="gpt-4", config=config)

# Generate text
response = provider.generate("Explain quantum computing")

# Count tokens
token_count = provider.count_tokens("How many tokens is this?")
```

## Error Handling
The provider implements comprehensive error handling:
- API authentication errors
- Rate limiting and quota errors
- Network and timeout errors
- Model-specific errors
- Input validation errors
"""

import os
from typing import Any, Dict, Optional, ClassVar

import tiktoken

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.errors import handle_error
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient(APIClient):
    """OpenAI API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the OpenAI client.

        Args:
            api_key: Optional API key for OpenAI
        """
        # Check for API key in environment if not provided
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            logger.debug("Retrieved API key from environment")

        # Validate API key
        if not api_key:
            logger.warning(
                "No OpenAI API key provided and OPENAI_API_KEY environment variable not set"
            )
        elif not api_key.startswith("sk-"):
            logger.warning(
                f"API key format appears incorrect. Expected to start with 'sk-', got: {api_key[:5]}..."
            )

        self.api_key = api_key
        logger.debug("Initialized OpenAI client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """
        Send a prompt to OpenAI and return the response.

        Args:
            prompt: The prompt to send
            config: Configuration for the request

        Returns:
            The generated text response

        Raises:
            ValueError: If API key is missing
            RuntimeError: If the API call fails
        """
        # Get API key from config or client
        api_key = config.api_key or self.api_key

        # Check for missing API key
        if not api_key:
            raise ValueError(
                "No API key provided. Please provide an API key either by setting the "
                "OPENAI_API_KEY environment variable or by passing it explicitly."
            )

        try:
            # Import OpenAI here to avoid dependency issues
            import openai

            # Configure client
            openai.api_key = api_key

            # Prepare generation parameters
            params = {
                "model": config.params.get("model_name", "gpt-4"),
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.params.get("top_p", 1.0),
                "frequency_penalty": config.params.get("frequency_penalty", 0.0),
                "presence_penalty": config.params.get("presence_penalty", 0.0),
                "stop": config.params.get("stop", None),
            }

            # Generate text
            response = openai.Completion.create(**params)
            return response.choices[0].text.strip()

        except Exception as e:
            error_info = handle_error(e, "OpenAIClient")
            logger.error(f"OpenAI API error: {error_info['error_message']}")
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e


class OpenAITokenCounter(TokenCounter):
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str = "gpt-4") -> None:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model to count tokens for
        """
        try:
            # Get the appropriate encoding for the model
            if "gpt-4" in model:
                encoding_name = "cl100k_base"
            elif "gpt-3.5" in model:
                encoding_name = "cl100k_base"
            else:
                encoding_name = "p50k_base"  # Default for older models

            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.debug(
                f"Initialized token counter for model {model} with encoding {encoding_name}"
            )
        except Exception as e:
            error_info = handle_error(e, "OpenAITokenCounter")
            logger.error(f"Error initializing token counter: {error_info['error_message']}")
            raise RuntimeError(f"Failed to initialize token counter: {str(e)}") from e

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text using the model's encoding.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            error_info = handle_error(e, "OpenAITokenCounter")
            logger.error(f"Error counting tokens: {error_info['error_message']}")
            raise RuntimeError(f"Failed to count tokens: {str(e)}") from e


class OpenAIProvider(ModelProviderCore):
    """
    OpenAI model provider implementation.

    This provider supports OpenAI models with configurable parameters,
    built-in token counting, and execution tracking. It handles communication
    with OpenAI's API, token counting, and response processing.

    ## Architecture
    OpenAIProvider extends ModelProviderCore and follows Sifaka's component-based
    architecture. It delegates API communication to OpenAIClient and token counting
    to OpenAITokenCounter.
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "gpt-4"

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
            import importlib.util

            if importlib.util.find_spec("openai") is None:
                raise ImportError()
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        # Initialize with ModelProviderCore
        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        This method is needed for compatibility with the critique service
        which expects an 'invoke' method.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        return self.generate(prompt, **kwargs)

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously invoke the model with a prompt.

        This method delegates to agenerate if it exists, or falls back to
        synchronous generate.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        if hasattr(self, "agenerate"):
            return await self.agenerate(prompt, **kwargs)

        # Fall back to synchronous generate
        return self.generate(prompt, **kwargs)

    def _create_default_client(self) -> APIClient:
        """Create a default OpenAI client."""
        return OpenAIClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter for the current model."""
        return OpenAITokenCounter(model=self.model_name)

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return f"OpenAI-{self.model_name}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about provider usage.

        Returns:
            Dictionary with usage statistics
        """
        # Get statistics from tracing manager
        return self._tracing_manager.get_statistics()
