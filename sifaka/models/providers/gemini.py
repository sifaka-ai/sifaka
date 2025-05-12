"""
Google Gemini model provider implementation.

This module provides the GeminiProvider class which implements the ModelProviderCore
interface for Google Gemini models.

## Overview
The Gemini provider connects to Google's Generative AI API for text generation, offering access
to models like Gemini Pro. It handles authentication, API communication, token counting,
response processing, and execution tracking.

## Components
- **GeminiProvider**: Main provider class for Google Gemini models
- **GeminiClient**: API client implementation for Google Gemini
- **GeminiTokenCounter**: Token counter implementation for Gemini models

## Usage Examples
```python
from sifaka.models.providers.gemini import GeminiProvider
from sifaka.utils.config.models import ModelConfig

# Create a provider with default configuration
provider = GeminiProvider(model_name="gemini-pro")

# Create a provider with custom configuration
config = ModelConfig(
    temperature=0.8,
    max_tokens=2000,
    api_key="your-api-key"
)
provider = GeminiProvider(model_name="gemini-pro", config=config)

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
from typing import Optional, Dict, Any, ClassVar

import tiktoken
import importlib.util
import google.generativeai as genai

from sifaka.models.base import APIClient, TokenCounter
from sifaka.utils.config.models import ModelConfig
from sifaka.models.core import ModelProviderCore
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiClient(APIClient):
    """Gemini API client implementation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the Gemini client.

        Args:
            api_key: Optional API key for Google Gemini
        """
        # Check for API key in environment if not provided
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
            logger.debug("Retrieved API key from environment")

        # Validate API key
        if not api_key:
            logger.warning(
                "No Google API key provided and GOOGLE_API_KEY environment variable not set"
            )

        # Configure the client
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model = None  # Lazy initialization
        logger.debug("Initialized Gemini client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """
        Send a prompt to Gemini and return the response.

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
                "GOOGLE_API_KEY environment variable or by passing it explicitly."
            )

        try:
            # Initialize model if needed
            if self.model is None:
                model_name = config.params.get("model_name", "gemini-pro")
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_tokens,
                        "top_p": config.params.get("top_p", 0.95),
                        "top_k": config.params.get("top_k", 40),
                    },
                )

            # Generate text
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            error_info = handle_error(e, "GeminiClient")
            logger.error(f"Gemini API error: {error_info['error_message']}")
            raise RuntimeError(f"Gemini API error: {str(e)}") from e


class GeminiTokenCounter(TokenCounter):
    """Token counter using tiktoken for Gemini models."""

    def __init__(self, model: str = "gemini-pro") -> None:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model to count tokens for
        """
        try:
            # Gemini uses similar tokenization to GPT-3.5/GPT-4
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug(f"Initialized token counter for model {model}")
        except Exception as e:
            error_info = handle_error(e, "GeminiTokenCounter")
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
            error_info = handle_error(e, "GeminiTokenCounter")
            logger.error(f"Error counting tokens: {error_info['error_message']}")
            raise RuntimeError(f"Failed to count tokens: {str(e)}") from e


class GeminiProvider(ModelProviderCore):
    """
    Google Gemini model provider implementation.

    This provider supports Gemini Pro models with configurable parameters,
    built-in token counting, and execution tracking. It handles communication
    with Google's Generative AI API, token counting, and response processing.

    ## Architecture
    GeminiProvider extends ModelProviderCore and follows Sifaka's component-based
    architecture. It delegates API communication to GeminiClient and token counting
    to GeminiTokenCounter. The provider uses standardized state management through
    the StateManager from utils/state.py.

    ## Lifecycle
    1. Initialization: Creates client and token counter managers
    2. Warm-up: Initializes API client and token counter
    3. Operation: Handles text generation and token counting
    4. Cleanup: Releases resources when no longer needed

    ## Error Handling
    The provider implements comprehensive error handling:
    - API authentication errors
    - Rate limiting and quota errors
    - Network and timeout errors
    - Model-specific errors
    - Input validation errors

    ## Examples
    ```python
    from sifaka.models.providers.gemini import GeminiProvider
    from sifaka.utils.config.models import ModelConfig

    # Create a provider with default configuration
    provider = GeminiProvider(model_name="gemini-pro")

    # Create a provider with custom configuration
    config = ModelConfig(
        temperature=0.8,
        max_tokens=2000,
        api_key="your-api-key"
    )
    provider = GeminiProvider(model_name="gemini-pro", config=config)

    # Generate text
    response = provider.generate("Explain quantum computing")

    # Count tokens
    token_count = provider.count_tokens("How many tokens is this?")
    ```

    Attributes:
        _state_manager (StateManager): Manages provider state
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "gemini-pro"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
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
            import importlib.util

            if importlib.util.find_spec("google.generativeai") is None:
                raise ImportError()
        except ImportError:
            raise ImportError(
                "Google Generative AI package is required. Install with: pip install google-generativeai"
            )

        # Initialize with ModelProviderCore
        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )

        # Initialize provider-specific stats
        stats = {
            "generation_count": 0,
            "token_count_calls": 0,
            "error_count": 0,
            "total_processing_time": 0,
        }
        self._state_manager.update("stats", stats)

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        This method sends a prompt to the Gemini API and returns the generated text.
        It delegates to the generate method and tracks statistics about the operation.
        This method is needed for compatibility with the critique service which expects
        an 'invoke' method.

        The invocation process includes:
        1. Tracking the start time
        2. Calling the generate method with the prompt and kwargs
        3. Updating statistics in the state manager
        4. Returning the generated text

        Args:
            prompt (str): The prompt to send to the model
            **kwargs: Additional keyword arguments to override configuration
                - temperature (float): Controls randomness (0.0-1.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - top_k (int): Top-k sampling parameter

        Returns:
            str: The generated text response

        Raises:
            RuntimeError: If generation fails due to API issues
            ValueError: If invalid configuration is provided

        Example:
            ```python
            provider = GeminiProvider(model_name="gemini-pro")

            # Basic invocation
            response = provider.invoke("Explain quantum computing")

            # Invocation with configuration overrides
            response = provider.invoke(
                "Write a poem about AI",
                temperature=0.9,
                max_tokens=200
            )
            ```
        """
        # Track generation count in state
        import time

        start_time = time.time()

        try:
            result = self.generate(prompt, **kwargs)

            # Update statistics in state
            stats = self._state_manager.get("stats", {})
            stats["generation_count"] = stats.get("generation_count", 0) + 1
            stats["total_processing_time"] = (
                stats.get("total_processing_time", 0) + (time.time() - start_time) * 1000
            )
            self._state_manager.update("stats", stats)

            return result

        except Exception:
            # Update error count in state
            stats = self._state_manager.get("stats", {})
            stats["error_count"] = stats.get("error_count", 0) + 1
            self._state_manager.update("stats", stats)

            # Re-raise the exception
            raise

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously invoke the model with a prompt.

        This method provides asynchronous invocation of the model. It delegates to
        agenerate if it exists, or falls back to synchronous generate. It also tracks
        statistics about the operation.

        The async invocation process includes:
        1. Tracking the start time
        2. Calling agenerate if available, otherwise falling back to generate
        3. Updating statistics in the state manager
        4. Returning the generated text

        Args:
            prompt (str): The prompt to send to the model
            **kwargs: Additional keyword arguments to override configuration
                - temperature (float): Controls randomness (0.0-1.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - top_k (int): Top-k sampling parameter

        Returns:
            str: The generated text response

        Raises:
            RuntimeError: If generation fails due to API issues
            ValueError: If invalid configuration is provided

        Example:
            ```python
            import asyncio

            async def generate_text():
                provider = GeminiProvider(model_name="gemini-pro")
                response = await provider.ainvoke("Explain quantum computing")
                return response

            # Run the async function
            response = asyncio.run(generate_text())
            ```
        """
        # Track generation count in state
        import time

        start_time = time.time()

        try:
            if hasattr(self, "agenerate"):
                result = await self.agenerate(prompt, **kwargs)
            else:
                # Fall back to synchronous generate
                result = self.generate(prompt, **kwargs)

            # Update statistics in state
            stats = self._state_manager.get("stats", {})
            stats["generation_count"] = stats.get("generation_count", 0) + 1
            stats["total_processing_time"] = (
                stats.get("total_processing_time", 0) + (time.time() - start_time) * 1000
            )
            self._state_manager.update("stats", stats)

            return result

        except Exception:
            # Update error count in state
            stats = self._state_manager.get("stats", {})
            stats["error_count"] = stats.get("error_count", 0) + 1
            self._state_manager.update("stats", stats)

            # Re-raise the exception
            raise

    def _create_default_client(self) -> APIClient:
        """
        Create a default Gemini client.

        This method creates and returns a new GeminiClient instance with the API key
        from the provider's configuration. It's called by the ModelProviderCore
        when no custom client is provided.

        Returns:
            APIClient: A new GeminiClient instance

        Raises:
            ImportError: If the Google Generative AI package is not installed
            ValueError: If no API key is available
        """
        return GeminiClient(api_key=self._state_manager.get("config").api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter for the current model.

        This method creates and returns a new GeminiTokenCounter instance for the
        provider's model. It's called by the ModelProviderCore when no custom
        token counter is provided.

        Returns:
            TokenCounter: A new GeminiTokenCounter instance

        Raises:
            ImportError: If the tiktoken package is not installed
            RuntimeError: If token counter initialization fails
        """
        return GeminiTokenCounter(model=self._state_manager.get("model_name"))

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return f"Gemini-{self._state_manager.get('model_name')}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about provider usage.

        This method returns a dictionary with statistics about the provider's usage,
        including generation count, token count calls, error count, and processing time.

        The statistics include:
        - generation_count: Number of text generation calls
        - token_count_calls: Number of token counting calls
        - error_count: Number of errors encountered
        - total_processing_time: Total processing time in milliseconds
        - Any additional statistics from the tracing manager

        Returns:
            Dict[str, Any]: Dictionary with usage statistics

        Example:
            ```python
            provider = GeminiProvider(model_name="gemini-pro")
            provider.generate("Hello, world!")
            provider.count_tokens("How many tokens?")

            # Get usage statistics
            stats = provider.get_statistics()
            print(f"Generation count: {stats['generation_count']}")
            print(f"Token count calls: {stats['token_count_calls']}")
            ```
        """
        # Get statistics from tracing manager and state
        tracing_manager = self._state_manager.get("tracing_manager")
        tracing_stats = tracing_manager.get_statistics()

        # Combine with any other stats from state
        stats = self._state_manager.get("stats", {})

        return {**tracing_stats, **stats}
