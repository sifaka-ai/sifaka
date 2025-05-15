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
import time
from typing import Optional, Dict, Any, ClassVar

import tiktoken
import google.generativeai as genai

from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.utils.config.models import ModelConfig
from sifaka.models.core.provider import ModelProviderCore
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
            if logger:
                logger.debug("Retrieved API key from environment")

        # Validate API key
        if not api_key:
            if logger:
                logger.warning(
                    "No Google API key provided and GOOGLE_API_KEY environment variable not set"
                )

        # Configure the client
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model: Optional[genai.GenerativeModel] = None  # Lazy initialization
        if logger:
            logger.debug("Initialized Gemini client")

    def send_prompt(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a prompt to Gemini and return the response.

        Args:
            prompt: The prompt to send
            params: Parameters for the request including model configuration

        Returns:
            The API response as a dictionary

        Raises:
            ValueError: If API key is missing
            RuntimeError: If the API call fails
        """
        # Get API key from params or client
        api_key = params.get("api_key", self.api_key)

        # Check for missing API key
        if not api_key:
            raise ValueError(
                "No API key provided. Please provide an API key either by setting the "
                "GOOGLE_API_KEY environment variable or by passing it explicitly."
            )

        try:
            # Initialize model if needed
            if self.model is None:
                model_name = params.get("model_name", "gemini-pro")

                # Create generation config from params
                generation_config = genai.types.GenerationConfig(
                    temperature=params.get("temperature", 0.7),
                    max_output_tokens=params.get("max_tokens", 1024),
                    top_p=params.get("top_p", 0.95),
                    top_k=params.get("top_k", 40),
                )

                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                )

            # Generate text
            response = self.model.generate_content(prompt) if self.model else None

            # Convert response to dictionary format
            result = {
                "choices": [
                    {
                        "text": response.text if response else "",
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "created": int(time.time()),
                "model": params.get("model_name", "gemini-pro"),
                "object": "text_completion",
                "usage": {
                    "prompt_tokens": 0,  # Will be updated by token counter
                    "completion_tokens": 0,  # Will be updated by token counter
                    "total_tokens": 0,  # Will be updated by token counter
                },
            }

            return result

        except Exception as e:
            error_info = handle_error(e, "GeminiClient")
            if logger:
                logger.error(f"Gemini API error: {error_info['error_message']}")
            raise RuntimeError(f"Gemini API error: {str(e)}") from e


class GeminiTokenCounter(TokenCounter):
    """Token counter using tiktoken for Gemini models."""

    def __init__(self, model: Optional[str] = "gemini-pro") -> None:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model to count tokens for
        """
        try:
            # Gemini uses similar tokenization to GPT-3.5/GPT-4
            self.encoding = tiktoken.get_encoding("cl100k_base")
            if logger:
                logger.debug(f"Initialized token counter for model {model}")
        except Exception as e:
            error_info = handle_error(e, "GeminiTokenCounter")
            if logger:
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
            return len(self.encoding.encode(text)) if hasattr(self, "encoding") else 0
        except Exception as e:
            error_info = handle_error(e, "GeminiTokenCounter")
            if logger:
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
        if self._state_manager:
            self._state_manager.update("stats", stats)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
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
            stats = self._state_manager.get("stats", {}) if self._state_manager else {}
            stats["generation_count"] = stats.get("generation_count", 0) + 1 if stats else 1
            stats["total_processing_time"] = (
                stats.get("total_processing_time", 0) + ((time.time() - start_time) * 1000)
                if stats
                else (time.time() - start_time) * 1000
            )
            if self._state_manager:
                self._state_manager.update("stats", stats)

            return result

        except Exception:
            # Update error count in state
            stats = self._state_manager.get("stats", {}) if self._state_manager else {}
            stats["error_count"] = stats.get("error_count", 0) + 1 if stats else 1
            if self._state_manager:
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
        config = self._state_manager.get("config") if self._state_manager else None
        api_key = config.api_key if config else None
        return GeminiClient(api_key=api_key)

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
        model_name = self._state_manager.get("model_name") if self._state_manager else ""
        return GeminiTokenCounter(model=model_name)

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        model_name = self._state_manager.get("model_name") if self._state_manager else ""
        return f"Gemini-{model_name}"

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
            provider.generate("Hello, world!") if provider else ""
            provider.count_tokens("How many tokens?") if provider else ""

            # Get usage statistics
            stats = provider.get_statistics() if provider else ""
            print(f"Generation count: {stats['generation_count']}")
            print(f"Token count calls: {stats['token_count_calls']}")
            ```
        """
        # Get statistics from tracing manager and state
        tracing_manager = self._state_manager and self._state_manager.get("tracing_manager")
        tracing_stats = tracing_manager.get_statistics() if tracing_manager else {}

        # Combine with any other stats from state
        stats = self._state_manager and self._state_manager.get("stats", {}) or {}

        return {**tracing_stats, **stats}

    @property
    def description(self) -> str:
        """
        Get a description of the provider.

        Returns:
            str: A description of the provider
        """
        model_name = self._state_manager and self._state_manager.get("model_name") or ""
        return f"Gemini provider using model {model_name}"

    def update_config(self, config: Any = None, **kwargs: Any) -> None:
        """
        Update the provider configuration.

        Args:
            config: New configuration (not used, included for compatibility with interface)
            **kwargs: Configuration parameters to update
        """
        config = self._state_manager and self._state_manager.get("config")
        if not config:
            return

        # Create a new config with updated values using the proper immutable pattern
        # First, check if any kwargs match direct config attributes
        config_kwargs: Dict[str, Any] = {}
        params_kwargs: Dict[str, Any] = {}

        for key, value in kwargs.items():
            if hasattr(config, key) and key != "params":
                config_kwargs[key] = value
            else:
                params_kwargs[key] = value

        # Create updated config using with_options for direct attributes
        if config_kwargs:
            new_config = config.with_options(**config_kwargs)
        else:
            new_config = config

        # Add any params using with_params
        if params_kwargs:
            new_config = new_config.with_params(**params_kwargs)

        # Update state
        if self._state_manager:
            self._state_manager.update("config", new_config)
