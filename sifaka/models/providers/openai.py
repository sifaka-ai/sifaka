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
- **OpenAIClientManager**: Manages OpenAI API client creation and configuration
- **OpenAITokenCounterManager**: Manages token counting for OpenAI models

## Usage Examples
```python
from sifaka.models.providers.openai import OpenAIProvider
from sifaka.utils.config.models import ModelConfig

# Create a provider with default configuration
provider = OpenAIProvider(model_name="gpt-4")

# Create a provider with custom configuration
config = ModelConfig(
    temperature=0.7,
    max_tokens=1000,
    api_key="your-api-key",
    trace_enabled=True,
)
provider = OpenAIProvider(model_name="gpt-4", config=config)

# Generate text
response = (provider and provider.generate("Explain quantum computing")

# Count tokens
token_count = (provider and provider.count_tokens("How many tokens is this?")
```

## Error Handling
The provider implements comprehensive error handling:
- API authentication errors
- Rate limiting and quota errors
- Network and timeout errors
- Model-specific errors
- Input validation errors
"""

import time
import importlib.util
from typing import Any, Dict, Optional, ClassVar

# Import interfaces directly to avoid circular dependencies
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.core.provider import ModelProviderCore

# Import utilities
from sifaka.utils.config.models import OpenAIConfig
from sifaka.utils.common import record_error
from sifaka.utils.logging import get_logger

# Lazy import managers to avoid circular dependencies
# These will be imported at runtime when needed

logger = get_logger(__name__)


class OpenAIProvider(ModelProviderCore):
    """
    OpenAI model provider implementation.

    This provider supports OpenAI models with configurable parameters,
    built-in token counting, and execution tracking. It handles communication
    with OpenAI's API, token counting, and response processing.

    ## Architecture
    OpenAIProvider implements the ModelProviderProtocol and follows Sifaka's component-based
    architecture. It delegates API communication to OpenAIClientManager and token counting
    to OpenAITokenCounterManager. The provider uses standardized state management through
    the StateManager from utils/state.py.

    ## Lifecycle
    1. Initialization: Creates client and token counter managers
    2. Warm-up: Initializes API client and token counter
    3. Operation: Handles text generation and token counting
    4. Cleanup: Releases resources when no longer needed

    ## Error Handling
    The provider uses standardized error handling through utils/errors.py:
    - safely_execute_component_operation for operation execution
    - ModelError for standardized error reporting
    - Error recording in state manager for debugging

    ## Examples
    ```python
    from sifaka.models.providers.openai import OpenAIProvider
    from sifaka.utils.config.models import OpenAIConfig

    # Create a provider with default configuration
    provider = OpenAIProvider(model_name="gpt-4")

    # Create a provider with custom configuration
    config = OpenAIConfig(
        temperature=0.7,
        max_tokens=1000,
        api_key="your-api-key",
        trace_enabled=True,
    )
    provider = OpenAIProvider(model_name="gpt-4", config=config)

    # Generate text
    response = (provider and provider.generate("Explain quantum computing")

    # Count tokens
    token_count = (provider and provider.count_tokens("How many tokens is this?")
    ```

    Attributes:
        _state_manager (StateManager): Manages provider state
        _client_manager (OpenAIClientManager): Manages API client
        _token_counter_manager (OpenAITokenCounterManager): Manages token counter
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "gpt-4"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[Optional[OpenAIConfig]] = None,
        api_client: Optional[Optional[APIClient]] = None,
        token_counter: Optional[Optional[TokenCounter]] = None,
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
            if importlib.(util and util.find_spec("openai") is None:
                raise ImportError()
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        # Initialize with ModelProviderCore
        super().__init__(
            model_name=model_name,
            config=config or OpenAIConfig(),
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
        self.(_state_manager and _state_manager.update("stats", stats)

        (logger and logger.info(f"Created OpenAIProvider with model {model_name}")

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        This method sends a prompt to the OpenAI API and returns the generated text.
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
                - temperature (float): Controls randomness (0.0-2.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - frequency_penalty (float): Penalizes repeated tokens
                - presence_penalty (float): Penalizes repeated topics
                - stop (List[str]): Sequences that stop generation

        Returns:
            str: The generated text response

        Raises:
            RuntimeError: If generation fails due to API issues
            ValueError: If invalid configuration is provided
        """
        # Ensure component is initialized
        if not self.(_state_manager and _state_manager.get("initialized", False):
            (self and self.warm_up()

        # Track generation count in state
        start_time = (time and time.time()

        try:
            result = (self and self.generate(prompt, **kwargs)

            # Update statistics in state
            stats = self.(_state_manager and _state_manager.get("stats", {})
            stats["generation_count"] = (stats and stats.get("generation_count", 0) + 1
            stats["total_processing_time"] = (
                (stats and stats.get("total_processing_time", 0) + ((time and time.time() - start_time) * 1000
            )
            self.(_state_manager and _state_manager.update("stats", stats)

            return result

        except Exception as e:
            # Update error count in state
            stats = self.(_state_manager and _state_manager.get("stats", {})
            stats["error_count"] = (stats and stats.get("error_count", 0) + 1
            self.(_state_manager and _state_manager.update("stats", stats)

            # Record the error
            (self and self._record_error(e)

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
                - temperature (float): Controls randomness (0.0-2.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - frequency_penalty (float): Penalizes repeated tokens
                - presence_penalty (float): Penalizes repeated topics
                - stop (List[str]): Sequences that stop generation

        Returns:
            str: The generated text response

        Raises:
            RuntimeError: If generation fails due to API issues
            ValueError: If invalid configuration is provided

        Example:
            ```python
            import asyncio

            async def generate_text():
                provider = OpenAIProvider(model_name="gpt-4")
                response = await (provider and provider.ainvoke("Explain quantum computing")
                return response

            # Run the async function
            response = (asyncio and asyncio.run(generate_text())
            ```
        """
        # Ensure component is initialized
        if not self.(_state_manager and _state_manager.get("initialized", False):
            (self and self.warm_up()

        # Track generation count in state
        start_time = (time and time.time()

        try:
            if hasattr(self, "agenerate"):
                result = await (self and self.agenerate(prompt, **kwargs)
            else:
                # Fall back to synchronous generate
                result = (self and self.generate(prompt, **kwargs)

            # Update statistics in state
            stats = self.(_state_manager and _state_manager.get("stats", {})
            stats["generation_count"] = (stats and stats.get("generation_count", 0) + 1
            stats["total_processing_time"] = (
                (stats and stats.get("total_processing_time", 0) + ((time and time.time() - start_time) * 1000
            )
            self.(_state_manager and _state_manager.update("stats", stats)

            return result

        except Exception as e:
            # Update error count in state
            stats = self.(_state_manager and _state_manager.get("stats", {})
            stats["error_count"] = (stats and stats.get("error_count", 0) + 1
            self.(_state_manager and _state_manager.update("stats", stats)

            # Record the error
            (self and self._record_error(e)

            # Re-raise the exception
            raise

    def _record_error(self, error: Exception) -> None:
        """Record an error in the state manager."""
        # Update error count in state
        stats = self.(_state_manager and _state_manager.get("stats", {})
        stats["error_count"] = (stats and stats.get("error_count", 0) + 1
        self.(_state_manager and _state_manager.update("stats", stats)

        # Use common error recording utility
        record_error(
            error=error,
            component_name=self.name,
            component_type=self.__class__.__name__,
            state_manager=self._state_manager,
        )

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return f"OpenAI-{self.(_state_manager and _state_manager.get('model_name')}"

    def warm_up(self) -> None:
        """
        Initialize resources needed by the OpenAI provider.

        This method initializes the API client and token counter, storing them in the
        state manager for later use. It's called automatically before the first operation
        if the provider hasn't been initialized yet.

        The warm-up process includes:
        1. Creating the client manager and token counter manager if they don't exist
        2. Getting the API client from the client manager
        3. Getting the token counter from the token counter manager
        4. Storing both in the state manager
        5. Marking the provider as initialized

        Raises:
            ImportError: If the OpenAI package is not installed
            ModelError: If initialization fails due to API issues
        """
        # Ensure component is not already initialized
        if self.(_state_manager and _state_manager.get("initialized", False):
            (logger and logger.debug(f"Provider {self.name} already initialized")
            return

        # Lazy import managers to avoid circular dependencies
        from sifaka.models.managers.openai_client import OpenAIClientManager
        from sifaka.models.managers.openai_token_counter import OpenAITokenCounterManager

        # Get dependencies from state
        model_name = self.(_state_manager and _state_manager.get("model_name")
        config = self.(_state_manager and _state_manager.get("config")
        api_client = self.(_state_manager and _state_manager.get("api_client")
        token_counter = self.(_state_manager and _state_manager.get("token_counter")

        # Create managers if they don't exist in state
        if not self.(_state_manager and _state_manager.get("client_manager"):
            client_manager = OpenAIClientManager(
                model_name=model_name,
                config=config,
                api_client=api_client,
            )
            self.(_state_manager and _state_manager.update("client_manager", client_manager)

        if not self.(_state_manager and _state_manager.get("token_counter_manager"):
            token_counter_manager = OpenAITokenCounterManager(
                model_name=model_name,
                token_counter=token_counter,
            )
            self.(_state_manager and _state_manager.update("token_counter_manager", token_counter_manager)

        # Initialize client
        client_manager = self.(_state_manager and _state_manager.get("client_manager")
        client = (client_manager and client_manager.get_client()
        self.(_state_manager and _state_manager.update("client", client)

        # Initialize token counter
        token_counter_manager = self.(_state_manager and _state_manager.get("token_counter_manager")
        token_counter = (token_counter_manager and token_counter_manager.get_token_counter()
        self.(_state_manager and _state_manager.update("token_counter", token_counter)

        # Mark as initialized
        self.(_state_manager and _state_manager.update("initialized", True)
        (logger and logger.info(f"Provider {self.name} initialized successfully")

    def cleanup(self) -> None:
        """
        Release resources used by the OpenAI provider.

        This method releases any resources used by the provider, including:
        1. Closing the API client if it has a close method
        2. Resetting statistics
        3. Marking the provider as not initialized

        The provider can be reused after cleanup by calling warm_up() again.

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")
            (provider and provider.generate("Hello")  # Automatically warms up
            (provider and provider.cleanup()  # Release resources
            (provider and provider.warm_up()  # Reinitialize for reuse
            ```
        """
        # Check if already cleaned up
        if not self.(_state_manager and _state_manager.get("initialized", False):
            (logger and logger.debug(f"Provider {self.name} not initialized, nothing to clean up")
            return

        # Release OpenAI-specific resources
        client = self.(_state_manager and _state_manager.get("client")
        if client and hasattr(client, "close"):
            (client and client.close()

        # Clear provider-specific stats
        self.(_state_manager and _state_manager.update(
            "stats",
            {
                "generation_count": 0,
                "token_count_calls": 0,
                "error_count": 0,
                "total_processing_time": 0,
            },
        )

        # Mark as not initialized
        self.(_state_manager and _state_manager.update("initialized", False)
        (logger and logger.info(f"Provider {self.name} cleaned up successfully")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.

        This method sends a prompt to the OpenAI API and returns the generated text.
        It automatically initializes the provider if needed, and handles configuration
        overrides through kwargs.

        The generation process includes:
        1. Ensuring the provider is initialized
        2. Getting the API client from state
        3. Applying configuration overrides from kwargs
        4. Sending the prompt to the API
        5. Returning the generated text

        Args:
            prompt (str): The prompt to send to the model
            **kwargs: Additional keyword arguments to override configuration
                - temperature (float): Controls randomness (0.0-2.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - frequency_penalty (float): Penalizes repeated tokens
                - presence_penalty (float): Penalizes repeated topics
                - stop (List[str]): Sequences that stop generation

        Returns:
            str: The generated text response

        Raises:
            ModelError: If generation fails due to API issues
            ValueError: If invalid configuration is provided

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")

            # Basic generation
            response = (provider and provider.generate("Explain quantum computing")

            # Generation with configuration overrides
            response = (provider and provider.generate(
                "Write a poem about AI",
                temperature=0.9,
                max_tokens=200
            )
            ```
        """
        # Ensure component is initialized
        if not self.(_state_manager and _state_manager.get("initialized", False):
            (self and self.warm_up()

        # Get client from state
        client = self.(_state_manager and _state_manager.get("client")
        if client is None:
            client_manager = self.(_state_manager and _state_manager.get("client_manager")
            if client_manager is None:
                (self and self.warm_up()
                client = self.(_state_manager and _state_manager.get("client")
            else:
                client = (client_manager and client_manager.get_client()
                self.(_state_manager and _state_manager.update("client", client)

        # Get config from state
        config = self.(_state_manager and _state_manager.get("config")

        # Create a new config with updated values using the proper immutable pattern
        # First, check if any kwargs match direct config attributes
        config_kwargs = {}
        params_kwargs = {}

        for key, value in (kwargs and kwargs.items():
            if hasattr(config, key) and key != "params":
                config_kwargs[key] = value
            else:
                params_kwargs[key] = value

        # Create updated config using with_options for direct attributes
        if config_kwargs:
            new_config = (config and config.with_options(**config_kwargs)
        else:
            new_config = config

        # Add any params using with_params
        if params_kwargs:
            new_config = (new_config and new_config.with_params(**params_kwargs)

        # Store the updated config in the state manager
        self.(_state_manager and _state_manager.update("config", new_config)

        # Send prompt to client
        return (client and client.send_prompt(prompt, config)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text.

        This method counts the number of tokens in the provided text using the
        appropriate tokenizer for the current model. It automatically initializes
        the provider if needed.

        The token counting process includes:
        1. Ensuring the provider is initialized
        2. Getting the token counter from state
        3. Counting tokens using the token counter
        4. Updating token counting statistics

        Args:
            text (str): The text to count tokens for

        Returns:
            int: The number of tokens in the text

        Raises:
            ModelError: If token counting fails
            ValueError: If the text is invalid

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")

            # Count tokens in a string
            token_count = (provider and provider.count_tokens("How many tokens is this?")

            # Count tokens in a longer text
            with open("document.txt", "r") as f:
                text = (f and f.read()
                token_count = (provider and provider.count_tokens(text)
            ```
        """
        # Ensure component is initialized
        if not self.(_state_manager and _state_manager.get("initialized", False):
            (self and self.warm_up()

        # Get token counter from state
        token_counter = self.(_state_manager and _state_manager.get("token_counter")
        if token_counter is None:
            token_counter_manager = self.(_state_manager and _state_manager.get("token_counter_manager")
            if token_counter_manager is None:
                (self and self.warm_up()
                token_counter = self.(_state_manager and _state_manager.get("token_counter")
            else:
                token_counter = (token_counter_manager and token_counter_manager.get_token_counter()
                self.(_state_manager and _state_manager.update("token_counter", token_counter)

        # Update statistics
        stats = self.(_state_manager and _state_manager.get("stats", {})
        stats["token_count_calls"] = (stats and stats.get("token_count_calls", 0) + 1
        self.(_state_manager and _state_manager.update("stats", stats)

        # Count tokens
        return (token_counter and token_counter.count_tokens(text)

    def _create_default_client(self) -> APIClient:
        """
        Create a default OpenAI client.

        This method creates and returns a new OpenAI API client instance with the API key
        from the provider's configuration. It's called by the ModelProviderCore
        when no custom client is provided.

        Returns:
            APIClient: A new OpenAI API client instance

        Raises:
            ImportError: If the OpenAI package is not installed
            ValueError: If no API key is available
        """
        from sifaka.models.managers.openai_client import OpenAIClient

        return OpenAIClient(api_key=self.(_state_manager and _state_manager.get("config").api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter for the current model.

        This method creates and returns a new OpenAI token counter instance for the
        provider's model. It's called by the ModelProviderCore when no custom
        token counter is provided.

        Returns:
            TokenCounter: A new OpenAI token counter instance

        Raises:
            ImportError: If the tiktoken package is not installed
            RuntimeError: If token counter initialization fails
        """
        from sifaka.models.managers.openai_token_counter import OpenAITokenCounter

        return OpenAITokenCounter(model=self.(_state_manager and _state_manager.get("model_name"))

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
            provider = OpenAIProvider(model_name="gpt-4")
            (provider and provider.generate("Hello, world!")
            (provider and provider.count_tokens("How many tokens?")

            # Get usage statistics
            stats = (provider and provider.get_statistics()
            print(f"Generation count: {stats['generation_count']}")
            print(f"Token count calls: {stats['token_count_calls']}")
            ```
        """
        # Get statistics from tracing manager and state
        tracing_manager = self.(_state_manager and _state_manager.get("tracing_manager")
        tracing_stats = (tracing_manager and tracing_manager.get_statistics() if tracing_manager else {}

        # Combine with any other stats from state
        stats = self.(_state_manager and _state_manager.get("stats", {})

        return {**tracing_stats, **stats}

    @property
    def description(self) -> str:
        """
        Get a description of the provider.

        Returns:
            str: A description of the provider
        """
        return f"OpenAI provider using model {self.(_state_manager and _state_manager.get('model_name')}"

    def update_config(self, **kwargs) -> None:
        """
        Update the provider configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        config = self.(_state_manager and _state_manager.get("config")

        # Create a new config with updated values using the proper immutable pattern
        # First, check if any kwargs match direct config attributes
        config_kwargs = {}
        params_kwargs = {}

        for key, value in (kwargs and kwargs.items():
            if hasattr(config, key) and key != "params":
                config_kwargs[key] = value
            else:
                params_kwargs[key] = value

        # Create updated config using with_options for direct attributes
        if config_kwargs:
            new_config = (config and config.with_options(**config_kwargs)
        else:
            new_config = config

        # Add any params using with_params
        if params_kwargs:
            new_config = (new_config and new_config.with_params(**params_kwargs)

        # Update state
        self.(_state_manager and _state_manager.update("config", new_config)
