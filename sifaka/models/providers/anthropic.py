"""
Anthropic Model Provider

This module provides the AnthropicProvider class which implements the ModelProviderProtocol
interface for Anthropic Claude models.

## Overview
The Anthropic provider connects to Anthropic's API for text generation, offering access
to Claude models like Claude 3 Opus, Claude 3 Sonnet, and others. It handles authentication,
API communication, token counting, and response processing.

## Components
- **AnthropicProvider**: Main provider class for Anthropic Claude models
- **AnthropicClientManager**: Manages Anthropic API client creation and configuration
- **AnthropicTokenCounterManager**: Manages token counting for Anthropic models

## Usage Examples
```python
from sifaka.models.providers.anthropic import AnthropicProvider
from sifaka.utils.config and config and config and config.models import ModelConfig

# Create a provider with default configuration
provider = AnthropicProvider(model_name="claude-3-opus-20240229")

# Create a provider with custom configuration
config = ModelConfig(
    temperature=0.7,
    max_tokens=1000,
    api_key="your-api-key",
    trace_enabled=True,
)
provider = AnthropicProvider(model_name="claude-3-sonnet-20240229", config=config)

# Generate text
response = provider.generate("Explain quantum computing") if provider else ""

# Count tokens
token_count = provider.count_tokens("How many tokens is this?") if provider else ""
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
from typing import Any, Dict, Optional, ClassVar, List

# Import interfaces directly to avoid circular dependencies
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.core.provider import ModelProviderCore

# Import utilities
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.logging import get_logger

# Lazy import managers to avoid circular dependencies
# These will be imported at runtime when needed

logger = get_logger(__name__)


class AnthropicProvider(ModelProviderCore):
    """
    Anthropic model provider implementation.

    This provider supports Anthropic Claude models with configurable parameters,
    built-in token counting, and execution tracking. It handles communication
    with Anthropic's API, token counting, and response processing.

    ## Architecture
    AnthropicProvider implements the ModelProviderProtocol and follows Sifaka's component-based
    architecture. It delegates API communication to AnthropicClientManager and token counting
    to AnthropicTokenCounterManager. The provider uses standardized state management through
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
    from sifaka.models.providers.anthropic import AnthropicProvider
    from sifaka.utils.config and config and config and config.models import ModelConfig

    # Create a provider with default configuration
    provider = AnthropicProvider(model_name="claude-3-opus-20240229")

    # Create a provider with custom configuration
    config = ModelConfig(
        temperature=0.7,
        max_tokens=1000,
        api_key="your-api-key",
        trace_enabled=True,
    )
    provider = AnthropicProvider(model_name="claude-3-sonnet-20240229", config=config)

    # Generate text
    response = provider.generate("Explain quantum computing") if provider else ""

    # Count tokens
    token_count = provider.count_tokens("How many tokens is this?") if provider else ""
    ```

    Attributes:
        _state_manager (StateManager): Manages provider state
        _client_manager (AnthropicClientManager): Manages API client
        _token_counter_manager (AnthropicTokenCounterManager): Manages token counter
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "claude-3-opus-20240229"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[Optional[ModelConfig]] = None,
        api_client: Optional[Optional[APIClient]] = None,
        token_counter: Optional[Optional[TokenCounter]] = None,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
        """
        # Verify Anthropic package is installed
        try:
            if importlib.util.find_spec("anthropic") is None:
                raise ImportError()
        except ImportError:
            raise ImportError("Anthropic package is required. Install with: pip install anthropic")

        # Initialize with ModelProviderCore
        super().__init__(
            model_name=model_name,
            config=config or ModelConfig(),
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

        logger.info(f"Created AnthropicProvider with model {model_name}")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        This method sends a prompt to the Anthropic API and returns the generated text.
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
                - stop_sequences (List[str]): Sequences that stop generation

        Returns:
            str: The generated text response

        Raises:
            RuntimeError: If generation fails due to API issues
            ValueError: If invalid configuration is provided
        """
        # Ensure component is initialized
        if not self._state_manager or not self._state_manager.get("initialized", False):
            self.warm_up()

        # Track generation count in state
        start_time = time.time()

        try:
            result = self.generate(prompt, **kwargs)

            # Update statistics in state
            if self._state_manager:
                stats = self._state_manager.get("stats", {})
                if stats:
                    stats["generation_count"] = stats.get("generation_count", 0) + 1
                    stats["total_processing_time"] = stats.get("total_processing_time", 0) + (
                        (time.time() - start_time) * 1000
                    )
                    self._state_manager.update("stats", stats)

            return result

        except Exception as e:
            # Update error count in state
            if self._state_manager:
                stats = self._state_manager.get("stats", {})
                if stats:
                    stats["error_count"] = stats.get("error_count", 0) + 1
                    self._state_manager.update("stats", stats)

            # Record the error
            self._record_error(e)

            # Re-raise the exception
            raise

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
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
                - stop_sequences (List[str]): Sequences that stop generation

        Returns:
            str: The generated text response

        Raises:
            RuntimeError: If generation fails due to API issues
            ValueError: If invalid configuration is provided

        Example:
            ```python
            import asyncio

            async def generate_text():
                provider = AnthropicProvider(model_name="claude-3-opus-20240229")
                response = await provider.ainvoke("Explain quantum computing") if provider else ""
                return response

            # Run the async function
            response = asyncio.run(generate_text() if asyncio else "")
            ```
        """
        # Ensure component is initialized
        if not self._state_manager or not self._state_manager.get("initialized", False):
            self.warm_up()

        # Track generation count in state
        start_time = time.time()

        try:
            if hasattr(self, "agenerate"):
                result = await self.agenerate(prompt, **kwargs)
            else:
                # Fall back to synchronous generate
                result = self.generate(prompt, **kwargs)

            # Update statistics in state
            if self._state_manager:
                stats = self._state_manager.get("stats", {})
                if stats:
                    stats["generation_count"] = stats.get("generation_count", 0) + 1
                    stats["total_processing_time"] = stats.get("total_processing_time", 0) + (
                        (time.time() - start_time) * 1000
                    )
                    self._state_manager.update("stats", stats)

            return str(result)

        except Exception as e:
            # Update error count in state
            if self._state_manager:
                stats = self._state_manager.get("stats", {})
                if stats:
                    stats["error_count"] = stats.get("error_count", 0) + 1
                    self._state_manager.update("stats", stats)

            # Record the error
            self._record_error(e)

            # Re-raise the exception
            raise

    def _record_error(self, error: Exception) -> None:
        """Record an error in the state manager."""
        # Update error count in state
        if self._state_manager:
            stats = self._state_manager.get("stats", {})
            if stats:
                stats["error_count"] = stats.get("error_count", 0) + 1
                self._state_manager.update("stats", stats)

            # Use common error recording utility
            from sifaka.models.core.error_handling import record_error as core_record_error

            core_record_error(self, error)

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        model_name = "unknown"
        if self._state_manager:
            model_name = self._state_manager.get("model_name", model_name)
        return f"Anthropic-{model_name}"

    def warm_up(self) -> None:
        """
        Initialize resources needed by the Anthropic provider.

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
            ImportError: If the Anthropic package is not installed
            ModelError: If initialization fails due to API issues
        """
        # Ensure component is not already initialized
        if self._state_manager.get("initialized", False):
            logger.debug(f"Provider {self.name} already initialized")
            return

        # Lazy import managers to avoid circular dependencies
        from sifaka.models.managers.anthropic_client import AnthropicClientManager
        from sifaka.models.managers.anthropic_token_counter import AnthropicTokenCounterManager

        if not self._state_manager:
            return

        # Get dependencies from state
        model_name = self._state_manager.get("model_name")
        config = self._state_manager.get("config")
        api_client = self._state_manager.get("api_client")
        token_counter = self._state_manager.get("token_counter")

        # Create managers if they don't exist in state
        if not self._state_manager.get("client_manager"):
            client_manager = AnthropicClientManager(
                model_name=model_name,
                config=config,
                api_client=api_client,
            )
            self._state_manager.update("client_manager", client_manager)

        if not self._state_manager.get("token_counter_manager"):
            token_counter_manager = AnthropicTokenCounterManager(
                model_name=model_name,
                token_counter=token_counter,
            )
            self._state_manager.update("token_counter_manager", token_counter_manager)

        # Initialize client
        client_manager = self._state_manager.get("client_manager")
        if client_manager:
            client = client_manager.get_client()
            self._state_manager.update("client", client)

        # Initialize token counter
        token_counter_manager = self._state_manager.get("token_counter_manager")
        if token_counter_manager:
            token_counter = token_counter_manager.get_token_counter()
            self._state_manager.update("token_counter", token_counter)

        # Mark as initialized
        self._state_manager.update("initialized", True)
        logger.info(f"Provider {self.name} initialized successfully")

    def cleanup(self) -> None:
        """
        Release resources used by the Anthropic provider.

        This method releases any resources used by the provider, including:
        1. Closing the API client if it has a close method
        2. Resetting statistics
        3. Marking the provider as not initialized

        The provider can be reused after cleanup by calling warm_up() again.

        Example:
            ```python
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")
            provider.generate("Hello") if provider else ""  # Automatically warms up
            provider.cleanup() if provider else ""  # Release resources
            provider.warm_up() if provider else ""  # Reinitialize for reuse
            ```
        """
        # Check if already cleaned up
        if not (self._state_manager and self._state_manager.get("initialized", False)):
            if logger:
                logger.debug(f"Provider {self.name} not initialized, nothing to clean up")
            return

        # Release Anthropic-specific resources
        client = self._state_manager and self._state_manager.get("client")
        if client and hasattr(client, "close"):
            client.close()

        # Clear provider-specific stats
        if self._state_manager:
            self._state_manager.update(
                "stats",
                {
                    "generation_count": 0,
                    "token_count_calls": 0,
                    "error_count": 0,
                    "total_processing_time": 0,
                },
            )

        # Mark as not initialized
        if self._state_manager:
            self._state_manager.update("initialized", False)
        if logger:
            logger.info(f"Provider {self.name} cleaned up successfully")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        This method sends a prompt to the Anthropic API and returns the generated text.
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
                - temperature (float): Controls randomness (0.0-1.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - top_k (int): Top-k sampling parameter
                - stop_sequences (List[str]): Sequences that stop generation

        Returns:
            str: The generated text response

        Raises:
            ModelError: If generation fails due to API issues
            ValueError: If invalid configuration is provided

        Example:
            ```python
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")

            # Basic generation
            response = provider.generate("Explain quantum computing") if provider else ""

            # Generation with configuration overrides
            response = provider.generate(
                "Write a poem about AI",
                temperature=0.9,
                max_tokens=200
            ) if provider else ""
            ```
        """
        # Ensure component is initialized
        if not (self._state_manager and self._state_manager.get("initialized", False)):
            self.warm_up()

        # Get client from state
        client = self._state_manager and self._state_manager.get("client")
        if client is None:
            client_manager = self._state_manager and self._state_manager.get("client_manager")
            if client_manager is None:
                self.warm_up()
                client = self._state_manager and self._state_manager.get("client")
            else:
                client = client_manager and client_manager.get_client()
                if self._state_manager:
                    self._state_manager.update("client", client)

        # Get config from state
        config = self._state_manager and self._state_manager.get("config")

        # Create a new config with updated values using the proper immutable pattern
        # First, check if any kwargs match direct config attributes
        config_kwargs = {}
        params_kwargs = {}

        for key, value in kwargs.items() if kwargs else []:
            if hasattr(config, key) and key != "params":
                config_kwargs[key] = value
            else:
                params_kwargs[key] = value

        # Create updated config using with_options for direct attributes
        if config_kwargs and config:
            new_config = config.with_options(**config_kwargs)
        else:
            new_config = config

        # Add any params using with_params
        if params_kwargs and new_config:
            new_config = new_config.with_params(**params_kwargs)

        # Store the updated config in the state manager
        if self._state_manager:
            self._state_manager.update("config", new_config)

        # Send prompt to client
        result = client.send_prompt(prompt, config) if client else ""
        return str(result)

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
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")

            # Count tokens in a string
            token_count = provider.count_tokens("How many tokens is this?") if provider else ""

            # Count tokens in a longer text
            with open("document.txt", "r") as f:
                text = f.read() if f else ""
                token_count = provider.count_tokens(text) if provider else ""
            ```
        """
        # Ensure component is initialized
        if not (self._state_manager and self._state_manager.get("initialized", False)):
            self.warm_up()

        # Get token counter from state
        token_counter = self._state_manager and self._state_manager.get("token_counter")
        if token_counter is None:
            token_counter_manager = self._state_manager and self._state_manager.get(
                "token_counter_manager"
            )
            if token_counter_manager:
                token_counter = token_counter_manager.get_token_counter()
                if self._state_manager:
                    self._state_manager.update("token_counter", token_counter)

        # Update statistics
        stats = self._state_manager and self._state_manager.get("stats", {}) or {}
        stats["token_count_calls"] = (stats.get("token_count_calls", 0) + 1) if stats else 1
        if self._state_manager:
            self._state_manager.update("stats", stats)

        # Count tokens
        return token_counter.count_tokens(text) if token_counter else 0

    # Note: Text analysis functionality has been removed from the provider
    # For text analysis, use the critics component instead (e.g., SelfRefineCritic)
    # Example:
    # from sifaka.critics.implementations.self_refine import create_self_refine_critic
    # critic = create_self_refine_critic(llm_provider=anthropic_provider)

    def _create_default_client(self) -> APIClient:
        """
        Create a default Anthropic client.

        This method creates and returns a new Anthropic API client instance with the API key
        from the provider's configuration. It's called by the ModelProviderCore
        when no custom client is provided.

        Returns:
            APIClient: A new Anthropic API client instance

        Raises:
            ImportError: If the Anthropic package is not installed
            ValueError: If no API key is available
        """
        from sifaka.models.managers.anthropic_client import AnthropicClient

        config = self._state_manager and self._state_manager.get("config")
        api_key = config.api_key if config else None
        return AnthropicClient(api_key=api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter for the current model.

        This method creates and returns a new Anthropic token counter instance for the
        provider's model. It's called by the ModelProviderCore when no custom
        token counter is provided.

        Returns:
            TokenCounter: A new Anthropic token counter instance

        Raises:
            ImportError: If the anthropic package is not installed
            RuntimeError: If token counter initialization fails
        """
        from sifaka.models.managers.anthropic_token_counter import AnthropicTokenCounter

        model_name = ""
        if self._state_manager:
            model_name = self._state_manager.get("model_name", "")
        return AnthropicTokenCounter(model=model_name)

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
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")
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
        model_name = self._state_manager and self._state_manager.get("model_name")
        return f"Anthropic provider using model {model_name}"

    def update_config(self, config: Any) -> None:
        """
        Update the provider configuration.

        Args:
            config: The new configuration object or values to update
        """
        if not self._state_manager:
            return

        # If config is a dict, convert it to kwargs and call the kwargs version
        if isinstance(config, dict):
            self._update_config_with_kwargs(**config)
        # If config is a ModelConfig, update directly
        elif isinstance(config, ModelConfig):
            self._state_manager.update("config", config)
        else:
            raise ValueError(f"Config must be a dict or ModelConfig, got {type(config)}")

    def _update_config_with_kwargs(self, **kwargs: Any) -> None:
        """
        Update the provider configuration with keyword arguments.

        Args:
            **kwargs: Configuration parameters to update
        """
        if not self._state_manager:
            return

        config = self._state_manager.get("config")
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
        if params_kwargs and hasattr(new_config, "with_params"):
            new_config = new_config.with_params(**params_kwargs)

        # Update state
        self._state_manager.update("config", new_config)
