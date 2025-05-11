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
from sifaka.utils.config import ModelConfig

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

import time
import importlib.util
from typing import Any, Dict, Optional, ClassVar, TYPE_CHECKING

# Import interfaces directly to avoid circular dependencies
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.interfaces.model import ModelProviderProtocol

# Import utilities
from sifaka.utils.config import OpenAIConfig
from sifaka.utils.errors import safely_execute_component_operation
from sifaka.utils.errors import ModelError
from sifaka.utils.common import record_error
from sifaka.utils.logging import get_logger

# Lazy import managers to avoid circular dependencies
# These will be imported at runtime when needed

logger = get_logger(__name__)


class OpenAIProvider(ModelProviderProtocol):
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
    from sifaka.utils.config import OpenAIConfig

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
    response = provider.generate("Explain quantum computing")

    # Count tokens
    token_count = provider.count_tokens("How many tokens is this?")
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
        config: Optional[OpenAIConfig] = None,
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
            if importlib.util.find_spec("openai") is None:
                raise ImportError()
        except ImportError:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        # Initialize state manager
        from sifaka.utils.state import StateManager

        self._state_manager = StateManager()

        # Store dependencies for lazy initialization
        self._model_name = model_name
        self._config = config or OpenAIConfig()
        self._api_client = api_client
        self._token_counter = token_counter

        # Initialize state
        self._state_manager.update("model_name", model_name)
        self._state_manager.update("config", self._config)
        self._state_manager.update("initialized", False)
        self._state_manager.update(
            "stats",
            {
                "generation_count": 0,
                "token_count_calls": 0,
                "error_count": 0,
                "total_processing_time": 0,
            },
        )

        logger.info(f"Created OpenAIProvider with model {model_name}")

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = time.time()

        # Define the operation
        def operation():
            # Actual processing logic
            return self.generate(prompt, **kwargs)

        # Use standardized error handling
        result = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            additional_metadata={"input_type": "prompt", "method": "invoke"},
        )

        # Update statistics
        processing_time = time.time() - start_time
        stats = self._state_manager.get("stats", {})
        stats["generation_count"] = stats.get("generation_count", 0) + 1
        stats["total_processing_time"] = (
            stats.get("total_processing_time", 0) + processing_time * 1000
        )
        self._state_manager.update("stats", stats)

        return result

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
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = time.time()

        try:
            # Define the async operation
            async def async_operation():
                if hasattr(self, "agenerate"):
                    return await self.agenerate(prompt, **kwargs)
                else:
                    # Fall back to synchronous generate
                    return self.generate(prompt, **kwargs)

            # Execute the async operation
            result = await async_operation()

            # Update statistics
            processing_time = time.time() - start_time
            stats = self._state_manager.get("stats", {})
            stats["generation_count"] = stats.get("generation_count", 0) + 1
            stats["total_processing_time"] = (
                stats.get("total_processing_time", 0) + processing_time * 1000
            )
            self._state_manager.update("stats", stats)

            return result

        except Exception as e:
            # Record the error using standardized error handling
            self._record_error(e)

            # Raise a standardized error
            raise ModelError(
                f"Error in async invocation: {str(e)}",
                metadata={
                    "component_name": self.name,
                    "model_name": self._state_manager.get("model_name"),
                    "method": "ainvoke",
                    "error_type": type(e).__name__,
                },
            ) from e

    def _record_error(self, error: Exception) -> None:
        """Record an error in the state manager."""
        # Update error count in state
        stats = self._state_manager.get("stats", {})
        stats["error_count"] = stats.get("error_count", 0) + 1
        self._state_manager.update("stats", stats)

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
        return f"OpenAI-{self._state_manager.get('model_name')}"

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
        if self._state_manager.get("initialized", False):
            logger.debug(f"Provider {self.name} already initialized")
            return

        # Lazy import managers to avoid circular dependencies
        from sifaka.models.managers.openai_client import OpenAIClientManager
        from sifaka.models.managers.openai_token_counter import OpenAITokenCounterManager

        # Create managers if they don't exist
        if not hasattr(self, "_client_manager"):
            self._client_manager = OpenAIClientManager(
                model_name=self._model_name,
                config=self._config,
                api_client=self._api_client,
            )

        if not hasattr(self, "_token_counter_manager"):
            self._token_counter_manager = OpenAITokenCounterManager(
                model_name=self._model_name,
                token_counter=self._token_counter,
            )

        # Initialize client
        client = self._client_manager.get_client()
        self._state_manager.update("client", client)

        # Initialize token counter
        token_counter = self._token_counter_manager.get_token_counter()
        self._state_manager.update("token_counter", token_counter)

        # Mark as initialized
        self._state_manager.update("initialized", True)
        logger.info(f"Provider {self.name} initialized successfully")

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
            provider.generate("Hello")  # Automatically warms up
            provider.cleanup()  # Release resources
            provider.warm_up()  # Reinitialize for reuse
            ```
        """
        # Check if already cleaned up
        if not self._state_manager.get("initialized", False):
            logger.debug(f"Provider {self.name} not initialized, nothing to clean up")
            return

        # Release OpenAI-specific resources
        client = self._state_manager.get("client")
        if client and hasattr(client, "close"):
            client.close()

        # Clear provider-specific stats
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
        self._state_manager.update("initialized", False)
        logger.info(f"Provider {self.name} cleaned up successfully")

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
            response = provider.generate("Explain quantum computing")

            # Generation with configuration overrides
            response = provider.generate(
                "Write a poem about AI",
                temperature=0.9,
                max_tokens=200
            )
            ```
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Get client from state
        client = self._state_manager.get("client")
        if client is None:
            client = self._client_manager.get_client()
            self._state_manager.update("client", client)

        # Get config from state
        config = self._state_manager.get("config")

        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config, "params"):
                config.params[key] = value

        # Send prompt to client
        return client.send_prompt(prompt, config)

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
            token_count = provider.count_tokens("How many tokens is this?")

            # Count tokens in a longer text
            with open("document.txt", "r") as f:
                text = f.read()
                token_count = provider.count_tokens(text)
            ```
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Get token counter from state
        token_counter = self._state_manager.get("token_counter")
        if token_counter is None:
            token_counter = self._token_counter_manager.get_token_counter()
            self._state_manager.update("token_counter", token_counter)

        # Update statistics
        stats = self._state_manager.get("stats", {})
        stats["token_count_calls"] = stats.get("token_count_calls", 0) + 1
        self._state_manager.update("stats", stats)

        # Count tokens
        return token_counter.count_tokens(text)

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
        tracing_stats = tracing_manager.get_statistics() if tracing_manager else {}

        # Combine with any other stats from state
        stats = self._state_manager.get("stats", {})

        return {**tracing_stats, **stats}
