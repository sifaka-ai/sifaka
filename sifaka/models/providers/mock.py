"""
Mock model provider for testing.

This module provides a mock model provider implementation for testing purposes.
It simulates the behavior of a real model provider without making actual API calls.

## Overview
The Mock provider simulates API responses for testing, offering a way to test
code that depends on model providers without making actual API calls. It handles
simulated token counting, response generation, and execution tracking.

## Components
- **MockProvider**: Main provider class for mock models
- **MockAPIClient**: API client implementation for mock responses
- **MockTokenCounter**: Token counter implementation for mock models

## Usage Examples
```python
from sifaka.models.providers.mock import MockProvider
from sifaka.utils.config.models import ModelConfig

# Create a provider with default configuration
provider = MockProvider(model_name="mock-model")

# Create a provider with custom configuration
config = ModelConfig(
    temperature=0.8,
    max_tokens=2000,
    api_key="mock-api-key"
)
provider = MockProvider(model_name="mock-model", config=config)

# Generate text
response = provider.generate("Explain quantum computing") if provider else ""

# Count tokens
token_count = provider.count_tokens("How many tokens is this?") if provider else ""
```

## Error Handling
The provider implements simulated error handling for testing:
- Simulated API errors
- Simulated rate limiting
- Simulated network errors
"""

import time
import logging
import random
from typing import Dict, Any, Optional, ClassVar, List, Union

from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.utils.config.models import ModelConfig
from sifaka.models.core.provider import ModelProviderCore
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class MockAPIClient(APIClient):
    """Mock API client for testing."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the mock client.

        Args:
            api_key: Optional API key (not used but included for compatibility)
        """
        self.api_key = api_key or "mock-api-key"
        if logger:
            logger.debug("Initialized mock client")

    def send_prompt(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a mock prompt and return a response.

        Args:
            prompt: The prompt to send
            params: Parameters for the request including model configuration

        Returns:
            A mock response as a dictionary
        """
        # Simulate different responses based on configuration
        temperature = params.get("temperature", 0.7)
        prefix = "Detailed" if temperature < 0.5 else "Creative"

        # Return a dictionary format similar to real API responses
        return {
            "choices": [
                {
                    "text": f"{prefix} mock response to: {prompt}",
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "model": params.get("model_name", "mock-model"),
            "object": "text_completion",
            "usage": {
                "prompt_tokens": len(prompt.split()) if prompt else 0,
                "completion_tokens": len(f"{prefix} mock response to: {prompt}".split()),
                "total_tokens": (len(prompt.split()) if prompt else 0)
                + len(f"{prefix} mock response to: {prompt}".split()),
            },
        }


class MockTokenCounter(TokenCounter):
    """Mock token counter for testing."""

    def __init__(self, model: str = "mock-model") -> None:
        """
        Initialize the mock token counter.

        Args:
            model: The model name (not used but included for compatibility)
        """
        if logger:
            logger.debug(f"Initialized mock token counter for model {model}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text by splitting on whitespace.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens (words) in the text
        """
        return len(text.split()) if text else 0


class MockProvider(ModelProviderCore):
    """
    Mock model provider implementation for testing.

    This provider simulates the behavior of a real model provider without making
    actual API calls. It's useful for testing code that depends on model providers
    without incurring API costs or requiring internet connectivity.

    ## Architecture
    MockProvider extends ModelProviderCore and follows Sifaka's component-based
    architecture. It delegates API communication to MockAPIClient and token counting
    to MockTokenCounter. The provider uses standardized state management through
    the StateManager from utils/state.py.

    ## Lifecycle
    1. Initialization: Creates client and token counter managers
    2. Warm-up: Initializes API client and token counter
    3. Operation: Handles text generation and token counting
    4. Cleanup: Releases resources when no longer needed

    ## Error Handling
    The provider implements simulated error handling for testing:
    - Simulated API errors
    - Simulated rate limiting
    - Simulated network errors

    ## Examples
    ```python
    from sifaka.models.providers.mock import MockProvider
    from sifaka.utils.config.models import ModelConfig

    # Create a provider with default configuration
    provider = MockProvider(model_name="mock-model")

    # Create a provider with custom configuration
    config = ModelConfig(
        temperature=0.8,
        max_tokens=2000,
        api_key="mock-api-key"
    )
    provider = MockProvider(model_name="mock-model", config=config)

    # Generate text
    response = provider.generate("Explain quantum computing") if provider else ""

    # Count tokens
    token_count = provider.count_tokens("How many tokens is this?") if provider else ""
    ```

    Attributes:
        _state_manager (StateManager): Manages provider state
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "mock-model"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the mock provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
        """
        # Create default config if not provided
        if config is None:
            config = ModelConfig(
                temperature=0.7,
                max_tokens=100,
                api_key="mock-api-key",
                trace_enabled=True,
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

        This method simulates sending a prompt to an API and returns a mock generated text.
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

        Returns:
            str: The mock generated text response

        Raises:
            RuntimeError: If generation fails (simulated errors)

        Example:
            ```python
            provider = MockProvider(model_name="mock-model")

            # Basic invocation
            response = provider.invoke("Explain quantum computing") if provider else ""

            # Invocation with configuration overrides
            response = provider.invoke(
                "Write a poem about AI",
                temperature=0.9,
                max_tokens=200
            ) if provider else ""
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
        Create a default mock API client.

        This method creates and returns a new MockAPIClient instance with the API key
        from the provider's configuration. It's called by the ModelProviderCore
        when no custom client is provided.

        Returns:
            APIClient: A new MockAPIClient instance
        """
        config = self._state_manager and self._state_manager.get("config")
        api_key = config.api_key if config else None
        return MockAPIClient(api_key=api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default mock token counter.

        This method creates and returns a new MockTokenCounter instance for the
        provider's model. It's called by the ModelProviderCore when no custom
        token counter is provided.

        Returns:
            TokenCounter: A new MockTokenCounter instance
        """
        model_name = self._state_manager and self._state_manager.get("model_name", "")
        return MockTokenCounter(model=str(model_name))

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        model_name = self._state_manager and self._state_manager.get("model_name")
        return f"Mock-{model_name}"

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
            provider = MockProvider(model_name="mock-model")
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
        return f"Mock provider using model {model_name}"

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
