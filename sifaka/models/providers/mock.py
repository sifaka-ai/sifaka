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
from sifaka.models.config import ModelConfig

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
response = provider.generate("Explain quantum computing")

# Count tokens
token_count = provider.count_tokens("How many tokens is this?")
```

## Error Handling
The provider implements simulated error handling for testing:
- Simulated API errors
- Simulated rate limiting
- Simulated network errors
"""

from typing import Dict, Any, Optional, ClassVar

from sifaka.models.base import ModelConfig, APIClient, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.errors import handle_error
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
        logger.debug("Initialized mock client")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """
        Send a mock prompt and return a response.

        Args:
            prompt: The prompt to send
            config: Configuration for the request (not used but included for compatibility)

        Returns:
            A mock response
        """
        # Simulate different responses based on configuration
        temperature = config.temperature if hasattr(config, "temperature") else 0.7
        prefix = "Detailed" if temperature < 0.5 else "Creative"

        return f"{prefix} mock response to: {prompt}"


class MockTokenCounter(TokenCounter):
    """Mock token counter for testing."""

    def __init__(self, model: str = "mock-model") -> None:
        """
        Initialize the mock token counter.

        Args:
            model: The model name (not used but included for compatibility)
        """
        logger.debug(f"Initialized mock token counter for model {model}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text by splitting on whitespace.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens (words) in the text
        """
        return len(text.split())


class MockProvider(ModelProviderCore):
    """
    Mock model provider implementation for testing.

    This provider simulates the behavior of a real model provider without making
    actual API calls. It's useful for testing code that depends on model providers
    without incurring API costs or requiring internet connectivity.

    ## Architecture
    MockProvider extends ModelProviderCore and follows Sifaka's component-based
    architecture. It delegates API communication to MockAPIClient and token counting
    to MockTokenCounter.
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
        """Create a default mock API client."""
        return MockAPIClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default mock token counter."""
        return MockTokenCounter(model=self.model_name)

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return f"Mock-{self.model_name}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about provider usage.

        Returns:
            Dictionary with usage statistics
        """
        # Get statistics from tracing manager
        return self._tracing_manager.get_statistics()
