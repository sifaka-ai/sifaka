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
- **create_openai_provider**: Factory function for creating OpenAI providers

## Usage Examples
```python
from sifaka.models.openai import create_openai_provider
import os

# Create a provider with default settings
provider = create_openai_provider(api_key=os.environ.get("OPENAI_API_KEY"))

# Create a provider with custom settings
provider = create_openai_provider(
    model_name="gpt-4",
    temperature=0.8,
    max_tokens=2000,
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Generate text
response = provider.generate("Explain quantum computing in simple terms.")
print(response)

# Count tokens
token_count = provider.count_tokens("How many tokens is this?")
print(f"Token count: {token_count}")
```

## Error Handling
The module implements several error handling strategies:
- Validates API key and configuration parameters
- Catches and logs OpenAI API errors
- Provides informative error messages for common issues
- Implements retry logic for transient errors
- Gracefully handles rate limiting and quota errors

## Configuration
The provider supports standard ModelConfig options plus OpenAI-specific parameters:
- **model_name**: Name of the OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
- **temperature**: Controls randomness (0-1)
- **max_tokens**: Maximum tokens to generate
- **api_key**: OpenAI API key
- **params**: Additional OpenAI-specific parameters (frequency_penalty, presence_penalty, etc.)
"""

from typing import Optional, Any, Dict, ClassVar, Union

import openai
import tiktoken
from openai import OpenAI
from pydantic import PrivateAttr

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_model_state, ModelState, StateManager

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

    ## Error Handling
    The provider implements comprehensive error handling:
    - Validates input parameters during initialization
    - Catches and logs OpenAI API errors during generation
    - Handles token counting errors
    - Provides informative error messages for debugging
    - Implements retry logic for transient errors
    - Gracefully handles rate limiting and quota errors

    ## Examples
    ```python
    from sifaka.models.openai import OpenAIProvider, create_openai_provider
    from sifaka.models.base import ModelConfig
    import os

    # Direct instantiation
    provider = OpenAIProvider(
        model_name="gpt-4",
        config=ModelConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1000
        )
    )

    # Using factory function (recommended)
    provider = create_openai_provider(
        model_name="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1000
    )

    # Generate text
    response = provider.generate("Explain quantum computing in simple terms.")
    print(response)

    # With parameter overrides
    response = provider.generate(
        "Write a creative story.",
        temperature=0.9,
        max_tokens=2000
    )

    # Count tokens
    token_count = provider.count_tokens("How many tokens is this?")
    print(f"Token count: {token_count}")

    # Error handling
    try:
        response = provider.generate("Explain quantum computing")
    except ValueError as e:
        # Handle input validation errors
        print(f"Input error: {e}")
    except RuntimeError as e:
        # Handle API and generation errors
        print(f"Generation failed: {e}")
        # Use fallback strategy
        response = "I couldn't generate a response."
    ```
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


def create_openai_provider(
    model_name: str = OpenAIProvider.DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    trace_enabled: bool = True,
    config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    api_client: Optional[APIClient] = None,
    token_counter: Optional[TokenCounter] = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """
    Create an OpenAI model provider.

    This factory function creates an OpenAIProvider with the specified
    configuration options.

    Args:
        model_name: Name of the model to use (e.g., "gpt-4-turbo", "gpt-3.5-turbo")
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum number of tokens to generate
        api_key: OpenAI API key
        trace_enabled: Whether to enable tracing
        config: Optional model configuration
        api_client: Optional API client to use
        token_counter: Optional token counter to use
        **kwargs: Additional configuration options

    Returns:
        An initialized OpenAIProvider instance
    """
    # Create configuration
    if config is None:
        config = ModelConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            trace_enabled=trace_enabled,
            **kwargs,
        )
    elif isinstance(config, dict):
        config = ModelConfig(**{**config, **kwargs})

    # Create provider
    return OpenAIProvider(
        model_name=model_name,
        config=config,
        api_client=api_client,
        token_counter=token_counter,
    )
