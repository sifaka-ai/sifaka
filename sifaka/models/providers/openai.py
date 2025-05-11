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
from sifaka.utils.config import ModelConfig

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
import time
from typing import Any, Dict, Optional, ClassVar

import tiktoken

from sifaka.models.base import APIClient, TokenCounter
from sifaka.utils.config import ModelConfig
from sifaka.models.core import ModelProviderCore
from sifaka.utils.error_patterns import safely_execute_component_operation
from sifaka.utils.errors import ModelError
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

        # Define the generation operation
        def generate_operation():
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

        # Use the standardized safely_execute_component_operation function
        return safely_execute_component_operation(
            operation=generate_operation,
            component_name="OpenAIClient",
            component_type="APIClient",
            additional_metadata={"model_name": config.params.get("model_name", "gpt-4")},
        )


class OpenAITokenCounter(TokenCounter):
    """Token counter using tiktoken for OpenAI models."""

    def __init__(self, model: str = "gpt-4") -> None:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model to count tokens for
        """

        # Define the initialization operation
        def init_operation():
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
            return self.encoding

        # Use the standardized safely_execute_component_operation function
        safely_execute_component_operation(
            operation=init_operation,
            component_name="OpenAITokenCounter",
            component_type="TokenCounter",
            additional_metadata={"model_name": model},
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text using the model's encoding.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """

        # Define the token counting operation
        def count_operation():
            return len(self.encoding.encode(text))

        # Use the standardized safely_execute_component_operation function
        return safely_execute_component_operation(
            operation=count_operation,
            component_name="OpenAITokenCounter",
            component_type="TokenCounter",
            additional_metadata={"model_name": "tiktoken"},
        )


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
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = __import__("time").time()

        # Define the operation
        def operation():
            # Actual processing logic
            return self.generate(prompt, **kwargs)

        # Use standardized error handling
        from sifaka.utils.error_patterns import safely_execute_component_operation

        result = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            additional_metadata={"input_type": "prompt", "method": "invoke"},
        )

        # Update statistics
        processing_time = __import__("time").time() - start_time
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
        start_time = __import__("time").time()

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
            processing_time = __import__("time").time() - start_time
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
            from sifaka.utils.errors import ModelError

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
        from sifaka.utils.common import record_error

        record_error(
            error=error,
            component_name=self.name,
            component_type=self.__class__.__name__,
            state_manager=self._state_manager,
        )

    def _create_default_client(self) -> APIClient:
        """Create a default OpenAI client."""
        return OpenAIClient(api_key=self._state_manager.get("config").api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter for the current model."""
        return OpenAITokenCounter(model=self._state_manager.get("model_name"))

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return f"OpenAI-{self._state_manager.get('model_name')}"

    def _initialize_resources(self) -> None:
        """
        Initialize resources needed by the OpenAI provider.

        This method is called by the warm_up method in the parent class.
        """
        # Call parent implementation first
        super()._initialize_resources()

        # Initialize OpenAI-specific resources
        # Ensure client is initialized
        client = self._state_manager.get("client")
        if client is None:
            client = self._create_default_client()
            self._state_manager.update("client", client)

        # Ensure token counter is initialized
        token_counter = self._state_manager.get("token_counter")
        if token_counter is None:
            token_counter = self._create_default_token_counter()
            self._state_manager.update("token_counter", token_counter)

        # Initialize provider-specific stats if not already present
        if not self._state_manager.get("stats"):
            stats = {
                "generation_count": 0,
                "token_count_calls": 0,
                "error_count": 0,
                "total_processing_time": 0,
            }
            self._state_manager.update("stats", stats)

    def _release_resources(self) -> None:
        """
        Release resources used by the OpenAI provider.

        This method is called by the cleanup method in the parent class.
        """
        # Call parent implementation first
        super()._release_resources()

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

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about provider usage.

        Returns:
            Dictionary with usage statistics
        """
        # Get statistics from tracing manager and state
        tracing_manager = self._state_manager.get("tracing_manager")
        tracing_stats = tracing_manager.get_statistics() if tracing_manager else {}

        # Combine with any other stats from state
        stats = self._state_manager.get("stats", {})

        return {**tracing_stats, **stats}
