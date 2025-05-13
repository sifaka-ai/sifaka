"""
Model Provider Module

This module provides the abstract base class for model providers in the Sifaka framework.

## Overview
The ModelProvider class is the foundation for all language model integrations in Sifaka.
It defines an abstract base class that standardizes how model providers are implemented,
configured, and used throughout the system, ensuring consistent behavior and error handling.

## Components
- **ModelProvider**: Abstract base class implementing the ModelProviderProtocol

## Architecture
The model provider follows a layered architecture:
1. **ModelProvider**: High-level interface for model interactions
2. **APIClient**: Low-level communication with model services
3. **TokenCounter**: Utility for token counting
4. **Config**: Configuration and settings management

## Usage Examples
```python
from sifaka.models.base.provider import ModelProvider
from sifaka.interfaces.client import APIClientProtocol
from sifaka.interfaces.counter import TokenCounterProtocol
from sifaka.utils.config.models import ModelConfig

# Basic usage
provider = ProviderClass(model_name="model-name")
response = provider.generate("Explain quantum computing") if provider else ""
token_count = provider.count_tokens("How many tokens is this?") if provider else ""

# With custom configuration
config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
provider = ProviderClass(model_name="model-name", config=config)

# With explicit dependencies
class CustomClient:
    def send_prompt(self, prompt, config):
        # Custom implementation
        return "Response"

class CustomTokenCounter:
    def count_tokens(self, text):
        # Custom implementation
        return len(text.split() if text else "")

# Verify protocol compliance
assert isinstance(CustomClient(), APIClientProtocol)
assert isinstance(CustomTokenCounter(), TokenCounterProtocol)

provider = ProviderClass(
    model_name="model-name",
    config=config,
    api_client=CustomClient(),
    token_counter=CustomTokenCounter()
)
```

## Error Handling
The model provider implements several error handling patterns:
1. **Input Validation**: Validates all inputs before processing
2. **Dependency Management**: Ensures required dependencies exist
3. **Exception Handling**: Proper exception hierarchy
4. **Tracing and Logging**: Records errors for diagnosis
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, Optional

# Import the interfaces for runtime use
from sifaka.interfaces.model import ModelProviderProtocol as ModelProviderInterface
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter

from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer

# Import type variables
from sifaka.models.base.types import C

# Import configuration from utils/config.py
from sifaka.utils.config.models import ModelConfig

logger = get_logger(__name__)


class ModelProvider(ModelProviderInterface, Generic[C], ABC):
    """
    Abstract base class for model providers.

    This class implements the ModelProviderProtocol interface and enforces a consistent
    interface for all model providers while allowing for flexible implementation of
    specific provider features.

    ## Architecture
    ModelProvider follows a layered architecture with:
    - High-level interface for model interactions
    - Dependency injection for API clients and token counters
    - Tracing and logging for monitoring and debugging
    - Generic type parameter for configuration customization
    - Abstract methods for provider-specific implementations

    Type Parameters:
        C: The configuration type, must be a subclass of ModelConfig

    ## Lifecycle
    1. **Initialization**: Set up with model name and configuration
       - Create/validate config
       - Initialize dependencies (API client, token counter)
       - Set up tracing if enabled

    2. **Configuration**: Configure generation parameters
       - Set temperature, max_tokens, etc.
       - Configure API keys and authentication

    3. **Operation**: Main usage phase
       - Generate text from prompts
       - Count tokens for optimization
       - Trace events for monitoring

    4. **Error Handling**: Manage failures
       - Handle API errors
       - Implement retries and circuit breakers
       - Provide meaningful error messages

    5. **Cleanup**: Release resources
       - Close API clients
       - Ensure proper resource management

    ## Error Handling
    This class implements several error handling patterns:

    1. **Input Validation**: Validates all inputs before processing
       - Checks prompt is a string and not empty
       - Validates configuration parameters

    2. **Dependency Management**: Ensures required dependencies exist
       - Creates default components when needed
       - Validates components implement required protocols

    3. **Exception Handling**: Proper exception hierarchy
       - Uses TypeError for type validation issues
       - Uses ValueError for invalid inputs
       - Uses RuntimeError for operational failures
       - Preserves original exception in context

    4. **Tracing and Logging**: Records errors for diagnosis
       - Logs all errors with context
       - Traces events for debugging
       - Includes relevant metadata with errors

    Attributes:
        _model_name (str): The name of the model being used
        _config (C): The configuration for the model provider
        _api_client (APIClient): The client for API communication
        _token_counter (TokenCounter): The counter for token estimation
        _tracer (Tracer): The tracer for event recording
        _initialized (bool): Whether the provider has been initialized
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Optional[C]] = None,
        api_client: Optional[Optional[APIClient]] = None,
        token_counter: Optional[Optional[TokenCounter]] = None,
        tracer: Optional[Optional[Tracer]] = None,
    ) -> None:
        """
        Initialize a model provider with explicit dependencies.

        This method initializes a model provider with the specified model name and
        optional dependencies. If dependencies are not provided, default ones will
        be created lazily when needed using the _create_default_* methods.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
            tracer: Optional tracer to use

        Raises:
            ValueError: If model_name is empty
            TypeError: If dependencies don't implement required protocols
        """
        self._model_name = model_name
        self._config = config or self._create_default_config()  # type: C
        self._api_client = api_client
        self._token_counter = token_counter
        self._tracer = tracer or (Tracer() if self._config.trace_enabled else None)
        self._initialized = False
        logger.info(f"Created {self.__class__.__name__} with model {model_name}")

    def initialize(self) -> None:
        """
        Initialize the model provider.

        This method initializes any resources needed by the model provider.
        It ensures that the API client and token counter are created if they
        haven't been provided during initialization.

        Raises:
            RuntimeError: If initialization fails for any reason
        """
        if self._initialized:
            logger.debug(f"{self.__class__.__name__} already initialized")
            return

        try:
            # Ensure API client and token counter are created
            self._ensure_api_client()
            self._ensure_token_counter()
            self._initialized = True
            logger.info(f"Initialized {self.__class__.__name__} with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise RuntimeError(f"Failed to initialize {self.__class__.__name__}: {e}") from e

    def cleanup(self) -> None:
        """
        Clean up the model provider.

        This method releases any resources held by the model provider.
        It ensures that the API client and token counter are properly closed
        to prevent resource leaks.

        Raises:
            RuntimeError: If cleanup fails for any reason
        """
        if not self._initialized:
            logger.debug(f"{self.__class__.__name__} not initialized, nothing to clean up")
            return

        try:
            # Close API client if it has a close method
            if self._api_client and hasattr(self._api_client, "close"):
                self._api_client.close()

            # Close token counter if it has a close method
            if self._token_counter and hasattr(self._token_counter, "close"):
                self._token_counter.close()

            self._initialized = False
            logger.info(f"Cleaned up {self.__class__.__name__} with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to clean up {self.__class__.__name__}: {e}")
            raise RuntimeError(f"Failed to clean up {self.__class__.__name__}: {e}") from e

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        This property provides read-only access to the model name that was
        specified during initialization.

        Returns:
            str: The name of the language model
        """
        return self._model_name

    @property
    def config(self) -> C:
        """
        Get the model configuration.

        This property provides read-only access to the current configuration
        object. The configuration can be updated using the update_config method.

        Returns:
            C: The current model configuration
        """
        return self._config

    @property
    def description(self) -> str:
        """
        Get the model provider description.

        This property returns a human-readable description of the provider,
        including the provider class name and the model name.

        Returns:
            str: The description of the model provider
        """
        return f"{self.__class__.__name__} for {self.model_name}"

    def update_config(self, config: C) -> None:
        """
        Update the model configuration.

        This method replaces the current configuration with a new one.
        It validates that the new configuration is an instance of ModelConfig
        before applying the change.

        Args:
            config (C): The new configuration object

        Raises:
            ValueError: If the configuration is not an instance of ModelConfig
        """
        if not isinstance(config, ModelConfig):
            raise ValueError(f"Config must be an instance of ModelConfig, got {type(config)}")
        self._config = config
        logger.info(f"Updated configuration for {self.__class__.__name__} with {self.model_name}")

    def _create_default_config(self) -> C:
        """
        Create a default configuration for this provider.

        This method creates a default configuration when none is provided during
        initialization. It's used by the constructor to ensure a valid configuration
        is always available.

        Returns:
            C: A default configuration instance appropriate for this provider
        """
        return ModelConfig()  # type: ignore

    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """
        Create a default API client if none was provided.

        This abstract method must be implemented by subclasses to provide
        a default API client implementation specific to the model provider.

        Returns:
            APIClient: A default API client for the model

        Raises:
            RuntimeError: If a default client cannot be created
        """
        ...

    @abstractmethod
    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter if none was provided.

        This abstract method must be implemented by subclasses to provide
        a default token counter implementation specific to the model provider.

        Returns:
            TokenCounter: A default token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        ...

    def _ensure_api_client(self) -> APIClient:
        """
        Ensure an API client is available, creating a default one if needed.

        This method implements the lazy initialization pattern for the API client.
        It checks if an API client has been provided during initialization, and if not,
        creates a default one using the _create_default_client method.

        Returns:
            The API client to use (existing or newly created)

        Raises:
            RuntimeError: If a default client cannot be created
            TypeError: If the created client doesn't implement APIClient protocol
        """
        if self._api_client is None:
            logger.debug(f"Creating default API client for {self.model_name}")
            self._api_client = self._create_default_client()
        return self._api_client

    def _ensure_token_counter(self) -> TokenCounter:
        """
        Ensure a token counter is available, creating a default one if needed.

        This method implements the lazy initialization pattern for the token counter.
        It checks if a token counter has been provided during initialization, and if not,
        creates a default one using the _create_default_token_counter method.

        Returns:
            The token counter to use (existing or newly created)

        Raises:
            RuntimeError: If a default token counter cannot be created
            TypeError: If the created counter doesn't implement TokenCounter protocol
        """
        if self._token_counter is None:
            logger.debug(f"Creating default token counter for {self.model_name}")
            self._token_counter = self._create_default_token_counter()
        return self._token_counter

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record a trace event if tracing is enabled.

        This method provides a standardized way to record events for monitoring,
        debugging, and analysis. It only records events if tracing is enabled
        in the configuration.

        Args:
            event_type: The type of event to record (e.g., "generate", "token_count")
            data: The data to record with the event as key-value pairs
        """
        if self._tracer and self._config.trace_enabled:
            trace_id = datetime.now().strftime(f"{self.model_name}_%Y%m%d%H%M%S")
            self._tracer.add_event(trace_id, event_type, data)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            TypeError: If text is not a string
            RuntimeError: If token counting fails
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        counter = self._ensure_token_counter()
        token_count = counter.count_tokens(text)

        self._trace_event(
            "token_count",
            {
                "text_length": len(text),
                "token_count": token_count,
            },
        )

        return token_count

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the model.

        Args:
            prompt: The prompt to generate from
            **kwargs: Optional overrides for model configuration
                - temperature: Control randomness (0-1)
                - max_tokens: Maximum tokens to generate
                - api_key: Override API key
                - trace_enabled: Override tracing setting

        Returns:
            The generated text

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            RuntimeError: If an error occurs during generation
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Update config with any override kwargs
        config = ModelConfig(
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            api_key=kwargs.pop("api_key", self.config.api_key),
            trace_enabled=kwargs.pop("trace_enabled", self.config.trace_enabled),
        )

        # Count tokens before generation
        prompt_tokens = self.count_tokens(prompt)
        if prompt_tokens > config.max_tokens:
            logger.warning(
                f"Prompt tokens ({prompt_tokens}) exceed max_tokens ({config.max_tokens})"
            )

        start_time = datetime.now()
        client = self._ensure_api_client()

        try:
            response = client.send_prompt(prompt, config)

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self._trace_event(
                "generate",
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": self.count_tokens(response),
                    "duration_ms": duration_ms,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "success": True,
                },
            )

            logger.debug(
                f"Generated response in {duration_ms:.2f}ms " f"(prompt: {prompt_tokens} tokens)"
            )

            return response

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text with {self.model_name}: {error_msg}")

            self._trace_event(
                "error",
                {
                    "error": error_msg,
                    "prompt_tokens": prompt_tokens,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
            )

            raise RuntimeError(f"Error generating text with {self.model_name}: {error_msg}") from e
