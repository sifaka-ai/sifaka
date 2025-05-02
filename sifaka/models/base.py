"""
Base protocols and types for model providers.

This module provides the core interfaces and base implementations for model providers,
including protocols for API clients, token counters, and language models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, Protocol, TypeVar, Union, runtime_checkable

from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)

# Type variables for generic type definitions
T = TypeVar("T", bound="ModelProvider")
R = TypeVar("R")  # Return type for model operations
C = TypeVar("C", bound="ModelConfig")  # Config type

@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration for model providers.

    This class represents a set of configuration options for model providers.
    Because it is immutable, you must use the with_* methods to create
    modified versions of a configuration.

    Examples:
        ```python
        # Create a default configuration
        config = ModelConfig()

        # Create a modified configuration
        new_config = config.with_temperature(0.9).with_max_tokens(2000)

        # Use the configuration with a model provider
        provider = AnthropicProvider(model_name="claude-3-opus", config=new_config)
        ```
    """

    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None
    trace_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    def with_temperature(self, temperature: float) -> "ModelConfig":
        """
        Create a new config with the specified temperature.

        Args:
            temperature: The new temperature value (0-1)

        Returns:
            A new ModelConfig with the updated temperature

        Raises:
            ValueError: If temperature is not between 0 and 1
        """
        return replace(self, temperature=temperature)

    def with_max_tokens(self, max_tokens: int) -> "ModelConfig":
        """
        Create a new config with the specified max_tokens.

        Args:
            max_tokens: The new max_tokens value

        Returns:
            A new ModelConfig with the updated max_tokens

        Raises:
            ValueError: If max_tokens is not positive
        """
        return replace(self, max_tokens=max_tokens)

    def with_api_key(self, api_key: Optional[str]) -> "ModelConfig":
        """
        Create a new config with the specified API key.

        Args:
            api_key: The new API key

        Returns:
            A new ModelConfig with the updated API key
        """
        return replace(self, api_key=api_key)

    def with_trace_enabled(self, trace_enabled: bool) -> "ModelConfig":
        """
        Create a new config with the specified trace_enabled setting.

        Args:
            trace_enabled: Whether tracing should be enabled

        Returns:
            A new ModelConfig with the updated trace_enabled setting
        """
        return replace(self, trace_enabled=trace_enabled)


@runtime_checkable
class APIClient(Protocol):
    """
    Protocol for API clients that handle direct communication with LLM services.

    Classes implementing this protocol are responsible for sending prompts to
    language model services and returning the responses.

    Lifecycle:
    1. Initialization: Set up any necessary client libraries and authentication
    2. Usage: Send prompts to the API and return responses
    3. Cleanup: Release any resources when no longer needed
    """

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """
        Send a prompt to the LLM service and return the response.

        Args:
            prompt: The prompt to send to the LLM service
            config: The configuration to use for the request

        Returns:
            The text response from the LLM service

        Raises:
            Exception: If there is an error communicating with the service
        """
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """
    Protocol for token counting functionality.

    Classes implementing this protocol are responsible for counting
    the number of tokens in a piece of text, using the appropriate
    tokenization method for a specific model.

    Lifecycle:
    1. Initialization: Set up any necessary tokenizers
    2. Usage: Count tokens in text
    3. Cleanup: Release any resources when no longer needed
    """

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            Exception: If there is an error counting tokens
        """
        ...


@runtime_checkable
class LanguageModel(Protocol[R]):
    """
    Protocol for language model interfaces.

    Classes implementing this protocol provide a high-level interface
    for generating text using language models.

    Type Parameters:
        R: The return type of the generate method

    Lifecycle:
    1. Initialization: Set up any necessary clients and resources
    2. Usage: Generate text from prompts
    3. Cleanup: Release any resources when no longer needed
    """

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The name of the language model
        """
        ...

    def generate(self, prompt: str, **kwargs) -> R:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional model-specific parameters

        Returns:
            The generated text or structured response

        Raises:
            Exception: If generation fails
        """
        ...


class ModelProvider(ABC, Generic[C]):
    """
    Abstract base class for model providers.

    This class enforces a consistent interface for all model providers
    while allowing for flexible implementation of specific provider features.

    Type Parameters:
        C: The configuration type, must be a subclass of ModelConfig

    Lifecycle:
    1. Initialization: Set up the provider with a model name and configuration
    2. Usage: Generate text, count tokens, and trace events
    3. Cleanup: Release any resources when no longer needed

    Examples:
        ```python
        # Create a provider with default configuration
        provider = AnthropicProvider(model_name="claude-3-opus")

        # Generate text
        response = provider.generate("Explain quantum computing")

        # Count tokens
        token_count = provider.count_tokens("How many tokens is this?")
        ```
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[C] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        """
        Initialize a model provider with explicit dependencies.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
            tracer: Optional tracer to use
        """
        self._model_name = model_name
        self._config = config or self._create_default_config()  # type: C
        self._api_client = api_client
        self._token_counter = token_counter
        self._tracer = tracer or (Tracer() if self._config.trace_enabled else None)
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The name of the language model
        """
        return self._model_name

    @property
    def config(self) -> C:
        """
        Get the model configuration.

        Returns:
            The current model configuration
        """
        return self._config

    def _create_default_config(self) -> C:
        """
        Create a default configuration for this provider.

        Returns:
            A default configuration
        """
        return ModelConfig()  # type: ignore

    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """
        Create a default API client if none was provided.

        Returns:
            A default API client for the model

        Raises:
            RuntimeError: If a default client cannot be created
        """
        ...

    @abstractmethod
    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter if none was provided.

        Returns:
            A default token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        ...

    def _ensure_api_client(self) -> APIClient:
        """
        Ensure an API client is available, creating a default one if needed.

        Returns:
            The API client to use
        """
        if self._api_client is None:
            logger.debug(f"Creating default API client for {self.model_name}")
            self._api_client = self._create_default_client()
        return self._api_client

    def _ensure_token_counter(self) -> TokenCounter:
        """
        Ensure a token counter is available, creating a default one if needed.

        Returns:
            The token counter to use
        """
        if self._token_counter is None:
            logger.debug(f"Creating default token counter for {self.model_name}")
            self._token_counter = self._create_default_token_counter()
        return self._token_counter

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record a trace event if tracing is enabled.

        Args:
            event_type: The type of event to record
            data: The data to record with the event
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

        Returns:
            The generated text

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty
            RuntimeError: If an error occurs during generation
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
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


def create_model_provider(
    provider_type: Callable[..., T],
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> T:
    """
    Factory function to create a model provider with a standardized configuration.

    This function simplifies the creation of model providers by providing
    a consistent interface for common configuration options.

    Args:
        provider_type: The class of the provider to create
        model_name: The name of the model to use
        api_key: Optional API key for the provider
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional provider-specific arguments

    Returns:
        A configured model provider instance

    Examples:
        ```python
        # Create an Anthropic provider
        provider = create_model_provider(
            AnthropicProvider,
            model_name="claude-3-opus",
            api_key="your-api-key",
            temperature=0.8
        )
        ```
    """
    config = ModelConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
    return provider_type(model_name=model_name, config=config, **kwargs)
