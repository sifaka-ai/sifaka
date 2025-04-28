"""
Base protocols and types for model providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)

@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for model providers."""

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

@runtime_checkable
class APIClient(Protocol):
    """Protocol for API clients that handle direct communication with LLM services."""

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """Send a prompt to the LLM service and return the response."""
        ...

@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting functionality."""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        ...

@runtime_checkable
class LanguageModel(Protocol):
    """Protocol for language model interfaces."""

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...

class ModelProvider(ABC):
    """
    Abstract base class for model providers.

    This class enforces a consistent interface for all model providers
    while allowing for flexible implementation of specific provider features.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        """Initialize a model provider with explicit dependencies."""
        self._model_name = model_name
        self._config = config or ModelConfig()
        self._api_client = api_client
        self._token_counter = token_counter
        self._tracer = tracer or (Tracer() if self._config.trace_enabled else None)
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._config

    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """Create a default API client if none was provided."""
        ...

    @abstractmethod
    def _create_default_token_counter(self) -> TokenCounter:
        """Create a default token counter if none was provided."""
        ...

    def _ensure_api_client(self) -> APIClient:
        """Ensure an API client is available, creating a default one if needed."""
        if self._api_client is None:
            logger.debug(f"Creating default API client for {self.model_name}")
            self._api_client = self._create_default_client()
        return self._api_client

    def _ensure_token_counter(self) -> TokenCounter:
        """Ensure a token counter is available, creating a default one if needed."""
        if self._token_counter is None:
            logger.debug(f"Creating default token counter for {self.model_name}")
            self._token_counter = self._create_default_token_counter()
        return self._token_counter

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record a trace event if tracing is enabled."""
        if self._tracer and self._config.trace_enabled:
            trace_id = datetime.now().strftime(f"{self.model_name}_%Y%m%d%H%M%S")
            self._tracer.add_event(trace_id, event_type, data)

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
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
        """Generate text using the model."""
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

# Type variable for generic model provider types
T = TypeVar("T", bound=ModelProvider)
