"""Provider-agnostic LLM client with automatic provider detection and PydanticAI integration.

This module provides a unified interface for interacting with various LLM providers
(OpenAI, Anthropic, Google Gemini, Groq) while handling provider-specific details
internally. It integrates with PydanticAI for structured outputs.

## Key Features:

- **Automatic Provider Detection**: Infers provider from model name
- **Unified Interface**: Same API regardless of provider
- **Structured Outputs**: PydanticAI integration for type-safe responses
- **Retry Logic**: Built-in retry for transient failures
- **OpenAI Compatibility**: Uses OpenAI SDK for compatible providers

## Supported Providers:

- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo models
- **Anthropic**: Claude 3 family (via PydanticAI)
- **Google Gemini**: Gemini Pro models (via PydanticAI)
- **Groq**: Fast inference for open models

## Usage:

    >>> from sifaka.core.llm_client import LLMManager
    >>>
    >>> # Automatic provider detection
    >>> client = await LLMManager.get_client(model="gpt-4")
    >>> response = await client.complete(messages)
    >>>
    >>> # Structured output with PydanticAI
    >>> agent = client.create_agent(
    ...     system_prompt="You are a helpful assistant",
    ...     result_type=MyResponseModel
    ... )
    >>> result = await agent.run("Analyze this text")

## Environment Variables:

Set API keys for each provider:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY
- GROQ_API_KEY

The client will automatically use the appropriate key based on the model.
"""

import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

try:
    import logfire
except ImportError:
    # logfire is optional
    logfire = None  # type: ignore[assignment]
import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

from ..core.retry import RETRY_STANDARD, with_retry

if TYPE_CHECKING:
    from .llm_client_pool import ConnectionMetrics, LLMClientPool

# Load environment variables
load_dotenv()

# Configure logfire if token is available
if logfire and os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))


class Provider(str, Enum):
    """Enumeration of supported LLM providers.

    Each provider represents a different LLM API service. The enum
    values are used for configuration and automatic detection.

    Attributes:
        OPENAI: OpenAI's GPT models (GPT-3.5, GPT-4, etc.)
        ANTHROPIC: Anthropic's Claude models (Claude 3 family)
        GEMINI: Google's Gemini models (Gemini Pro, etc.)
        GROQ: Groq's fast inference service for open models
        OLLAMA: Local Ollama models (Llama, Mistral, etc.)
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider.

    Provides a consistent response format regardless of which provider
    is used, making it easy to switch between providers.

    Attributes:
        content: The generated text from the LLM
        usage: Token usage statistics with keys like 'prompt_tokens',
            'completion_tokens', and 'total_tokens'
        model: The actual model name used for generation

    Example:
        >>> response = LLMResponse(
        ...     content="Generated text here",
        ...     usage={"prompt_tokens": 10, "completion_tokens": 20},
        ...     model="gpt-4"
        ... )
    """

    content: str
    usage: Dict[str, int] = Field(default_factory=dict)
    model: str


class LLMClient:
    """Unified client for interacting with multiple LLM providers.

    This client abstracts away provider-specific details and provides a
    consistent interface for all supported LLMs. It handles:
    - Provider-specific API endpoints and authentication
    - Model name mapping between providers
    - Retry logic for transient failures
    - Integration with PydanticAI for structured outputs

    For providers with OpenAI-compatible APIs (OpenAI, Groq), it uses
    the OpenAI SDK directly. For others (Anthropic, Gemini), it uses
    PydanticAI's built-in support.

    Example:
        >>> # Create client for OpenAI
        >>> client = LLMClient(
        ...     provider=Provider.OPENAI,
        ...     model="gpt-4",
        ...     temperature=0.7
        ... )
        >>>
        >>> # Use with messages
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = await client.complete(messages)
        >>> print(response.content)

    Attributes:
        provider: The LLM provider being used
        model: The model name requested by the user
        temperature: Generation temperature (0.0-2.0)
        client: The underlying API client (OpenAI SDK or similar)
    """

    # Provider base URLs for OpenAI-compatible APIs
    PROVIDER_URLS = {
        Provider.OPENAI: "https://api.openai.com/v1",
        Provider.GROQ: "https://api.groq.com/openai/v1",
        Provider.OLLAMA: "http://localhost:11434/v1",  # Default local Ollama
    }

    # Model mappings
    MODEL_MAPPINGS = {
        Provider.OPENAI: {
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        },
        Provider.GROQ: {
            "gpt-4o-mini": "llama-3.1-8b-instant",
            "gpt-4": "llama-3.1-70b-versatile",
            "mixtral": "mixtral-8x7b-32768",
        },
        Provider.OLLAMA: {
            # Popular Ollama models - pass through model names directly
            "llama3.2": "llama3.2",
            "llama3.2:1b": "llama3.2:1b",
            "llama3.2:3b": "llama3.2:3b",
            "llama3.1": "llama3.1",
            "llama3.1:8b": "llama3.1:8b",
            "llama3.1:70b": "llama3.1:70b",
            "mistral": "mistral",
            "mixtral": "mixtral",
            "qwen2.5-coder": "qwen2.5-coder",
            "qwen2.5": "qwen2.5",
            "gemma2": "gemma2",
            "phi3": "phi3",
            # Map common aliases to Ollama models
            "gpt-4o-mini": "llama3.2:3b",  # Fast, small model
            "gpt-4": "llama3.1:70b",  # Large, capable model
            "gpt-3.5-turbo": "llama3.2",  # Default medium model
        },
    }

    def __init__(
        self,
        provider: Provider,
        model: str,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or self._get_api_key(provider)

        # For OpenAI-compatible providers
        if provider in [Provider.OPENAI, Provider.GROQ, Provider.OLLAMA]:
            base_url = self.PROVIDER_URLS.get(provider)
            # Ollama can use custom base URL from environment
            if provider == Provider.OLLAMA:
                base_url = os.getenv("OLLAMA_BASE_URL", base_url)
                # Ollama doesn't require API key by default
                self._api_key = self._api_key or "ollama"
            self.client = openai.AsyncOpenAI(api_key=self._api_key, base_url=base_url)
        else:
            # For other providers, fallback to OpenAI
            self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_api_key(self, provider: Provider) -> Optional[str]:
        """Retrieve API key for the specified provider from environment.

        Looks for provider-specific environment variables containing
        API keys. This allows users to configure multiple providers
        without code changes.

        Args:
            provider: The provider to get API key for

        Returns:
            API key string if found in environment, None otherwise

        Environment Variables:
            - OPENAI_API_KEY for OpenAI
            - ANTHROPIC_API_KEY for Anthropic
            - GEMINI_API_KEY for Google Gemini
            - GROQ_API_KEY for Groq
            - OLLAMA_API_KEY for Ollama (optional, defaults to "ollama")
        """
        env_keys = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.GEMINI: "GEMINI_API_KEY",
            Provider.GROQ: "GROQ_API_KEY",
            Provider.OLLAMA: "OLLAMA_API_KEY",
        }
        env_var = env_keys.get(provider)
        return os.getenv(env_var) if env_var else None

    def _get_provider_model(self) -> str:
        """Map generic model names to provider-specific identifiers.

        Different providers use different names for similar models.
        This method translates common model names to provider-specific
        ones, allowing users to use familiar names across providers.

        Returns:
            Provider-specific model identifier

        Example:
            - "gpt-4" on Groq maps to "llama-3.1-70b-versatile"
            - "gpt-4o-mini" on Groq maps to "llama-3.1-8b-instant"
        """
        # For Ollama, always pass through the model name as-is (including tags)
        if self.provider == Provider.OLLAMA:
            return self.model

        mappings = self.MODEL_MAPPINGS.get(self.provider, {})
        return mappings.get(self.model, self.model)

    def create_agent(self, system_prompt: str, result_type: type[BaseModel]) -> Agent:
        """Create a PydanticAI agent for structured outputs."""
        # Map provider and model to PydanticAI format
        provider_model = self._get_provider_model()

        # Create model string for PydanticAI
        if self.provider == Provider.OPENAI:
            model_str = f"openai:{provider_model}"
        elif self.provider == Provider.ANTHROPIC:
            model_str = f"anthropic:{provider_model}"
        elif self.provider == Provider.GEMINI:
            model_str = f"gemini:{provider_model}"
        elif self.provider == Provider.GROQ:
            model_str = f"groq:{provider_model}"
        else:  # Provider.OLLAMA
            # For Ollama, use the openai provider with custom base URL
            model_str = f"openai:{provider_model}"

            # Ensure pydantic-ai knows about Ollama's base URL
            # This is required for pydantic-ai to properly route requests
            if self.provider == Provider.OLLAMA:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                os.environ["OPENAI_BASE_URL"] = base_url
                # Ensure we have an API key set for OpenAI provider
                if not os.getenv("OPENAI_API_KEY"):
                    os.environ["OPENAI_API_KEY"] = self._api_key or "ollama"

        # Create agent with structured output
        return Agent(
            model=model_str,
            output_type=result_type,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                temperature=self.temperature,
            ),
        )

    @with_retry(RETRY_STANDARD)
    async def complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Complete a conversation."""
        provider_model = self._get_provider_model()

        try:
            # Cast messages to the expected type
            typed_messages = cast(List[ChatCompletionMessageParam], messages)
            response = await self.client.chat.completions.create(
                model=provider_model,
                messages=typed_messages,
                temperature=kwargs.get("temperature", self.temperature),
                **{k: v for k, v in kwargs.items() if k != "temperature"},
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage={
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                model=provider_model,
            )
        except Exception as e:
            # Convert to appropriate custom exception
            from ..core.exceptions import ModelProviderError

            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise ModelProviderError(
                    "Rate limit exceeded",
                    provider=self.provider.value,
                    error_code="rate_limit",
                ) from e
            elif "api key" in error_msg or "unauthorized" in error_msg:
                raise ModelProviderError(
                    "Authentication failed",
                    provider=self.provider.value,
                    error_code="authentication",
                ) from e
            else:
                raise ModelProviderError(
                    f"LLM API error: {e!s}", provider=self.provider.value
                ) from e


class LLMManager:
    """Manager for LLM clients with connection pooling."""

    _pool: Optional["LLMClientPool"] = None

    @classmethod
    def set_pool(cls, pool: "LLMClientPool") -> None:
        """Set the connection pool for the manager."""
        cls._pool = pool

    @classmethod
    def get_pool(cls) -> "LLMClientPool":
        """Get the connection pool, creating one if needed."""
        if cls._pool is None:
            from .llm_client_pool import get_global_pool

            cls._pool = get_global_pool()
        return cls._pool

    @classmethod
    async def get_client(
        cls,
        provider: Optional[Union[str, Provider]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> LLMClient:
        """Get a client from the connection pool."""
        # Auto-detect provider if not specified
        if provider is None:
            if os.getenv("OPENAI_API_KEY"):
                provider = Provider.OPENAI
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = Provider.ANTHROPIC
            elif os.getenv("GROQ_API_KEY"):
                provider = Provider.GROQ
            elif os.getenv("GEMINI_API_KEY"):
                provider = Provider.GEMINI
            elif os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_BASE_URL"):
                provider = Provider.OLLAMA
            else:
                from .exceptions import ModelProviderError

                raise ModelProviderError(
                    "No LLM provider available for critic. Please set up at least one provider by configuring the appropriate API key.",
                    provider="none",
                    error_code="no_provider",
                )

        # Convert string to Provider enum
        if isinstance(provider, str):
            provider = Provider(provider.lower())

        # Get client from pool
        pool = cls.get_pool()
        return await pool.get_client(provider, model, temperature, api_key)

    @classmethod
    async def return_client(cls, client: LLMClient) -> None:
        """Return a client to the connection pool."""
        pool = cls.get_pool()
        await pool.return_client(client)

    @classmethod
    async def warm_up(
        cls,
        provider: Provider,
        model: str,
        temperature: float = 0.7,
        connections: int = 1,
    ) -> None:
        """Pre-create connections for a provider/model combination."""
        pool = cls.get_pool()
        await pool.warm_up(provider, model, temperature, connections)

    @classmethod
    def get_metrics(cls) -> "ConnectionMetrics":
        """Get connection pool metrics."""
        pool = cls.get_pool()
        return pool.get_metrics()

    @classmethod
    async def close(cls) -> None:
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
