"""Simplified provider-agnostic LLM client with PydanticAI integration."""

import os
from typing import Dict, Optional, List, Union, Any, cast
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
import logfire

from ..core.retry import with_retry, RETRY_STANDARD


# Load environment variables
load_dotenv()

# Configure logfire if token is available
if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"


class LLMResponse(BaseModel):
    """Response from LLM."""

    content: str
    usage: Dict[str, int] = Field(default_factory=dict)
    model: str


class LLMClient:
    """Simple provider-agnostic LLM client using OpenAI API compatibility."""

    # Provider base URLs for OpenAI-compatible APIs
    PROVIDER_URLS = {
        Provider.OPENAI: "https://api.openai.com/v1",
        Provider.GROQ: "https://api.groq.com/openai/v1",
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
        if provider in [Provider.OPENAI, Provider.GROQ]:
            base_url = self.PROVIDER_URLS.get(provider)
            self.client = openai.AsyncOpenAI(api_key=self._api_key, base_url=base_url)
        else:
            # For other providers, fallback to OpenAI
            self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_api_key(self, provider: Provider) -> Optional[str]:
        """Get API key for provider."""
        env_keys = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.GEMINI: "GEMINI_API_KEY",
            Provider.GROQ: "GROQ_API_KEY",
        }
        env_var = env_keys.get(provider)
        return os.getenv(env_var) if env_var else None

    def _get_provider_model(self) -> str:
        """Get provider-specific model name."""
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
        else:
            # Fallback to OpenAI
            model_str = f"openai:{provider_model}"

        # Create agent with structured output
        return Agent(
            model=model_str,
            output_type=result_type,  # Use output_type instead of deprecated result_type
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                temperature=self.temperature,
                api_key=self._api_key,
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
                    f"LLM API error: {str(e)}", provider=self.provider.value
                ) from e


class LLMManager:
    """Manager for LLM clients."""

    _clients: Dict[str, LLMClient] = {}

    @classmethod
    def get_client(
        cls,
        provider: Optional[Union[str, Provider]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> LLMClient:
        """Get or create an LLM client."""
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
            else:
                raise ValueError(
                    "No API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GEMINI_API_KEY"
                )

        # Convert string to Provider enum
        if isinstance(provider, str):
            provider = Provider(provider.lower())

        # Create client key
        client_key = f"{provider}:{model}:{temperature}"

        # Get or create client
        if client_key not in cls._clients:
            cls._clients[client_key] = LLMClient(provider, model, temperature, api_key)

        return cls._clients[client_key]
