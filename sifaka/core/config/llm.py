"""LLM-specific configuration."""

from typing import Optional

from pydantic import Field

from .base import BaseConfig


class LLMConfig(BaseConfig):
    """Configuration for Large Language Model settings.

    Controls model selection, generation parameters, and provider-specific
    settings for both main generation and critic models.

    Example:
        >>> llm_config = LLMConfig(
        ...     model="gpt-4",
        ...     temperature=0.7,
        ...     provider="openai"
        ... )
    """

    # Primary model settings
    model: str = Field(
        default="gpt-4o-mini", description="LLM model to use for text generation"
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature (0.0 = deterministic, 2.0 = very creative)",
    )

    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        le=128000,
        description="Maximum tokens to generate (None = model default)",
    )

    provider: Optional[str] = Field(
        default=None,
        description="LLM provider (openai, anthropic, gemini, groq). Auto-detected if None.",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider. Uses environment variable if None.",
    )

    # Critic model settings (optional overrides)
    critic_model: Optional[str] = Field(
        default="gpt-3.5-turbo",
        description="Model for critics (defaults to a cheaper/faster model)",
    )

    critic_temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for critics (defaults to main temperature)",
    )

    # Timeout settings
    timeout_seconds: float = Field(
        default=30.0, gt=0, le=300, description="Timeout for LLM API calls in seconds"
    )

    # Connection pooling (future enhancement)
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent connections to LLM provider",
    )

    @property
    def effective_critic_temperature(self) -> float:
        """Get the effective temperature for critics."""
        return (
            self.critic_temperature
            if self.critic_temperature is not None
            else self.temperature
        )

    def model_config_dict(self) -> dict:
        """Get model configuration as a dictionary for LLM clients."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout_seconds,
        }
