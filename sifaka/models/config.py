"""
Configuration classes for model providers.

This module provides standardized configuration classes for model providers,
following the same pattern as RuleConfig and ClassifierConfig.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """
    Immutable configuration for model providers.

    This class provides a consistent way to configure model providers across the Sifaka framework.
    It handles common configuration options like temperature and max_tokens, while
    allowing model-specific options through the params dictionary.

    Lifecycle:
        1. Creation: Instantiated with configuration options
        2. Usage: Accessed by model providers during setup and generation
        3. Modification: New instances created with updated options (immutable pattern)
        4. Extension: Specialized config classes can extend this base class

    Examples:
        ```python
        from sifaka.models.config import ModelConfig

        # Create a basic model configuration
        config = ModelConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "system_prompt": "You are a helpful assistant.",
                "top_p": 0.9,
            }
        )

        # Use the configuration with a model provider
        provider = OpenAIProvider(model_name="gpt-4", config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        parameterized_config = config.with_params(system_prompt="You are an expert coder.")

        # Access configuration values
        print(f"Temperature: {config.temperature}")
        print(f"System prompt: {config.params.get('system_prompt')}")
        ```

    Attributes:
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        api_key: Optional API key for the model provider
        trace_enabled: Whether to enable tracing
        params: Dictionary of model-specific configuration parameters
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens to generate",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider",
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether to enable tracing",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific configuration parameters",
    )

    def with_options(self, **kwargs: Any) -> "ModelConfig":
        """
        Create a new config with updated options.

        This method is useful for updating top-level configuration
        options without modifying the params dictionary.

        Args:
            **kwargs: Options to update

        Returns:
            New config with updated options
        """
        return ModelConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "ModelConfig":
        """
        Create a new config with updated params.

        This method is useful for updating or adding model-specific
        parameters without modifying other configuration options.

        Args:
            **kwargs: Params to update

        Returns:
            New config with updated params
        """
        return ModelConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )

    def with_temperature(self, temperature: float) -> "ModelConfig":
        """
        Create a new config with the specified temperature.

        Args:
            temperature: The new temperature value (0.0 to 1.0)

        Returns:
            A new ModelConfig with the updated temperature

        Raises:
            ValueError: If temperature is outside the valid range
        """
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {temperature}")
        return self.with_options(temperature=temperature)

    def with_max_tokens(self, max_tokens: int) -> "ModelConfig":
        """
        Create a new config with the specified max_tokens.

        Args:
            max_tokens: The new max_tokens value (must be positive)

        Returns:
            A new ModelConfig with the updated max_tokens

        Raises:
            ValueError: If max_tokens is not positive
        """
        if max_tokens <= 0:
            raise ValueError(f"Max tokens must be positive, got {max_tokens}")
        return self.with_options(max_tokens=max_tokens)

    def with_api_key(self, api_key: Optional[str]) -> "ModelConfig":
        """
        Create a new config with the specified API key.

        Args:
            api_key: The new API key

        Returns:
            A new ModelConfig with the updated API key
        """
        return self.with_options(api_key=api_key)


class OpenAIConfig(ModelConfig):
    """
    Configuration for OpenAI model providers.

    This class extends ModelConfig with OpenAI-specific configuration options.

    Examples:
        ```python
        from sifaka.models.config import OpenAIConfig

        # Create an OpenAI configuration
        config = OpenAIConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_p": 0.9,
                "frequency_penalty": 0.5,
            }
        )

        # Use the configuration with an OpenAI provider
        provider = OpenAIProvider(model_name="gpt-4", config=config)
        ```
    """

    def with_options(self, **kwargs: Any) -> "OpenAIConfig":
        """Create a new config with updated options."""
        return OpenAIConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "OpenAIConfig":
        """Create a new config with updated params."""
        return OpenAIConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )


class AnthropicConfig(ModelConfig):
    """
    Configuration for Anthropic model providers.

    This class extends ModelConfig with Anthropic-specific configuration options.

    Examples:
        ```python
        from sifaka.models.config import AnthropicConfig

        # Create an Anthropic configuration
        config = AnthropicConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_k": 50,
                "top_p": 0.9,
            }
        )

        # Use the configuration with an Anthropic provider
        provider = AnthropicProvider(model_name="claude-3-opus", config=config)
        ```
    """

    def with_options(self, **kwargs: Any) -> "AnthropicConfig":
        """Create a new config with updated options."""
        return AnthropicConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "AnthropicConfig":
        """Create a new config with updated params."""
        return AnthropicConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )


class GeminiConfig(ModelConfig):
    """
    Configuration for Google Gemini model providers.

    This class extends ModelConfig with Gemini-specific configuration options.

    Examples:
        ```python
        from sifaka.models.config import GeminiConfig

        # Create a Gemini configuration
        config = GeminiConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_k": 40,
                "top_p": 0.95,
            }
        )

        # Use the configuration with a Gemini provider
        provider = GeminiProvider(model_name="gemini-pro", config=config)
        ```
    """

    def with_options(self, **kwargs: Any) -> "GeminiConfig":
        """Create a new config with updated options."""
        return GeminiConfig(**{**self.model_dump(), **kwargs})

    def with_params(self, **kwargs: Any) -> "GeminiConfig":
        """Create a new config with updated params."""
        return GeminiConfig(
            **{
                **self.model_dump(exclude={"params"}),
                "params": {**self.params, **kwargs},
            }
        )
