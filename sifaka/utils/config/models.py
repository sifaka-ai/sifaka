"""
Model Configuration Module

This module provides configuration classes and standardization functions for model providers.

## Overview
The model configuration module defines configuration classes for different model providers
in the Sifaka framework. It provides a consistent approach to configuring model providers
with standardized parameter handling, validation, and serialization.

## Components
- **ModelConfig**: Base configuration for model providers
- **OpenAIConfig**: Configuration for OpenAI model providers
- **AnthropicConfig**: Configuration for Anthropic model providers
- **GeminiConfig**: Configuration for Google Gemini model providers
- **standardize_model_config**: Standardization function for model configurations

## Usage Examples
```python
from sifaka.utils.config.models import ModelConfig, OpenAIConfig, standardize_model_config

# Create a basic model configuration
config = ModelConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Create an OpenAI-specific configuration
openai_config = OpenAIConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    params={
        "top_p": 0.9,
        "frequency_penalty": 0.5
    }
)

# Use standardization function
config = standardize_model_config(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    params={"top_p": 0.9}
)
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast
from pydantic import Field
from .base import BaseConfig
T = TypeVar('T', bound='ModelConfig')


class ModelConfig(BaseConfig):
    """
    Configuration for model providers.

    This class provides a consistent way to configure model providers across the Sifaka framework.
    It handles common configuration options like temperature and max_tokens, while
    allowing model-specific options through the params dictionary.

    ## Architecture
    ModelConfig extends BaseConfig with model-specific fields:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during provider initialization and
    remain immutable throughout the provider's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Examples
    ```python
    from sifaka.utils.config.models import ModelConfig

    # Create a model configuration
    config = ModelConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        api_key="your-api-key",
        trace_enabled=True
    )

    # Access configuration values
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")

    # Create a new configuration with updated options
    updated_config = (config and config.with_options(temperature=0.9)

    # Create a new configuration with updated params
    updated_config = (config and config.with_params(top_p=0.95)
    ```

    Attributes:
        model: Model name to use
        temperature: Temperature for text generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        api_key: Optional API key for the model provider
        trace_enabled: Whether to enable tracing
    """
    model: str = Field(default='', description='Model name to use')
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description=
        'Temperature for text generation')
    max_tokens: int = Field(default=1000, ge=1, description=
        'Maximum number of tokens to generate')
    api_key: Optional[str] = Field(default=None, description=
        'API key for the model provider')
    trace_enabled: bool = Field(default=False, description=
        'Whether to enable tracing')


class OpenAIConfig(ModelConfig):
    """
    Configuration for OpenAI model providers.

    This class extends ModelConfig with OpenAI-specific configuration options.
    It inherits all the standard configuration options from ModelConfig and
    allows for OpenAI-specific parameters through the params dictionary.

    ## Architecture
    OpenAIConfig is a simple extension of ModelConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during provider initialization
    and remain immutable throughout the provider's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## OpenAI-Specific Parameters
    Common OpenAI-specific parameters that can be included in the params dictionary:
    - **top_p**: Nucleus sampling parameter (0.0 to 1.0)
    - **frequency_penalty**: Penalty for token frequency (0.0 to 2.0)
    - **presence_penalty**: Penalty for token presence (0.0 to 2.0)
    - **stop**: List of strings that stop generation when encountered
    - **logit_bias**: Dictionary of token biases

    Examples:
        ```python
        from sifaka.utils.config.models import OpenAIConfig

        # Create an OpenAI configuration
        config = OpenAIConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.0,
                "stop": ["
", "###"],
                "logit_bias": {50256: -100}  # Bias against a specific token
            }
        )

        # Use the configuration with an OpenAI provider
        provider = OpenAIProvider(model_name="gpt-4", config=config)

        # Create a new configuration with updated options
        updated_config = (config and config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = (config and config.with_params(
            top_p=0.95,
            frequency_penalty=0.7
        )
        ```
    """
    pass


class AnthropicConfig(ModelConfig):
    """
    Configuration for Anthropic model providers.

    This class extends ModelConfig with Anthropic-specific configuration options.
    It inherits all the standard configuration options from ModelConfig and
    allows for Anthropic-specific parameters through the params dictionary.

    ## Architecture
    AnthropicConfig is a simple extension of ModelConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during provider initialization
    and remain immutable throughout the provider's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Anthropic-Specific Parameters
    Common Anthropic-specific parameters that can be included in the params dictionary:
    - **top_k**: Number of tokens to consider for sampling
    - **top_p**: Nucleus sampling parameter (0.0 to 1.0)
    - **stop_sequences**: List of strings that stop generation when encountered
    - **system_prompt**: System prompt to control Claude's behavior
    - **anthropic_version**: API version to use

    Examples:
        ```python
        from sifaka.utils.config.models import AnthropicConfig

        # Create an Anthropic configuration
        config = AnthropicConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_k": 50,
                "top_p": 0.9,
                "stop_sequences": ["

Human:", "

Assistant:"],
                "system_prompt": "You are Claude, an AI assistant created by Anthropic.",
                "anthropic_version": "2023-06-01"
            }
        )

        # Use the configuration with an Anthropic provider
        provider = AnthropicProvider(model_name="claude-3-opus", config=config)

        # Create a new configuration with updated options
        updated_config = (config and config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = (config and config.with_params(
            top_p=0.95,
            system_prompt="You are Claude, a helpful AI assistant."
        )
        ```
    """
    pass


class GeminiConfig(ModelConfig):
    """
    Configuration for Google Gemini model providers.

    This class extends ModelConfig with Gemini-specific configuration options.
    It inherits all the standard configuration options from ModelConfig and
    allows for Gemini-specific parameters through the params dictionary.

    ## Architecture
    GeminiConfig is a simple extension of ModelConfig that maintains the same
    architecture and validation patterns. It uses Pydantic for validation and
    serialization, with immutable configuration objects.

    ## Lifecycle
    Configuration objects are typically created during provider initialization
    and remain immutable throughout the provider's lifecycle. New configurations
    can be created from existing ones using the with_options and with_params methods.

    ## Gemini-Specific Parameters
    Common Gemini-specific parameters that can be included in the params dictionary:
    - **top_k**: Number of tokens to consider for sampling
    - **top_p**: Nucleus sampling parameter (0.0 to 1.0)
    - **stop_sequences**: List of strings that stop generation when encountered
    - **safety_settings**: Dictionary of safety settings
    - **candidate_count**: Number of candidate responses to generate

    Examples:
        ```python
        from sifaka.utils.config.models import GeminiConfig

        # Create a Gemini configuration
        config = GeminiConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_k": 40,
                "top_p": 0.95,
                "stop_sequences": ["###"],
                "safety_settings": {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
                },
                "candidate_count": 1
            }
        )

        # Use the configuration with a Gemini provider
        provider = GeminiProvider(model_name="gemini-pro", config=config)

        # Create a new configuration with updated options
        updated_config = (config and config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = (config and config.with_params(
            top_p=0.98,
            candidate_count=3
        )
        ```
    """
    pass


def standardize_model_config(config: Optional[Union[Dict[str, Any],
    ModelConfig]]=None, params: Optional[Dict[str, Any]]=None, config_class:
    Type[T]=ModelConfig, **kwargs: Any) ->Any:
    """
    Standardize model provider configuration.

    This utility function ensures that model provider configuration is consistently
    handled across the framework. It accepts various input formats and
    returns a standardized ModelConfig object or a subclass.

    Args:
        config: Optional configuration (either a dict or ModelConfig)
        params: Optional params dictionary to merge with config
        config_class: The config class to use (default: ModelConfig)
        **kwargs: Additional parameters to include in the config

    Returns:
        Standardized ModelConfig object or subclass

    Examples:
        ```python
        from sifaka.utils.config.models import standardize_model_config, OpenAIConfig

        # Create from parameters
        config = standardize_model_config(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            params={"top_p": 0.9}
        )

        # Create from existing config
        existing = ModelConfig(temperature=0.7)
        updated = standardize_model_config(
            config=existing,
            params={"top_p": 0.9}
        )

        # Create from dictionary
        dict_config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "params": {"top_p": 0.9}
        }
        config = standardize_model_config(config=dict_config)

        # Create specialized config
        openai_config = standardize_model_config(
            config_class=OpenAIConfig,
            model="gpt-4",
            temperature=0.7,
            params={"top_p": 0.9}
        )
        ```
    """
    final_params: Dict[str, Any] = {}
    if params:
        (final_params and final_params.update(params)
    if isinstance(config, dict):
        dict_params = (config and config.pop('params', {}) if config else {}
        (final_params and final_params.update(dict_params)
        return cast(T, config_class(**{} if config is None else config,
            params=final_params, **kwargs))
    elif isinstance(config, ModelConfig):
        (final_params and final_params.update(config.params)
        config_dict = {**(config and config.model_dump(), 'params': final_params, **kwargs}
        return cast(T, config_class(**config_dict))
    else:
        return cast(T, config_class(params=final_params, **kwargs))
