"""
Model Provider Configuration

This module provides specialized configuration classes for different model providers
in the Sifaka framework, extending the base ModelConfig from utils/config.py with
provider-specific functionality.

## Overview
The configuration classes in this module provide a consistent way to configure
model providers across the Sifaka framework. Each provider has its own configuration
class that extends the base ModelConfig with provider-specific options and defaults.

## Components
- **OpenAIConfig**: Configuration for OpenAI model providers
- **AnthropicConfig**: Configuration for Anthropic model providers
- **GeminiConfig**: Configuration for Google Gemini model providers

## Usage Examples
```python
from sifaka.models.config import OpenAIConfig, AnthropicConfig, GeminiConfig
from sifaka.utils.config import standardize_model_config

# Create a basic model configuration
config = standardize_model_config(
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

# Create specialized provider configurations
openai_config = OpenAIConfig(
    temperature=0.7,
    max_tokens=1000,
    params={"top_p": 0.9}
)

anthropic_config = AnthropicConfig(
    temperature=0.7,
    max_tokens=1000,
    params={"top_k": 50}
)

gemini_config = GeminiConfig(
    temperature=0.7,
    max_tokens=1000,
    params={"top_k": 40}
)
```

## Error Handling
Configuration validation is handled by Pydantic, which ensures that all configuration
values are valid and properly typed. If invalid configuration is provided, Pydantic
will raise validation errors with detailed information about the validation failure.

## Configuration
All configuration classes in this module extend the base ModelConfig from utils/config.py,
which provides common configuration options like temperature and max_tokens. Provider-specific
options are added through the params dictionary.
"""

from sifaka.utils.config import ModelConfig


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
        from sifaka.models.config import OpenAIConfig

        # Create an OpenAI configuration
        config = OpenAIConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_p": 0.9,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.0,
                "stop": ["\n", "###"],
                "logit_bias": {50256: -100}  # Bias against a specific token
            }
        )

        # Use the configuration with an OpenAI provider
        provider = OpenAIProvider(model_name="gpt-4", config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = config.with_params(
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
        from sifaka.models.config import AnthropicConfig

        # Create an Anthropic configuration
        config = AnthropicConfig(
            temperature=0.7,
            max_tokens=1000,
            params={
                "top_k": 50,
                "top_p": 0.9,
                "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"],
                "system_prompt": "You are Claude, an AI assistant created by Anthropic.",
                "anthropic_version": "2023-06-01"
            }
        )

        # Use the configuration with an Anthropic provider
        provider = AnthropicProvider(model_name="claude-3-opus", config=config)

        # Create a new configuration with updated options
        updated_config = config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = config.with_params(
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
        from sifaka.models.config import GeminiConfig

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
        updated_config = config.with_options(temperature=0.9)

        # Create a new configuration with updated params
        updated_config = config.with_params(
            top_p=0.98,
            candidate_count=3
        )
        ```
    """

    pass
