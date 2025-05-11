"""
Configuration classes for model providers.

This module imports standardized configuration classes from utils/config.py and
extends them with model-specific functionality.

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
"""

from sifaka.utils.config import ModelConfig


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

    pass


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

    pass


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

    pass
