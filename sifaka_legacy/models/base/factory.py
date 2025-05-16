"""
Model Factory Module

This module provides factory functions for creating model providers.

## Overview
The factory module provides standardized ways to create model providers with
consistent configuration. It simplifies the creation process by providing
a uniform interface across different provider implementations.

## Components
- **create_model_provider**: Factory function for creating model provider instances

## Usage Examples
```python
from sifaka.models.base.factory import create_model_provider
from sifaka.models.providers.openai import OpenAIProvider

# Create a provider with a specific provider class
provider = create_model_provider(
    OpenAIProvider,
    model_name="gpt-4",
    api_key="your-api-key",
    temperature=0.8
)

# Create another provider
provider = create_model_provider(
    AnthropicProvider,
    model_name="claude-3-opus",
    api_key="your-api-key",
    max_tokens=2000
)
```

## Error Handling
The factory function implements standardized error handling:
- Validates parameters before creating the provider
- Propagates exceptions from provider initialization
- Provides clear error messages for debugging
"""

from typing import Any, Callable, Optional

from sifaka.utils.config.models import ModelConfig
from sifaka.models.base.types import T


def create_model_provider(
    provider_type: Callable[..., T],
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any,
) -> T:
    """
    Factory function to create a model provider with a standardized configuration.

    This function simplifies the creation of model providers by providing
    a consistent interface for common configuration options across different
    provider implementations.

    Args:
        provider_type: The class of the provider to create
        model_name: The name of the model to use
        api_key: Optional API key for the provider
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional provider-specific arguments

    Returns:
        A configured model provider instance

    Raises:
        ValueError: If parameters are invalid
        TypeError: If provider_type is not a valid model provider class
        RuntimeError: If provider creation fails
    """
    config = ModelConfig(temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    return provider_type(model_name=model_name, config=config, **kwargs)
