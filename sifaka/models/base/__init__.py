"""
Model Base Package

This package provides the core base implementations for model providers in the Sifaka framework.

## Overview
The model base package serves as the foundation for all language model integrations in Sifaka.
It defines abstract base classes, factory functions, and type definitions that standardize
how model providers are implemented, configured, and used throughout the system.

## Components
- **ModelProvider**: Abstract base class implementing the ModelProviderProtocol
- **create_model_provider**: Factory function for creating model provider instances
- **Type Variables**: Generic type variables for model providers and configurations

## Architecture
The model system follows a layered architecture:

1. **ModelProvider**: High-level interface for model interactions
2. **APIClient**: Low-level communication with model services
3. **TokenCounter**: Utility for token counting
4. **Config**: Configuration and settings management

## Usage Examples
```python
from sifaka.models.base import ModelProvider, create_model_provider
from sifaka.interfaces.model import ModelProviderProtocol

# Basic usage with factory function
provider = create_model_provider(
    ProviderClass,  # Any class implementing ModelProviderProtocol
    model_name="model-name",
    api_key="your-api-key",
    temperature=0.8
)
response = provider.generate("Explain quantum computing in simple terms.")

# Direct instantiation with custom configuration
from sifaka.utils.config.models import ModelConfig
config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
provider = ProviderClass(model_name="model-name", config=config)

# Error handling pattern
try:
    response = provider.generate("Explain quantum computing")
except ValueError as e:
    # Handle input validation errors
    print(f"Input error: {e}")
except RuntimeError as e:
    # Handle API and generation errors
    print(f"Generation failed: {e}")
    # Implement fallback strategy
    fallback_provider = create_model_provider(
        FallbackProviderClass,
        model_name="fallback-model"
    )
    response = fallback_provider.generate("Explain quantum computing briefly")
```

## Error Handling
The model system implements several error handling patterns:

1. **Typed Exceptions**: Use specific exception types for different error cases
   - TypeError: For type validation issues
   - ValueError: For invalid inputs
   - RuntimeError: For operational failures

2. **Automatic Retries**: Implement backoff strategy for transient errors
3. **Graceful Degradation**: Fallback to simpler models when primary fails
4. **Thorough Logging**: Log all errors with context for diagnosis
5. **Tracing**: Record detailed events for monitoring and debugging

## Configuration
Model providers can be configured with:
- **model_name**: Name of the model to use
- **temperature**: Controls randomness (0-1)
- **max_tokens**: Maximum tokens to generate
- **api_key**: API key for authentication
- **trace_enabled**: Enable/disable tracing
"""

# Import from provider module
from .provider import ModelProvider

# Import from factory module
from .factory import create_model_provider

# Import from types module
from .types import T, C

# Define public API
__all__ = [
    # Classes
    "ModelProvider",
    # Factory functions
    "create_model_provider",
    # Type variables
    "T",
    "C",
]
