# Sifaka Model Providers

This package provides model provider implementations for different LLM services, along with a component-based architecture for better separation of concerns.

## Architecture

The model provider architecture follows the Single Responsibility Principle by breaking down functionality into smaller, focused components:

```
ModelProviderCore
├── ClientManager
├── TokenCounterManager
├── TracingManager
└── GenerationService
```

### Core Components

- **ModelProviderCore**: Main interface that delegates to specialized components
- **ClientManager**: Manages API client creation and management
- **TokenCounterManager**: Manages token counting functionality
- **TracingManager**: Manages tracing and logging
- **GenerationService**: Handles text generation and error handling

### Provider Implementations

- **OpenAIProvider**: Provider for OpenAI models
- **AnthropicProvider**: Provider for Anthropic models
- **GeminiProvider**: Provider for Google Gemini models

## Usage

### Basic Usage

```python
from sifaka.models import OpenAIProvider, ModelConfig

# Create a provider with default configuration
provider = OpenAIProvider(model_name="gpt-4")

# Create a provider with custom configuration
provider = OpenAIProvider(
    model_name="gpt-4",
    config=ModelConfig(
        temperature=0.7,
        max_tokens=1000,
        api_key="your-api-key",
        trace_enabled=True,
    ),
)

# Generate text
text = provider.generate("Hello, world!")

# Count tokens
token_count = provider.count_tokens("Hello, world!")
```

### Advanced Usage

```python
from sifaka.models import OpenAIProvider, ModelConfig
from sifaka.models.managers import TracingManager
from sifaka.utils.tracing import Tracer

# Create a custom tracer
tracer = Tracer()

# Create a provider with custom components
provider = OpenAIProvider(
    model_name="gpt-4",
    config=ModelConfig(
        temperature=0.7,
        max_tokens=1000,
        api_key="your-api-key",
        trace_enabled=True,
    ),
    tracer=tracer,
)

# Generate text with config overrides
text = provider.generate(
    "Hello, world!",
    temperature=0.5,
    max_tokens=500,
)
```

## Extending

To create a new model provider, extend the `ModelProviderCore` class and implement the required methods:

```python
from sifaka.models.base import APIClient, TokenCounter
from sifaka.models.core.provider import ModelProviderCore

class MyCustomProvider(ModelProviderCore):
    def _create_default_client(self) -> APIClient:
        # Create and return a default API client
        return MyCustomClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        # Create and return a default token counter
        return MyCustomTokenCounter(model=self.model_name)
```
