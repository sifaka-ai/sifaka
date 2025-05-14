# Sifaka Models

This package provides model provider implementations for different LLM services, enabling standardized text generation capabilities across the Sifaka framework.

## Architecture

The models architecture follows a component-based design for maximum flexibility and extensibility:

```
ModelProviderCore
├── Client Components
│   ├── ClientManager (manages API client creation)
│   └── APIClient (handles API communication)
├── Token Management
│   ├── TokenCounterManager (manages token counters)
│   └── TokenCounter (handles token counting)
├── Execution Support
│   ├── TracingManager (handles execution tracing)
│   ├── GenerationService (coordinates text generation)
│   └── StateManager (manages provider state)
└── Model Providers
    ├── OpenAIProvider (OpenAI models)
    ├── AnthropicProvider (Anthropic models)
    ├── GeminiProvider (Google Gemini models)
    └── MockProvider (Testing and simulation)
```

## Core Components

- **ModelProviderCore**: Foundation implementation used by all model providers
- **Managers**: Specialized components for client, token counting, and tracing
- **Services**: Components for generation and error handling
- **Results**: Standardized result objects for generations and token counts
- **Factories**: Factory functions for creating model providers with sensible defaults

## Model Providers

- **OpenAIProvider**: Provider for OpenAI models (GPT-3.5, GPT-4, etc.)
- **AnthropicProvider**: Provider for Anthropic models (Claude, etc.)
- **GeminiProvider**: Provider for Google Gemini models
- **MockProvider**: Provider for testing and simulation

## Usage

### Basic Usage

```python
from sifaka.models import OpenAIProvider, AnthropicProvider, GeminiProvider

# Create an OpenAI provider
openai_provider = OpenAIProvider(model_name="gpt-4")

# Create an Anthropic provider
anthropic_provider = AnthropicProvider(model_name="claude-3-opus-20240229")

# Create a Gemini provider
gemini_provider = GeminiProvider(model_name="gemini-1.0-pro")

# Generate text
openai_result = openai_provider.generate("Write a short story about a robot.")
claude_result = anthropic_provider.generate("Explain quantum computing in simple terms.")
gemini_result = gemini_provider.generate("Provide three interesting facts about space.")

# Count tokens
token_count = openai_provider.count_tokens("How many tokens is this?")
```

### Using Factory Functions

Factory functions provide a standardized way to create model providers with sensible defaults:

```python
from sifaka.models import create_openai_provider, create_anthropic_provider, create_gemini_provider

# Create providers using factories
openai = create_openai_provider(
    model_name="gpt-4",
    api_key="your-openai-api-key",
    temperature=0.7,
    max_tokens=1000
)

anthropic = create_anthropic_provider(
    model_name="claude-3-opus-20240229",
    api_key="your-anthropic-api-key",
    temperature=0.5,
    max_tokens=2000
)

gemini = create_gemini_provider(
    model_name="gemini-1.0-pro",
    api_key="your-google-api-key",
    temperature=0.8,
    max_tokens=1500
)
```

### Custom Configuration

Model providers can be configured with model-specific parameters:

```python
from sifaka.models import OpenAIProvider
from sifaka.utils.config.models import OpenAIConfig

# Create configuration
config = OpenAIConfig(
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    api_key="your-api-key",
    trace_enabled=True
)

# Create provider with custom configuration
provider = OpenAIProvider(
    model_name="gpt-4",
    config=config
)

# Generate with per-call overrides
response = provider.generate(
    "Write a story about a robot",
    temperature=0.5,  # Override just for this call
    max_tokens=500    # Override just for this call
)
```

### Async Generation

Model providers support asynchronous generation for non-blocking operations:

```python
import asyncio
from sifaka.models import OpenAIProvider

# Create provider
provider = OpenAIProvider(model_name="gpt-4")

# Define async function
async def generate_texts(prompts):
    results = []
    for prompt in prompts:
        # Use ainvoke for async generation
        result = await provider.ainvoke(prompt)
        results.append(result)
    return results

# Run async function
prompts = [
    "Write a short story",
    "Explain quantum computing",
    "Give me three healthy recipes"
]
results = asyncio.run(generate_texts(prompts))
```

### Provider Statistics

Model providers track statistics about their operations:

```python
from sifaka.models import OpenAIProvider

# Create provider
provider = OpenAIProvider(model_name="gpt-4")

# Generate some text
provider.generate("Hello world")
provider.generate("How are you?")

# Get statistics
stats = provider.get_statistics()
print(f"Generation count: {stats['generation_count']}")
print(f"Token count calls: {stats['token_count_calls']}")
print(f"Error count: {stats['error_count']}")
print(f"Average processing time: {stats['avg_processing_time']}ms")
```

## Extending

### Creating a Custom Model Provider

You can create a custom model provider by extending the `ModelProviderCore` class:

```python
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.models.core.provider import ModelProviderCore
from sifaka.utils.config.models import ModelConfig

class MyCustomProvider(ModelProviderCore):
    """Custom model provider implementation."""

    DEFAULT_MODEL = "my-custom-model"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: ModelConfig = None,
        api_client: APIClient = None,
        token_counter: TokenCounter = None,
    ):
        # Initialize the core provider
        super().__init__(
            model_name=model_name,
            config=config or ModelConfig(),
            api_client=api_client,
            token_counter=token_counter,
        )

    def _create_default_client(self) -> APIClient:
        """Create default API client."""
        from my_custom_module import MyCustomClient
        return MyCustomClient(
            api_key=self.config.api_key,
            base_url=self.config.params.get("base_url", "https://api.example.com"),
        )

    def _create_default_token_counter(self) -> TokenCounter:
        """Create default token counter."""
        from my_custom_module import MyCustomTokenCounter
        return MyCustomTokenCounter(model=self.model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        # This method just calls the ModelProviderCore's invoke method
        return self.invoke(prompt, **kwargs)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Get token counter from state
        token_counter = self._state_manager.get("token_counter")
        if token_counter:
            return token_counter.count_tokens(text)
        return 0
```

### Creating a Factory Function

To simplify creation of your custom provider, create a factory function:

```python
from typing import Any, Dict, Optional
from sifaka.models.core.provider import ModelProviderCore
from sifaka.utils.config.models import ModelConfig

def create_custom_provider(
    model_name: str = "my-custom-model",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs: Any
) -> ModelProviderCore:
    """
    Create a custom model provider with sensible defaults.

    Args:
        model_name: Name of the model to use
        api_key: API key for authentication
        temperature: Controls randomness (0.0-1.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration parameters

    Returns:
        A configured custom model provider
    """
    from my_module import MyCustomProvider

    # Create configuration
    config_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }

    config = ModelConfig(
        api_key=api_key,
        params=config_params
    )

    # Create and return provider
    return MyCustomProvider(
        model_name=model_name,
        config=config
    )
```
