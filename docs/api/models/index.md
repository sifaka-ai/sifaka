# Models

This page documents the model interfaces and implementations in Sifaka.

## Overview

Sifaka provides a unified interface for working with different LLM providers. The core of this interface is the `Model` protocol, which defines the methods that all model implementations must provide.

## Model Protocol

The `Model` protocol defines the interface that all model implementations must follow.

### Methods

#### `generate`

```python
def generate(self, prompt: str, **options: Any) -> str
```

Generate text from a prompt.

**Parameters:**
- `prompt`: The prompt to generate text from.
- `**options`: Additional options to pass to the model.

**Returns:**
- The generated text.

#### `count_tokens`

```python
def count_tokens(self, text: str) -> int
```

Count tokens in text.

**Parameters:**
- `text`: The text to count tokens in.

**Returns:**
- The number of tokens in the text.

## Creating Models

Sifaka provides a factory function for creating model instances:

```python
def create_model(provider: str, model_name: str, **options: Any) -> Model
```

Create a model instance based on provider and model name.

**Parameters:**
- `provider`: The provider name (e.g., "openai", "anthropic").
- `model_name`: The model name (e.g., "gpt-4", "claude-3").
- `**options`: Additional options to pass to the model constructor.

**Returns:**
- A model instance.

**Raises:**
- `ModelNotFoundError`: If the provider or model is not found.
- `ConfigurationError`: If the required package for the provider is not installed.
- `ModelError`: If there is an error initializing the model.

**Example:**
```python
from sifaka.models import create_model

# Create an OpenAI model
openai_model = create_model("openai", "gpt-4", api_key="your-api-key")

# Create an Anthropic model
anthropic_model = create_model("anthropic", "claude-3-opus", api_key="your-api-key")

# Create a Google Gemini model
gemini_model = create_model("gemini", "gemini-pro", api_key="your-api-key")

# Create a mock model for testing
mock_model = create_model("mock", "test-model")
```

## Available Model Implementations

Sifaka provides implementations for several popular LLM providers:

- [OpenAI](openai.md): Implementation for OpenAI models (GPT-3.5, GPT-4, etc.)
- [Anthropic](anthropic.md): Implementation for Anthropic models (Claude, etc.)
- [Gemini](gemini.md): Implementation for Google Gemini models

## Using Models Directly

While models are typically used through the Chain API, you can also use them directly:

```python
from sifaka.models import create_model

# Create a model
model = create_model("openai", "gpt-4", api_key="your-api-key")

# Generate text
response = model.generate(
    "Write a short story about a robot.",
    temperature=0.7,
    max_tokens=500
)

print(response)

# Count tokens
token_count = model.count_tokens(response)
print(f"Token count: {token_count}")
```

## Creating Custom Model Implementations

You can create custom model implementations by implementing the Model protocol:

```python
from typing import Any
from sifaka.models import Model

class CustomModel:
    """A custom model implementation."""
    
    def __init__(self, model_name: str, **options):
        self.model_name = model_name
        self.options = options
    
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt."""
        # Implement your custom generation logic here
        return f"Generated text for prompt: {prompt}"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Implement your custom token counting logic here
        return len(text.split())

# The CustomModel class can be used anywhere a Model is expected
```

## Error Handling

Model implementations can raise several types of errors:

- `ModelNotFoundError`: If the provider or model is not found.
- `ConfigurationError`: If the required package for the provider is not installed.
- `ModelAPIError`: If there is an error communicating with the model API.
- `ModelError`: Base class for all model-related errors.

It's a good practice to handle these errors when using models directly:

```python
from sifaka.models import create_model
from sifaka.errors import ModelError, ModelAPIError, ConfigurationError

try:
    model = create_model("openai", "gpt-4", api_key="your-api-key")
    response = model.generate("Write a short story about a robot.")
    print(response)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ModelAPIError as e:
    print(f"API error: {e}")
except ModelError as e:
    print(f"Model error: {e}")
```
