# Registry System

The registry system is a key part of Sifaka's architecture. It provides a way to register and retrieve components such as models, validators, and improvers (critics) without creating circular dependencies.

## Overview

The registry system consists of three main parts:

1. **Registry**: A central registry that stores factory functions for different component types
2. **Decorators**: Decorators for registering factory functions with the registry
3. **Factory Functions**: High-level functions for creating components using the registry

## Component Types

The registry supports three main component types:

1. **Models**: Components that generate text (e.g., OpenAI, Anthropic, Gemini)
2. **Validators**: Components that validate text (e.g., length, content, format)
3. **Improvers**: Components that improve text (e.g., LAC, constitutional, reflexion)

## Registering Components

Components are registered using decorators:

```python
from sifaka.registry import register_model, register_validator, register_improver

# Register a model factory
@register_model("openai")
def create_openai_model(model_name, **options):
    return OpenAIModel(model_name, **options)

# Register a validator factory
@register_validator("length")
def create_length_validator(min_words=None, max_words=None, **options):
    return LengthValidator(min_words=min_words, max_words=max_words)

# Register an improver factory
@register_improver("lac")
def create_lac_critic(model, **options):
    return LACCritic(model, **options)
```

## Creating Components

Components can be created using factory functions:

```python
from sifaka.factories import create_model, create_validator, create_improver

# Create a model
model = create_model("openai", "gpt-4", temperature=0.7)

# Create a validator
validator = create_validator("length", min_words=50, max_words=200)

# Create an improver
improver = create_improver("lac", model, level="high")
```

You can also create models from a string specification:

```python
from sifaka.factories import create_model_from_string

# Create a model from a string
model = create_model_from_string("openai:gpt-4", temperature=0.7)
```

## Using Components with Chain

The Chain class can use components created with the registry:

```python
from sifaka import Chain

# Create a chain with a model
chain = Chain().with_model("openai:gpt-4").with_prompt("Tell me a story")

# Add a validator
chain.validate_with(create_validator("length", min_words=50, max_words=200))

# Add an improver
chain.improve_with(create_improver("lac", create_model("openai", "gpt-4")))

# Run the chain
result = chain.run()
```

## Lazy Loading

The registry uses lazy loading to prevent circular imports. Components are only imported when they are needed, not when the registry is initialized.

When you call `get_model_factory("openai")`, the registry will:

1. Check if the "model" component type has been initialized
2. If not, import all modules registered for the "model" component type
3. Return the factory function for the "openai" provider

## Custom Components

You can create custom components and register them with the registry:

```python
from sifaka.registry import register_model
from sifaka.interfaces import Model

# Define a custom model
class CustomModel:
    def __init__(self, model_name, **options):
        self.model_name = model_name
        self.options = options
    
    def generate(self, prompt, **options):
        return f"Custom response for: {prompt}"
    
    def count_tokens(self, text):
        return len(text.split())

# Register the custom model
@register_model("custom")
def create_custom_model(model_name, **options):
    return CustomModel(model_name, **options)

# Use the custom model
model = create_model("custom", "my-model")
```

## Benefits

The registry system provides several benefits:

1. **Eliminates Circular Imports**: Components don't need to import each other directly
2. **Improves Testability**: Dependencies can be easily mocked for testing
3. **Enhances Extensibility**: New components can be added without modifying existing code
4. **Simplifies Configuration**: Components can be configured with different implementations
5. **Standardizes Component Registration**: Uses a single, consistent approach to registration
6. **Reduces Complexity**: Consolidates functionality into fewer files with a cleaner API

## Implementation Details

The registry system is implemented in the following files:

- `sifaka/registry.py`: The main registry implementation
- `sifaka/factories.py`: High-level factory functions
- `sifaka/interfaces.py`: Protocol interfaces for components
- `sifaka/models/*.py`: Model implementations
- `sifaka/validators/*.py`: Validator implementations
- `sifaka/critics/*.py`: Critic (improver) implementations
