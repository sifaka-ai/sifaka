# Simplified Dependency Injection System

Sifaka uses a dependency injection system to decouple components and eliminate circular import issues. This document explains how the simplified system works and how to use it.

## Overview

The dependency injection system consists of three main parts:

1. **Registry**: A central registry for component registration and retrieval
2. **Factories**: Factory functions for creating components
3. **Interfaces**: Protocol interfaces that define the contract between components

## Registry System

The registry system provides a way to register and retrieve component factories without direct imports. It is implemented in the `sifaka.registry` module.

### Component Types

The registry supports different types of components:

- **Models**: LLM providers like OpenAI, Anthropic, etc.
- **Validators**: Components that check if text meets specific criteria
- **Improvers**: Components that enhance the quality of text

### Registration

Components can be registered with the registry using decorators:

```python
from sifaka.registry import register_model, register_validator, register_improver

# Register a model factory
@register_model("openai")
def create_openai_model(model_name, **options):
    return OpenAIModel(model_name, **options)

# Register a validator factory
@register_validator("length")
def create_length_validator(**options):
    return LengthValidator(**options)

# Register an improver factory
@register_improver("clarity")
def create_clarity_improver(model, **options):
    return ClarityImprover(model, **options)
```

### Retrieval

Components can be retrieved from the registry using the `get_*_factory` functions:

```python
from sifaka.registry import get_model_factory, get_validator_factory, get_improver_factory

# Get a model factory
openai_factory = get_model_factory("openai")
model = openai_factory("gpt-4", temperature=0.7)

# Get a validator factory
length_factory = get_validator_factory("length")
validator = length_factory(min_words=50, max_words=200)

# Get an improver factory
clarity_factory = get_improver_factory("clarity")
improver = clarity_factory(model, level="high")
```

## Factory Functions

The `sifaka.factories` module provides high-level factory functions that use the registry to create components:

```python
from sifaka.factories import create_model, create_validator, create_improver

# Create a model
model = create_model("openai", "gpt-4", temperature=0.7)

# Create a model from a string
model = create_model_from_string("openai:gpt-4", temperature=0.7)

# Create a validator
validator = create_validator("length", min_words=50, max_words=200)

# Create an improver
improver = create_improver("clarity", model, level="high")
```

## Lazy Loading

The registry system uses lazy loading to prevent circular imports. Components are only imported when they are needed, not when the registry is initialized.

When you call `get_model_factory("openai")`, the registry will:

1. Check if the "model" component type has been initialized
2. If not, import all modules registered for the "model" component type
3. Return the factory function for the "openai" provider

## Dependency Injection

The Chain class and other components use dependency injection to avoid circular imports. Instead of importing dependencies directly, they accept them through constructors.

### Chain Class

The Chain class accepts a model factory through its constructor:

```python
from sifaka import Chain
from sifaka.factories import create_model

# Using the default factory
chain = Chain()

# Using a custom factory
chain = Chain(model_factory=create_model)

# Using a custom implementation
def my_model_factory(provider, model_name, **options):
    if provider == "custom":
        return CustomModel(model_name, **options)
    return create_model(provider, model_name, **options)

chain = Chain(model_factory=my_model_factory)
```

## Complete Example

Here's a complete example of using the dependency injection system:

```python
from sifaka import Chain
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

# Create a chain with the custom model
chain = Chain().with_model("custom:my-model").with_prompt("Tell me a story")

# Run the chain
result = chain.run()
print(result.text)
```

## Benefits

The simplified dependency injection system provides several benefits:

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
