# Registry System Tutorial

This tutorial explains how to use Sifaka's registry system to register and use custom components.

## Introduction

The registry system is a key part of Sifaka's architecture. It provides a way to register and retrieve components such as models, validators, and improvers (critics) without creating circular dependencies.

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

### Models

To register a custom model, create a factory function and decorate it with `@register_model`:

```python
from sifaka.registry import register_model
from sifaka.interfaces import Model

class CustomModel:
    def __init__(self, model_name, **options):
        self.model_name = model_name
        self.options = options
    
    def generate(self, prompt, **options):
        return f"Custom response for: {prompt}"
    
    def count_tokens(self, text):
        return len(text.split())

@register_model("custom")
def create_custom_model(model_name, **options):
    return CustomModel(model_name, **options)
```

### Validators

To register a custom validator, create a factory function and decorate it with `@register_validator`:

```python
from sifaka.registry import register_validator
from sifaka.interfaces import Validator
from sifaka.results import ValidationResult

class CustomValidator:
    def __init__(self, **options):
        self.options = options
    
    def validate(self, text):
        is_valid = len(text) > 10
        return ValidationResult(
            passed=is_valid,
            message="Text is valid" if is_valid else "Text is too short",
            details={"length": len(text)}
        )

@register_validator("custom")
def create_custom_validator(**options):
    return CustomValidator(**options)
```

### Improvers

To register a custom improver, create a factory function and decorate it with `@register_improver`:

```python
from sifaka.registry import register_improver
from sifaka.interfaces import Improver, Model
from sifaka.results import ImprovementResult

class CustomImprover:
    def __init__(self, model, **options):
        self.model = model
        self.options = options
    
    def improve(self, text):
        improved_text = text.upper()
        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=True,
            message="Text has been improved",
            details={"method": "uppercase"}
        )

@register_improver("custom")
def create_custom_improver(model, **options):
    return CustomImprover(model, **options)
```

## Using Components

### Direct Registry Access

You can access the registry directly to get factory functions:

```python
from sifaka.registry import get_model_factory, get_validator_factory, get_improver_factory

# Get factory functions
model_factory = get_model_factory("custom")
validator_factory = get_validator_factory("custom")
improver_factory = get_improver_factory("custom")

# Create components
model = model_factory("my-model", temperature=0.7)
validator = validator_factory(min_length=10)
improver = improver_factory(model, level="high")
```

### Factory Functions

Sifaka provides high-level factory functions that use the registry:

```python
from sifaka.factories import create_model, create_validator, create_improver

# Create components
model = create_model("custom", "my-model", temperature=0.7)
validator = create_validator("custom", min_length=10)
improver = create_improver("custom", model, level="high")
```

You can also create models from a string specification:

```python
from sifaka.factories import create_model_from_string

# Create a model from a string
model = create_model_from_string("custom:my-model", temperature=0.7)
```

### Using with Chain

You can use the components with the Chain class:

```python
from sifaka import Chain
from sifaka.factories import create_model, create_validator, create_improver

# Create components
model = create_model("custom", "my-model")
validator = create_validator("custom", min_length=10)
improver = create_improver("custom", model)

# Create a chain
chain = Chain()
chain.with_model(model)
chain.with_prompt("Hello, world!")
chain.validate_with(validator)
chain.improve_with(improver)

# Run the chain
result = chain.run()
```

## Lazy Loading

The registry uses lazy loading to prevent circular imports. Components are only imported when they are needed, not when the registry is initialized.

When you call `get_model_factory("custom")`, the registry will:

1. Check if the "model" component type has been initialized
2. If not, import all modules registered for the "model" component type
3. Return the factory function for the "custom" provider

## Complete Example

Here's a complete example of using the registry system:

```python
from sifaka.registry import register_model, register_validator, register_improver
from sifaka.interfaces import Model, Validator, Improver
from sifaka.results import ValidationResult, ImprovementResult
from sifaka.factories import create_model, create_validator, create_improver
from sifaka import Chain

# Define custom components
class EchoModel:
    def __init__(self, model_name, **options):
        self.model_name = model_name
        self.options = options
    
    def generate(self, prompt, **options):
        return f"Echo: {prompt}"
    
    def count_tokens(self, text):
        return len(text.split())

class LengthValidator:
    def __init__(self, min_length=None, max_length=None, **options):
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, text):
        length = len(text)
        passed = True
        message = "Text length is valid"
        
        if self.min_length and length < self.min_length:
            passed = False
            message = f"Text is too short ({length} chars, min {self.min_length})"
        
        if self.max_length and length > self.max_length:
            passed = False
            message = f"Text is too long ({length} chars, max {self.max_length})"
        
        return ValidationResult(
            passed=passed,
            message=message,
            details={"length": length}
        )

class ReverseImprover:
    def __init__(self, model, **options):
        self.model = model
        self.options = options
    
    def improve(self, text):
        improved_text = text[::-1]
        return improved_text, ImprovementResult(
            original_text=text,
            improved_text=improved_text,
            changes_made=True,
            message="Text has been reversed",
            details={"method": "reverse"}
        )

# Register components
@register_model("echo")
def create_echo_model(model_name, **options):
    return EchoModel(model_name, **options)

@register_validator("length")
def create_length_validator(min_length=None, max_length=None, **options):
    return LengthValidator(min_length, max_length, **options)

@register_improver("reverse")
def create_reverse_improver(model, **options):
    return ReverseImprover(model, **options)

# Create components using factory functions
model = create_model("echo", "echo-model")
validator = create_validator("length", min_length=10, max_length=100)
improver = create_improver("reverse", model)

# Create a chain
chain = Chain()
chain.with_model(model)
chain.with_prompt("Hello, world!")
chain.validate_with(validator)
chain.improve_with(improver)

# Run the chain
result = chain.run()
print(result.text)  # Output: "!dlrow ,olleH :ohcE"
```

## Best Practices

1. **Use Descriptive Names**: Choose clear, descriptive names for your components
2. **Follow the Interface**: Ensure your components implement the required methods
3. **Document Your Components**: Add docstrings to explain what your components do
4. **Handle Errors Gracefully**: Catch and handle errors in your component methods
5. **Use Type Hints**: Add type hints to make your code more readable and maintainable
6. **Test Your Components**: Write tests to ensure your components work correctly

## Conclusion

The registry system provides a flexible way to extend Sifaka with custom components. By using the registry, you can:

1. Create custom models, validators, and improvers
2. Register them with the system
3. Use them with the Chain API
4. Avoid circular dependencies

For more information, see the [Registry System Architecture](../architecture/registry.md) document.
