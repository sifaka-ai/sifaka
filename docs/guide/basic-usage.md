# Basic Usage

This guide covers the fundamental concepts and usage patterns of Sifaka.

## Core Concepts

### The Improvement Loop

Sifaka works through an iterative improvement loop:

1. **Generate**: Create or modify text
2. **Critique**: Analyze the text using various critics
3. **Improve**: Apply suggestions from critiques
4. **Validate**: Check if the text meets requirements
5. **Repeat**: Continue until satisfied or max iterations reached

### Key Components

- **Critics**: Analyze text and provide feedback
- **Validators**: Check if text meets specific requirements
- **Config**: Controls behavior and parameters
- **Result**: Contains the final text and full audit trail

## Simple Examples

For basic usage examples, see:
- **[Quickstart Guide](../quickstart.md)** - Getting started examples
- **[API Reference](../API.md)** - Complete API documentation with examples

### Using Different Critics

```python
# Default critic (reflexion is used when none specified)
result = await improve("Explain quantum computing")

# Explicitly specify single critic
result = await improve(
    "Explain quantum computing",
    critics=["reflexion"]  # Same as default
)

# Multiple critics
result = await improve(
    "Explain quantum computing",
    critics=["reflexion", "constitutional", "self_rag"]
)
```

### Adding Validators

```python
from sifaka.validators import LengthValidator, ContentValidator

result = await improve(
    "Write a product description",
    validators=[
        LengthValidator(min_length=100, max_length=200),
        ContentValidator(required_terms=["features", "benefits"])
    ]
)
```

### Controlling Iterations

```python
# More iterations for higher quality
result = await improve(
    text="Draft email to client",
    max_iterations=5  # Default is 3
)

# Force improvements even if validation passes
result = await improve(
    text="Good text that passes validation",
    force_improvements=True
)
```

## Understanding Results

The `SifakaResult` object contains:

```python
# Access the final improved text
print(result.final_text)

# See all critiques
for critique in result.critiques:
    print(f"{critique.critic}: {critique.feedback}")

# Check validation results
for validation in result.validations:
    print(f"{validation.validator}: {'✓' if validation.passed else '✗'}")

# View generation history
for generation in result.generations:
    print(f"Generation: {generation.text[:100]}...")

# Get improvement summary
print(f"Iterations: {result.iteration}")
print(f"Improved: {result.final_text != result.original_text}")
print(f"Processing time: {result.processing_time:.2f}s")
```

## Configuration

### Using Config Object

```python
from sifaka import Config

config = Config(
    model="gpt-4",
    temperature=0.7,
    max_iterations=5,
    force_improvements=True
)

result = await improve("Your text", config=config)
```

### Common Configuration Options

```python
config = Config(
    # Model settings
    model="gpt-4",              # LLM model to use
    temperature=0.7,            # Creativity (0.0-2.0)
    max_tokens=1000,            # Max response length

    # Critic settings
    critic_temperature=0.3,     # Lower = more consistent
    critic_context_window=3,    # Previous critiques to consider

    # Behavior settings
    max_iterations=3,           # Max improvement cycles
    force_improvements=False,   # Improve even if valid
    timeout_seconds=300,        # Overall timeout
)
```

## Error Handling

```python
from sifaka.core.exceptions import (
    ValidationError,
    CriticError,
    ModelProviderError
)

try:
    result = await improve(text)
except ValidationError as e:
    print(f"Validation failed: {e}")
except CriticError as e:
    print(f"Critic error: {e}")
except ModelProviderError as e:
    print(f"LLM error: {e}")
    print(f"Suggestion: {e.suggestion}")
```

## Best Practices

1. **Start Simple**: Use default settings first
2. **Choose Critics Wisely**: Different critics for different tasks
3. **Set Clear Validators**: Define what "good" means
4. **Monitor Costs**: More iterations = more API calls
5. **Handle Errors**: Always wrap in try/except for deployment

## Next Steps

- Learn about [different critics](../critics/overview.md)
- Explore [advanced usage](advanced-usage.md)
- Understand [configuration options](configuration.md)
- Create [custom validators](validators.md)
