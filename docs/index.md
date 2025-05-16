# Sifaka Documentation

Welcome to the Sifaka documentation! Sifaka is a streamlined framework for building reliable LLM applications with validation and improvement capabilities.

## Overview

Sifaka provides a clean, intuitive API for working with large language models (LLMs). It focuses on:

- **Simplicity**: Easy to use, with a fluent API that's intuitive and straightforward
- **Reliability**: Built-in validation and improvement mechanisms
- **Flexibility**: Support for multiple model providers and customization options
- **Testability**: Designed for easy testing and integration

## Getting Started

To get started with Sifaka, install the package:

```bash
pip install sifaka
```

Then, create a simple chain:

```python
import sifaka

result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .run())

print(result.text)
```

## Documentation Sections

- [API Reference](api/index.md): Detailed documentation of all Sifaka components
- [Architecture](architecture/index.md): Overview of Sifaka's design and architecture
- [Tutorials](tutorials/index.md): Step-by-step guides for common use cases

## Core Components

Sifaka is built around a few core components:

- **Chain**: The main orchestrator that coordinates the generation, validation, and improvement of text
- **Models**: Interfaces to various LLM providers (OpenAI, Anthropic, etc.)
- **Validators**: Components that check if generated text meets specific criteria
- **Critics**: LLM-based components that validate and improve text

## Example

Here's a more complete example that demonstrates validation and improvement:

```python
import sifaka
from sifaka.validators import length, clarity, factual_accuracy

result = (sifaka.Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short explanation of quantum computing.")
    .validate_with(length(min_words=50, max_words=200))
    .validate_with(factual_accuracy())
    .improve_with(clarity())
    .run())

print(f"Result passed validation: {result.passed}")
print(result.text)

# Print validation results
for i, validation_result in enumerate(result.validation_results):
    print(f"Validation {i+1}: {validation_result.message}")

# Print improvement results
for i, improvement_result in enumerate(result.improvement_results):
    print(f"Improvement {i+1}: {improvement_result.message}")
```

## Contributing

Contributions to Sifaka are welcome! Please see our [contribution guidelines](CONTRIBUTING.md) for more information.

## License

Sifaka is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
