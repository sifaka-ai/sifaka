# Sifaka Usage Guide

This guide provides comprehensive instructions for using the Sifaka framework to build reliable LLM applications.

## Table of Contents

1. [Installation](#installation)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Working with Models](#working-with-models)
5. [Validation](#validation)
6. [Critics and Improvement](#critics-and-improvement)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Installation

Install Sifaka using pip:

```bash
pip install sifaka
```

For full functionality, you'll also want to install the dependencies for the model providers you plan to use:

```bash
# For OpenAI models
pip install openai tiktoken

# For Anthropic models
pip install anthropic

# For Google Gemini models
pip install google-generativeai
```

## Core Concepts

Sifaka is built around a few core concepts:

- **Chain**: The main orchestrator that coordinates the generation, validation, and improvement of text
- **Models**: Interfaces to various LLM providers (OpenAI, Anthropic, etc.)
- **Validators**: Components that check if generated text meets specific criteria
- **Critics**: LLM-based components that validate and improve text
- **Results**: Types that represent the results of various operations

## Basic Usage

Here's a simple example of using Sifaka to generate text:

```python
from sifaka import Chain

# Create a simple chain
result = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .run())

print(result.text)
```

This example:
1. Creates a new Chain
2. Configures it to use the OpenAI GPT-4 model
3. Sets the prompt to "Write a short story about a robot."
4. Runs the chain
5. Prints the generated text

## Working with Models

Sifaka supports multiple model providers through a unified interface.

### Specifying Models

You can specify a model in two ways:

1. Using a string in the format "provider:model_name":

```python
chain = Chain().with_model("openai:gpt-4")
```

2. Creating a model instance directly:

```python
from sifaka.models import create_model

model = create_model("openai", "gpt-4", api_key="your-api-key")
chain = Chain().with_model(model)
```

### Supported Providers

Sifaka supports the following model providers:

- **OpenAI**: GPT-3.5, GPT-4, etc.
- **Anthropic**: Claude models
- **Google**: Gemini models

### Model Options

You can configure model options using the `with_options` method:

```python
chain = (Chain()
    .with_model("openai:gpt-4")
    .with_options(
        temperature=0.7,
        max_tokens=500,
        system_message="You are a creative writing assistant."
    ))
```

Common options include:
- `temperature`: Controls randomness (0.0 to 1.0)
- `max_tokens`: Maximum tokens to generate
- `system_message`: A system message for chat models

## Validation

Validators check if generated text meets specific criteria.

### Built-in Validators

Sifaka includes several built-in validators:

```python
from sifaka.validators import length, content, factual_accuracy

# Length validator
chain.validate_with(length(min_words=50, max_words=200))

# Content validator
chain.validate_with(content(required_terms=["robot", "AI"]))

# Factual accuracy validator
chain.validate_with(factual_accuracy())
```

### Validation Results

When you run a chain with validators, the result includes validation information:

```python
result = chain.run()

print(f"Validation passed: {result.passed}")

for validation_result in result.validation_results:
    print(f"Passed: {validation_result.passed}")
    print(f"Message: {validation_result.message}")
```

## Critics and Improvement

Critics are LLM-based components that can both validate and improve text.

### Built-in Critics

Sifaka includes several built-in critics:

```python
from sifaka.critics import create_lac_critic, create_reflexion_critic, create_constitutional_critic

# LAC (Language Agent Correction) critic
chain.improve_with(create_lac_critic(model))

# Reflexion critic
chain.improve_with(create_reflexion_critic(model))

# Constitutional critic
chain.improve_with(create_constitutional_critic(model, constitution=[
    "Your output should be factually accurate and not misleading.",
    "Your output should be helpful, harmless, and honest.",
]))
```

### Improvement Results

When you run a chain with improvers, the result includes improvement information:

```python
result = chain.run()

for improvement_result in result.improvement_results:
    print(f"Changes made: {improvement_result.changes_made}")
    print(f"Message: {improvement_result.message}")
```

## Advanced Features

### Combining Validators and Critics

You can combine multiple validators and critics in a single chain:

```python
from sifaka import Chain
from sifaka.validators import length, factual_accuracy
from sifaka.critics import create_lac_critic

result = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short explanation of quantum computing.")
    .validate_with(length(min_words=50, max_words=200))
    .validate_with(factual_accuracy())
    .improve_with(create_lac_critic(model))
    .run())
```

### Using Critics Directly

You can also use critics directly without a chain:

```python
from sifaka.critics import create_lac_critic
from sifaka.models import create_model

model = create_model("openai", "gpt-4")
critic = create_lac_critic(model)

text = "This is a text that could be improved."
critique = critic._critique(text)
improved_text = critic._improve(text, critique)

print(f"Original: {text}")
print(f"Improved: {improved_text}")
```

### Error Handling

It's important to handle errors when working with LLMs:

```python
from sifaka import Chain
from sifaka.errors import ModelError, ValidationError, ChainError

try:
    result = chain.run()
except ModelError as e:
    print(f"Model error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except ChainError as e:
    print(f"Chain error: {e}")
```

## Best Practices

### Prompt Engineering

- Be specific and clear in your prompts
- Provide examples when appropriate
- Use system messages to set the tone and context

### Validation Strategy

- Start with basic validators (length, content)
- Add more specific validators as needed
- Use factual accuracy validators for factual content

### Improvement Strategy

- Choose critics based on your specific needs
- LAC critics are good for general improvement
- Constitutional critics are good for ensuring outputs follow specific guidelines
- Reflexion critics are good for iterative improvement

### Performance Optimization

- Use caching for repeated operations
- Set appropriate token limits
- Use streaming for long-running operations

## Troubleshooting

### Common Issues

- **API Key Issues**: Make sure you have set the appropriate API keys for your model providers
- **Rate Limiting**: If you're hitting rate limits, consider implementing retry logic or using a different provider
- **Token Limits**: If you're getting truncated responses, increase the max_tokens parameter

### Debugging

- Enable logging to see what's happening behind the scenes
- Use the `_critique` and `_improve` methods directly to debug critic behavior
- Check the validation and improvement results for detailed information

### Getting Help

- Check the [API Reference](api/index.md) for detailed documentation
- Look at the [Examples](examples/) for inspiration
- Join the community forum for support
