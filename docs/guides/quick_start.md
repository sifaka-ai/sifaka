# Quick Start Guide

This guide will help you get started with Sifaka quickly. You'll learn how to install Sifaka, create basic components, and run a simple validation chain.

## Installation

### Basic Installation

```bash
pip install sifaka
```

### Installation with Specific Features

```bash
# Install with OpenAI support
pip install "sifaka[openai]"

# Install with Anthropic support
pip install "sifaka[anthropic]"

# Install with all classifiers
pip install "sifaka[classifiers]"

# Install everything (except development tools)
pip install "sifaka[all]"
```

## Basic Usage

### Creating and Using Rules

Rules validate text against specific criteria. Here's how to create and use a simple length rule:

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a length rule
rule = create_length_rule(min_chars=10, max_chars=100)

# Validate text
text = "This is a test."
result = rule.validate(text)

print(f"Text: {text}")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

### Using Multiple Rules

You can use multiple rules together:

```python
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.formatting.style import create_style_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule

# Create rules
rules = [
    create_length_rule(min_chars=10, max_chars=100),
    create_style_rule(capitalization="sentence"),
    create_prohibited_content_rule(terms=["bad", "inappropriate"])
]

# Validate text with each rule
text = "This is a good example."
for rule in rules:
    result = rule.validate(text)
    print(f"Rule: {rule._name}")
    print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

### Using a Model Provider

Model providers connect to language model APIs:

```python
import os
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig

# Get API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Create a provider
provider = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    api_key=api_key,
    config=ModelConfig(temperature=0.7, max_tokens=500)
)

# Generate text
response = provider.generate("Write a short story about a robot.")
print(f"Generated text: {response}")
```

### Creating a Chain

Chains combine models, rules, and critics:

```python
import os
from sifaka.models.openai import OpenAIProvider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.chain import create_simple_chain

# Get API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Create a provider
provider = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    api_key=api_key
)

# Create a rule
rule = create_length_rule(min_chars=10, max_chars=100)

# Create a chain
chain = create_simple_chain(
    model=provider,
    rules=[rule],
    max_attempts=3
)

# Run the chain
result = chain.run("Write a short story about a robot.")

print(f"Output: {result.output}")
print(f"Validation passed: {result.all_passed}")
```

### Using a Critic

Critics help improve text that fails validation:

```python
import os
from sifaka.models.openai import OpenAIProvider
from sifaka.rules.formatting.length import create_length_rule
from sifaka.critics import create_prompt_critic
from sifaka.chain import create_simple_chain

# Get API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Create a provider
provider = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    api_key=api_key
)

# Create a rule
rule = create_length_rule(min_chars=10, max_chars=100)

# Create a critic
critic = create_prompt_critic(
    model=provider,
    system_prompt="You are an editor who helps adjust text length."
)

# Create a chain
chain = create_simple_chain(
    model=provider,
    rules=[rule],
    critic=critic,
    max_attempts=3
)

# Run the chain
result = chain.run("Write a story about a robot.")

print(f"Output: {result.output}")
print(f"Validation passed: {result.all_passed}")
print(f"Attempts: {result.attempts}")
```

## Next Steps

Now that you've seen the basics of Sifaka, you might want to explore:

1. [Rules and Validators](../rules_and_validators.md) - Learn more about validation
2. [Chain Architecture](../chain_architecture.md) - Understand how chains work
3. [Examples](../../examples/) - See more complete examples
4. [API Reference](../api/reference.md) - Explore the full API

For more detailed guides, see the [Documentation](../README.md).
