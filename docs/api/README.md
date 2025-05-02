# API Reference

This document provides detailed API reference information for Sifaka.

## Core Components

### Domain

The `Domain` class manages a specific area of operation with its own rules, models, and critics.

```python
from sifaka.domain import Domain

# Create a domain
domain = Domain(config={
    "name": "code",
    "rules": {...},
    "models": {...},
    "critics": {...}
})

# Use the domain
result = domain.process(input_data)
```

#### Methods

- `process(input_data: Any) -> Any`
  - Process input data through the domain
  - Returns processed output

- `validate(config: Dict) -> bool`
  - Validate domain configuration
  - Returns True if valid

- `add_rule(name: str, rule: Rule) -> None`
  - Add a new rule to the domain
  - Raises DomainError if rule exists

- `remove_rule(name: str) -> None`
  - Remove a rule from the domain
  - Raises DomainError if rule not found

### Rules

Rules validate and transform data within a domain.

```python
from sifaka.rules import Rule

# Create a rule
rule = Rule(
    name="length",
    validator=lambda x: len(x) > 0,
    transformer=lambda x: x.strip()
)

# Use the rule
result = rule.apply(" test ")
```

#### Methods

- `apply(input_data: Any) -> Any`
  - Apply rule to input data
  - Returns transformed data

- `validate(input_data: Any) -> bool`
  - Validate input data
  - Returns True if valid

### Models

Models provide AI capabilities for the domain.

```python
from sifaka.models import ModelProvider

# Create a model
model = ModelProvider(
    name="openai",
    config={
        "model": "gpt-4",
        "temperature": 0.7
    }
)

# Use the model
result = model.generate("Hello")
```

#### Methods

- `generate(prompt: str) -> Dict`
  - Generate output from prompt
  - Returns generation result

- `validate(config: Dict) -> bool`
  - Validate model configuration
  - Returns True if valid

### Critics

Critics evaluate and improve domain output.

```python
from sifaka.critics import Critic

# Create a critic
critic = Critic(
    name="quality",
    evaluator=lambda x: score_quality(x),
    improver=lambda x: improve_quality(x)
)

# Use the critic
result = critic.evaluate(output_data)
```

#### Methods

- `evaluate(output_data: Any) -> Dict`
  - Evaluate output data
  - Returns evaluation result

- `improve(output_data: Any) -> Any`
  - Improve output data
  - Returns improved output

## Configuration

### Domain Configuration

```python
domain_config = {
    "name": "code",
    "description": "Code generation domain",
    "rules": {
        "syntax": {
            "enabled": True,
            "config": {
                "language": "python"
            }
        }
    },
    "models": {
        "default": {
            "provider": "openai",
            "config": {
                "model": "gpt-4"
            }
        }
    },
    "critics": {
        "quality": {
            "enabled": True,
            "config": {
                "threshold": 0.8
            }
        }
    }
}
```

### Rule Configuration

```python
rule_config = {
    "name": "length",
    "enabled": True,
    "config": {
        "min_length": 1,
        "max_length": 100
    }
}
```

### Model Configuration

```python
model_config = {
    "provider": "openai",
    "config": {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 100
    }
}
```

### Critic Configuration

```python
critic_config = {
    "name": "quality",
    "enabled": True,
    "config": {
        "threshold": 0.8,
        "metrics": ["coherence", "relevance"]
    }
}
```

## Error Handling

### DomainError

Raised when domain operations fail.

```python
try:
    domain.process(input_data)
except DomainError as e:
    print(f"Domain error: {e}")
```

### RuleError

Raised when rule operations fail.

```python
try:
    rule.apply(input_data)
except RuleError as e:
    print(f"Rule error: {e}")
```

### ModelError

Raised when model operations fail.

```python
try:
    model.generate(prompt)
except ModelError as e:
    print(f"Model error: {e}")
```

### CriticError

Raised when critic operations fail.

```python
try:
    critic.evaluate(output_data)
except CriticError as e:
    print(f"Critic error: {e}")
```

## Examples

### Basic Usage

```python
from sifaka.domain import Domain

# Create domain
domain = Domain({
    "name": "text",
    "rules": {
        "length": {"enabled": True}
    }
})

# Process input
result = domain.process("Hello, world!")
```

### Advanced Usage

```python
from sifaka.domain import Domain
from sifaka.rules import Rule
from sifaka.models import ModelProvider
from sifaka.critics import Critic

# Create components
rule = Rule(name="length", validator=lambda x: len(x) > 0)
model = ModelProvider(name="openai", config={"model": "gpt-4"})
critic = Critic(name="quality", evaluator=lambda x: score_quality(x))

# Create domain
domain = Domain({
    "name": "code",
    "rules": {"length": rule},
    "models": {"default": model},
    "critics": {"quality": critic}
})

# Process input
result = domain.process("def hello():")
```

## Best Practices

1. **Configuration**
   - Use environment variables for sensitive data
   - Validate configurations before use
   - Document all configuration options

2. **Error Handling**
   - Catch specific exceptions
   - Provide meaningful error messages
   - Log errors appropriately

3. **Performance**
   - Cache expensive operations
   - Use async operations when possible
   - Monitor resource usage

4. **Security**
   - Validate all input
   - Sanitize all output
   - Use secure defaults

## Testing

See [Testing Guide](../testing/README.md) for information on testing the API.