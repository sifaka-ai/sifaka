# Standardized Configuration in Sifaka

This document describes the standardized configuration approach used throughout the Sifaka framework.

## Configuration Principles

Sifaka follows these principles for configuration:

1. **Consistency**: All components use a similar configuration pattern
2. **Immutability**: Configuration objects are immutable to prevent accidental changes
3. **Extensibility**: Base configuration classes can be extended for specialized needs
4. **Standardization**: All component-specific parameters are stored in a `params` dictionary
5. **Type Safety**: All configurations use Pydantic models for type safety and validation

## Configuration Classes

### ChainConfig

The `ChainConfig` class is used for configuring chains:

```python
from sifaka.chain.config import ChainConfig

# Create a chain configuration
config = ChainConfig(
    max_attempts=3,
    trace_enabled=True,
    params={
        "system_prompt": "You are a helpful assistant.",
        "use_critic": True,
    }
)

# Access configuration values
print(f"Max attempts: {config.max_attempts}")
print(f"System prompt: {config.params.get('system_prompt')}")

# Create a new configuration with updated options
updated_config = config.with_options(max_attempts=5)

# Create a new configuration with updated params
parameterized_config = config.with_params(system_prompt="You are an expert coder.")
```

### RetryConfig

The `RetryConfig` class is used for configuring retry strategies:

```python
from sifaka.chain.config import RetryConfig, BackoffRetryConfig

# Create a basic retry configuration
config = RetryConfig(
    max_attempts=3,
    params={
        "use_backoff": True,
    }
)

# Create a backoff retry configuration
backoff_config = BackoffRetryConfig(
    max_attempts=5,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0,
    params={
        "jitter": True,
    }
)

# Access configuration values
print(f"Max attempts: {config.max_attempts}")
print(f"Use backoff: {config.params.get('use_backoff')}")

# Create a new configuration with updated options
updated_config = config.with_options(max_attempts=5)

# Create a new configuration with updated params
parameterized_config = config.with_params(use_backoff=False)
```

### ValidationConfig

The `ValidationConfig` class is used for configuring validation managers:

```python
from sifaka.chain.config import ValidationConfig

# Create a validation configuration
config = ValidationConfig(
    prioritize_by_cost=True,
    params={
        "fail_fast": True,
    }
)

# Access configuration values
print(f"Prioritize by cost: {config.prioritize_by_cost}")
print(f"Fail fast: {config.params.get('fail_fast')}")

# Create a new configuration with updated options
updated_config = config.with_options(prioritize_by_cost=False)

# Create a new configuration with updated params
parameterized_config = config.with_params(fail_fast=False)
```

### RuleConfig

The `RuleConfig` class is used for configuring rules:

```python
from sifaka.rules.base import RuleConfig, RulePriority

# Create a rule configuration
config = RuleConfig(
    priority=RulePriority.HIGH,
    cost=5,
    cache_size=100,
    params={
        "min_length": 10,
        "max_length": 100,
    }
)

# Access configuration values
print(f"Priority: {config.priority}")
print(f"Min length: {config.params.get('min_length')}")

# Create a new configuration with updated options
updated_config = config.with_options(priority=RulePriority.CRITICAL)

# Create a new configuration with updated params
parameterized_config = config.with_params(min_length=20, max_length=200)
```

### CriticConfig

The `CriticConfig` class is used for configuring critics:

```python
from sifaka.critics.models import CriticConfig, PromptCriticConfig

# Create a basic critic configuration
config = CriticConfig(
    name="my_critic",
    description="A custom critic",
    min_confidence=0.7,
    max_attempts=3,
    params={
        "system_prompt": "You are an expert editor.",
    }
)

# Create a specialized critic configuration
prompt_config = PromptCriticConfig(
    name="prompt_critic",
    description="A prompt-based critic",
    min_confidence=0.7,
    max_attempts=3,
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)
```

## Configuration Methods

All configuration classes in Sifaka provide these standard methods:

### with_options

```python
# Update top-level configuration options
updated_config = config.with_options(
    max_attempts=5,
    trace_enabled=True
)
```

### with_params

```python
# Update or add component-specific parameters
parameterized_config = config.with_params(
    system_prompt="You are an expert coder.",
    use_critic=True
)
```

## Factory Functions

Factory functions in Sifaka use the standardized configuration approach:

### Chain Factory Functions

```python
from sifaka.chain import ChainOrchestrator
from sifaka.chain.config import ChainConfig

# Create a chain with direct parameters
chain = ChainOrchestrator(
    model=model,
    rules=rules,
    max_attempts=3,
    trace_enabled=True
)

# Create a chain with a ChainConfig
config = ChainConfig(
    max_attempts=3,
    trace_enabled=True,
    params={
        "system_prompt": "You are a helpful assistant.",
    }
)
chain = ChainOrchestrator(model=model, rules=rules, config=config)
```

### Rule Factory Functions

```python
from sifaka.rules.base import RuleConfig
from sifaka.rules.formatting.length import create_length_rule

# Create a rule with direct parameters
rule = create_length_rule(
    min_chars=10,
    max_chars=100,
    min_words=2,
    max_words=20
)

# Create a rule with a RuleConfig
rule_config = RuleConfig(
    priority="HIGH",
    cost=5,
    params={
        "min_chars": 10,
        "max_chars": 100,
    }
)
rule = create_length_rule(config=rule_config)
```

### Critic Factory Functions

```python
from sifaka.critics.models import PromptCriticConfig
from sifaka.critics.factories import create_prompt_critic

# Create a critic with direct parameters
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)

# Create a critic with a PromptCriticConfig
config = PromptCriticConfig(
    name="prompt_critic",
    description="A prompt-based critic",
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)
critic = create_prompt_critic(config=config)
```

## Best Practices

1. **Use factory functions** instead of direct class instantiation
2. **Use RuleConfig with params** for all rule configuration
3. **Use ClassifierConfig with params** for all classifier configuration
4. **Use CriticConfig with params** for all critic configuration
5. **Use standardization utilities** to ensure consistent configuration handling
6. **Extend base configuration classes** for specialized needs rather than creating new ones
7. **Keep configuration immutable** by using with_options() and with_params() methods
