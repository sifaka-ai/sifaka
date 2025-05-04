# Standardized Configuration in Sifaka

This document describes the standardized configuration approach used throughout the Sifaka framework.

## Configuration Principles

Sifaka follows these principles for configuration:

1. **Consistency**: All components use a similar configuration pattern
2. **Immutability**: Configuration objects are immutable to prevent accidental changes
3. **Extensibility**: Base configuration classes can be extended for specialized needs
4. **Standardization**: All component-specific parameters are stored in a `params` dictionary

## Configuration Classes

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

### ClassifierConfig

The `ClassifierConfig` class is used for configuring classifiers:

```python
from sifaka.classifiers.base import ClassifierConfig

# Create a classifier configuration
config = ClassifierConfig(
    labels=["positive", "negative", "neutral"],
    min_confidence=0.7,
    cache_size=100,
    cost=5,
    params={
        "model_name": "sentiment-large",
        "threshold": 0.7,
    }
)

# Access configuration values
print(f"Labels: {config.labels}")
print(f"Model name: {config.params.get('model_name')}")

# Create a new configuration with updated options
updated_config = config.with_options(min_confidence=0.8)
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

## Standardization Utilities

Sifaka provides utility functions to standardize configuration handling:

### standardize_rule_config

```python
from sifaka.utils.config import standardize_rule_config

# Create a rule config with direct parameters
config = standardize_rule_config(
    priority="HIGH",
    cost=5,
    params={
        "min_length": 10,
        "max_length": 100,
    }
)

# Update an existing config
updated_config = standardize_rule_config(
    config=config,
    params={"min_length": 20},
    cost=10
)

# Create from dictionary
config_dict = {
    "priority": "LOW",
    "params": {
        "min_length": 5,
        "max_length": 50,
    }
}
dict_config = standardize_rule_config(config=config_dict)
```

### standardize_classifier_config

```python
from sifaka.utils.config import standardize_classifier_config

# Create a classifier config with direct parameters
config = standardize_classifier_config(
    labels=["positive", "negative", "neutral"],
    min_confidence=0.7,
    params={
        "model_name": "sentiment-large",
        "threshold": 0.7,
    }
)

# Update an existing config
updated_config = standardize_classifier_config(
    config=config,
    params={"threshold": 0.8},
    min_confidence=0.9
)
```

### standardize_critic_config

```python
from sifaka.utils.config import standardize_critic_config
from sifaka.critics.models import PromptCriticConfig

# Create a critic config with direct parameters
config = standardize_critic_config(
    name="basic_critic",
    description="A basic critic",
    min_confidence=0.7,
    max_attempts=3,
    params={
        "system_prompt": "You are an expert editor.",
    }
)

# Create a specialized critic config
prompt_config = standardize_critic_config(
    config_class=PromptCriticConfig,
    name="prompt_critic",
    description="A prompt-based critic",
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)
```

## Factory Functions

Factory functions in Sifaka use the standardized configuration approach:

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

### Classifier Factory Functions

```python
from sifaka.classifiers.base import ClassifierConfig
from sifaka.classifiers.toxicity import create_toxicity_classifier

# Create a classifier with direct parameters
classifier = create_toxicity_classifier(
    general_threshold=0.5,
    severe_toxic_threshold=0.7,
    threat_threshold=0.7
)

# Create a classifier with a ClassifierConfig
config = ClassifierConfig(
    labels=["toxic", "non-toxic"],
    min_confidence=0.7,
    params={
        "general_threshold": 0.5,
        "severe_toxic_threshold": 0.7,
        "threat_threshold": 0.7,
    }
)
classifier = create_toxicity_classifier(config=config)
```

### Critic Factory Functions

```python
from sifaka.critics.models import PromptCriticConfig
from sifaka.critics.factories import create_prompt_critic
from sifaka.models.mock import MockProvider

# Create a critic with direct parameters
model = MockProvider()
critic = create_prompt_critic(
    llm_provider=model,
    name="my_critic",
    description="A custom critic",
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)

# Create a critic with a CriticConfig
config = PromptCriticConfig(
    name="prompt_critic",
    description="A prompt-based critic",
    system_prompt="You are an expert editor.",
    temperature=0.7,
    max_tokens=1000
)
critic = create_prompt_critic(llm_provider=model, config=config)
```

## Best Practices

1. **Use factory functions** instead of direct class instantiation
2. **Use RuleConfig with params** for all rule configuration
3. **Use ClassifierConfig with params** for all classifier configuration
4. **Use CriticConfig with params** for all critic configuration
5. **Use standardization utilities** to ensure consistent configuration handling
6. **Extend base configuration classes** for specialized needs rather than creating new ones
7. **Keep configuration immutable** by using with_options() and with_params() methods