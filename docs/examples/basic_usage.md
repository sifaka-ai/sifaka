# Basic Usage Examples

This document provides basic usage examples for Sifaka's core functionality.

## Basic Validation

### Using Built-in Rules

```python
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.formatting.style import create_style_rule

# Create rules
length_rule = create_length_rule(min_chars=10, max_chars=100)
style_rule = create_style_rule(capitalization="sentence")

# Validate text
text = "this is a test"
length_result = length_rule.validate(text)
style_result = style_rule.validate(text)

print(f"Length validation: {'Passed' if length_result.passed else 'Failed'}")
print(f"Style validation: {'Passed' if style_result.passed else 'Failed'}")
```

### Using Multiple Rules Together

```python
from sifaka.chain import create_validation_chain
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.formatting.style import create_style_rule

# Create rules
rules = [
    create_length_rule(min_chars=10, max_chars=100),
    create_style_rule(capitalization="sentence")
]

# Create a validation chain
validation_chain = create_validation_chain(
    name="basic_validation",
    rules=rules
)

# Validate text
text = "this is a test"
result = validation_chain.process(text)

print(f"All validations passed: {result.all_passed}")
for rule_result in result.rule_results:
    print(f"Rule: {rule_result.rule_name}")
    print(f"Passed: {rule_result.passed}")
    print(f"Message: {rule_result.message}")
```

## Model Integration

### Using a Model Provider

```python
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.chain import create_validation_chain
from sifaka.rules.formatting.length import create_length_rule

# Create model provider
model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(temperature=0.7, max_tokens=500)
)

# Create validation chain
validation_chain = create_validation_chain(
    name="model_validation",
    rules=[create_length_rule(min_chars=10, max_chars=100)]
)

# Generate and validate text
prompt = "Write a short story about a cat."
response = model.generate(prompt)
result = validation_chain.process(response["text"])

print(f"Generated text: {response['text']}")
print(f"Validation passed: {result.all_passed}")
```

## Critic Integration

### Using a Critic

```python
from sifaka.critics import create_style_critic
from sifaka.chain import create_validation_chain
from sifaka.rules.formatting.length import create_length_rule

# Create critic
critic = create_style_critic()

# Create validation chain
validation_chain = create_validation_chain(
    name="critic_validation",
    rules=[create_length_rule(min_chars=10, max_chars=100)]
)

# Analyze and improve text
text = "this is a test"
critique = critic.critique(text)
result = validation_chain.process(text)

if not result.all_passed:
    improved_text = critic.improve(text, [r.message for r in result.rule_results])
    print(f"Original text: {text}")
    print(f"Improved text: {improved_text}")
```

## Advanced Usage

### Custom Rule Implementation

```python
from sifaka.rules.base import Rule, RuleResult, RuleConfig, BaseValidator
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class CustomConfig(RuleConfig):
    """Configuration for custom rule."""
    min_words: int = 5
    max_words: int = 20

class CustomValidator(BaseValidator[str]):
    """Custom validator implementation."""

    def __init__(self, config: CustomConfig):
        self.config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        words = text.split()
        if len(words) < self.config.min_words:
            return RuleResult(
                passed=False,
                message=f"Too few words: {len(words)} (min {self.config.min_words})"
            )
        if len(words) > self.config.max_words:
            return RuleResult(
                passed=False,
                message=f"Too many words: {len(words)} (max {self.config.max_words})"
            )
        return RuleResult(passed=True, message="Validation passed")

class CustomRule(Rule[str, RuleResult, CustomValidator, None]):
    """Custom rule implementation."""

    def _create_default_validator(self) -> CustomValidator:
        config = CustomConfig(**self._rule_params)
        return CustomValidator(config)

# Create and use custom rule
rule = CustomRule(
    name="word_count_rule",
    description="Validates word count",
    config=CustomConfig(min_words=3, max_words=10)
)

result = rule.validate("This is a test")
print(f"Validation passed: {result.passed}")
print(f"Message: {result.message}")
```

### Combining Multiple Rules in a Chain

```python
from sifaka.chain import create_validation_chain
from sifaka.rules.formatting.length import create_length_rule
from sifaka.rules.formatting.style import create_style_rule
from sifaka.rules.content.prohibited import create_prohibited_content_rule

# Create validation chain with multiple rules
validation_chain = create_validation_chain(
    name="comprehensive_validation",
    description="Chain with multiple validation rules",
    rules=[
        create_length_rule(min_chars=10, max_chars=100),
        create_style_rule(capitalization="sentence"),
        create_prohibited_content_rule(terms=["bad", "inappropriate"])
    ]
)

# Validate text
text = "this is a test"
result = validation_chain.process(text)

print(f"Text: {text}")
print(f"All validations passed: {result.all_passed}")
for rule_result in result.rule_results:
    print(f"\nRule: {rule_result.rule_name}")
    print(f"Passed: {rule_result.passed}")
    print(f"Message: {rule_result.message}")
```

### Integration with Model and Critic

```python
from sifaka.models.openai import OpenAIProvider
from sifaka.models.base import ModelConfig
from sifaka.critics import create_style_critic
from sifaka.chain import create_improvement_chain
from sifaka.rules.formatting.length import create_length_rule

# Create components
model = OpenAIProvider(
    model_name="gpt-3.5-turbo",
    config=ModelConfig(temperature=0.7, max_tokens=500)
)
critic = create_style_critic()

# Create improvement chain
improvement_chain = create_improvement_chain(
    name="integrated_improvement",
    rules=[create_length_rule(min_chars=10, max_chars=100)],
    critic=critic,
    max_attempts=3
)

# Generate and improve text
prompt = "Write a short story about a cat."
response = model.generate(prompt)
text = response["text"]

# Validate and improve if needed
result = improvement_chain.process(text)
if result.improved_text:
    print(f"Original text: {text}")
    print(f"Improved text: {result.improved_text}")
else:
    print(f"Generated text: {text}")
    print(f"All rules passed: {result.all_passed}")
```