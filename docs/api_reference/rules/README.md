# Rules API Reference

Rules are the validation components in Sifaka that perform binary pass/fail checks on text. They are used to enforce constraints on generated content and provide feedback when those constraints are not met.

## Core Classes and Protocols

### Rule

`Rule` is the base class for all validation rules in Sifaka.

```python
from sifaka.rules.base import Rule, RuleConfig, RuleResult

class MyRule(Rule):
    """Custom rule implementation."""
    
    def __init__(self, name: str, description: str, config: RuleConfig):
        super().__init__(name=name, description=description, config=config)
    
    def validate(self, text: str) -> RuleResult:
        """Validate the text."""
        if len(text) > 100:
            return RuleResult(
                passed=False,
                rule_name=self.name,
                message="Text is too long",
                metadata={"length": len(text)}
            )
        return RuleResult(
            passed=True,
            rule_name=self.name,
            message="Text length is acceptable",
            metadata={"length": len(text)}
        )
```

### RuleValidator

`RuleValidator` is the protocol for validation logic that can be used by rules.

```python
from sifaka.rules.base import RuleValidator, RuleResult

class MyValidator(RuleValidator[str]):
    """Custom validator implementation."""
    
    def validate(self, text: str) -> RuleResult:
        """Validate the text."""
        if len(text) > 100:
            return RuleResult(
                passed=False,
                rule_name="length_validator",
                message="Text is too long",
                metadata={"length": len(text)}
            )
        return RuleResult(
            passed=True,
            rule_name="length_validator",
            message="Text length is acceptable",
            metadata={"length": len(text)}
        )
```

### RuleProtocol

`RuleProtocol` defines the interface for all rule-like objects.

```python
from sifaka.rules.base import RuleProtocol, RuleResult

# Any class implementing RuleProtocol can be used where a Rule is expected
def process_rule(rule: RuleProtocol):
    """Process a rule."""
    result = rule.validate("This is a test")
    print(f"Rule {rule.name} {'passed' if result.passed else 'failed'}")
```

## Configuration

### RuleConfig

`RuleConfig` is the configuration class for rules.

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
print(f"Cost: {config.cost}")
print(f"Min length: {config.params['min_length']}")

# Create a new configuration with updated options
updated_config = config.with_options(
    priority=RulePriority.MEDIUM,
    params={"min_length": 20}
)
```

## Results

### RuleResult

`RuleResult` represents the result of a rule validation.

```python
from sifaka.rules.base import RuleResult

# Create a rule result
result = RuleResult(
    passed=True,
    rule_name="length_rule",
    message="Text length is acceptable",
    metadata={"length": 50}
)

# Access result values
print(f"Passed: {result.passed}")
print(f"Rule name: {result.rule_name}")
print(f"Message: {result.message}")
print(f"Length: {result.metadata['length']}")
```

## Factory Functions

### create_rule

`create_rule` is a factory function for creating rules.

```python
from sifaka.rules.base import create_rule, RuleConfig, RuleValidator

# Create a validator
class MyValidator(RuleValidator[str]):
    def validate(self, text: str) -> RuleResult:
        return RuleResult(passed=True, rule_name="test", message="OK")

# Create a rule using the factory function
rule = create_rule(
    name="my_rule",
    description="A custom rule",
    validator=MyValidator(),
    config=RuleConfig(priority="HIGH")
)
```

## Rule Types

Sifaka provides several types of rules:

### Content Rules

Content rules validate the semantic content of text.

```python
from sifaka.rules.content.toxicity import create_toxicity_rule

# Create a toxicity rule
rule = create_toxicity_rule(
    threshold=0.7,
    name="toxicity_rule",
    description="Ensures text is not toxic"
)
```

### Formatting Rules

Formatting rules validate the structure and format of text.

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a length rule
rule = create_length_rule(
    min_chars=10,
    max_chars=100,
    name="length_rule",
    description="Ensures text is the right length"
)
```

### Accuracy Rules

Accuracy rules validate the factual accuracy of text.

```python
from sifaka.rules.accuracy import create_fact_checking_rule

# Create a fact checking rule
rule = create_fact_checking_rule(
    name="fact_rule",
    description="Ensures text is factually accurate"
)
```

## Rule Composition

Rules can be composed using logical operators.

```python
from sifaka.rules.composition import create_and_rule, create_or_rule, create_not_rule

# Create component rules
length_rule = create_length_rule(min_chars=10, max_chars=100)
toxicity_rule = create_toxicity_rule(threshold=0.7)

# Create composite rules
and_rule = create_and_rule(
    rules=[length_rule, toxicity_rule],
    name="length_and_toxicity",
    description="Ensures text is the right length and not toxic"
)

or_rule = create_or_rule(
    rules=[length_rule, toxicity_rule],
    name="length_or_toxicity",
    description="Ensures text is either the right length or not toxic"
)

not_rule = create_not_rule(
    rule=toxicity_rule,
    name="not_toxicity",
    description="Ensures text is toxic"
)
```

## Usage Examples

### Basic Rule Usage

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a rule
rule = create_length_rule(min_chars=10, max_chars=100)

# Validate text
result = rule.validate("This is a test")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

### Rule with Custom Validator

```python
from sifaka.rules.base import create_rule, RuleConfig, RuleValidator, RuleResult

# Create a custom validator
class CustomValidator(RuleValidator[str]):
    def validate(self, text: str) -> RuleResult:
        words = text.split()
        if len(words) < 5:
            return RuleResult(
                passed=False,
                rule_name="word_count",
                message="Text has too few words",
                metadata={"word_count": len(words)}
            )
        return RuleResult(
            passed=True,
            rule_name="word_count",
            message="Text has enough words",
            metadata={"word_count": len(words)}
        )

# Create a rule with the custom validator
rule = create_rule(
    name="word_count_rule",
    description="Ensures text has enough words",
    validator=CustomValidator(),
    config=RuleConfig(priority="MEDIUM")
)

# Validate text
result = rule.validate("This is a test")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```

### Function-Based Rule

```python
from sifaka.rules.base import create_function_rule

# Create a rule using a function
def validate_word_count(text: str) -> bool:
    return len(text.split()) >= 5

rule = create_function_rule(
    func=validate_word_count,
    name="word_count_rule",
    description="Ensures text has enough words",
    failure_message="Text has too few words"
)

# Validate text
result = rule.validate("This is a test")
print(f"Validation {'passed' if result.passed else 'failed'}: {result.message}")
```
