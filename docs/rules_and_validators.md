# Rules and Validators in Sifaka

This document explains the relationship between Rules and Validators in the Sifaka framework and provides guidance on consistent instantiation patterns.

## Relationship Between Rules and Validators

### Core Concepts

- **Validators**: Perform the actual validation logic on text inputs
- **Rules**: Serve as high-level containers that delegate validation to validators
- **Factory Functions**: Provide a consistent way to create both validators and rules

### Delegation Pattern

Rules in Sifaka follow a delegation pattern:

1. A **Rule** receives text to validate
2. The Rule delegates the validation work to its **Validator**
3. The Validator performs the actual validation logic
4. The Validator returns a result to the Rule
5. The Rule may perform additional processing on the result

This separation of concerns allows for:
- Reusing validation logic across different rules
- Testing validation logic independently
- Extending the framework with custom validators

## Instantiation Patterns

### Recommended Approach

Always use factory functions to create rules and validators:

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a rule using the factory function
rule = create_length_rule(
    min_chars=10,
    max_chars=100,
    rule_id="length_validation"
)

# Validate text
result = rule.validate("This is a test")
```

### Factory Function Hierarchy

Sifaka provides two levels of factory functions:

1. **Validator Factory Functions**: Create standalone validators
   ```python
   from sifaka.rules.formatting.length import create_length_validator
   
   validator = create_length_validator(min_chars=10, max_chars=100)
   ```

2. **Rule Factory Functions**: Create rules with validators
   ```python
   from sifaka.rules.formatting.length import create_length_rule
   
   rule = create_length_rule(min_chars=10, max_chars=100)
   ```

Rule factory functions use validator factory functions internally, ensuring consistent configuration.

### Internal Implementation

When you create a rule using a factory function:

1. The factory function creates a validator using the validator factory
2. The factory function creates a rule with the validator
3. The rule stores the validator and delegates validation to it

```python
# Simplified implementation of a rule factory function
def create_example_rule(**kwargs):
    # Create validator using validator factory
    validator = create_example_validator(**kwargs)
    
    # Create and return rule with validator
    return ExampleRule(validator=validator, **kwargs)
```

## Custom Validators and Rules

You can create custom validators and rules by:

1. Extending the base validator class for your domain
2. Implementing the validation logic
3. Creating a rule that uses your validator
4. Providing factory functions for both

Example:

```python
from sifaka.rules.base import Rule, RuleResult
from sifaka.rules.domain.base import BaseDomainValidator

# Custom validator
class MyCustomValidator(BaseDomainValidator):
    def validate(self, text, **kwargs) -> RuleResult:
        # Implement validation logic
        passed = len(text) > 0
        return RuleResult(passed=passed, message="Custom validation")

# Custom rule
class MyCustomRule(Rule):
    def _create_default_validator(self):
        return MyCustomValidator(self.config)

# Factory functions
def create_my_custom_validator(**kwargs):
    config = MyCustomConfig(**kwargs)
    return MyCustomValidator(config)

def create_my_custom_rule(**kwargs):
    validator = create_my_custom_validator(**kwargs)
    return MyCustomRule(validator=validator, **kwargs)
```

## Best Practices

1. **Always use factory functions** to create rules and validators
2. **Don't instantiate rule or validator classes directly** unless you have a specific reason
3. **Pass configuration as direct parameters** to factory functions, not as dictionaries
4. **Implement both validator and rule factory functions** for all rule types
5. **Document the relationship** between rules and validators in module docstrings
