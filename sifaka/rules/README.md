# Sifaka Rules System

This directory contains the rules system for Sifaka, which provides a flexible and extensible way to validate text.

## Architecture

The rules system follows a delegation pattern:

1. **Rules** are high-level containers that define what to validate
2. **Validators** implement the actual validation logic
3. Rules delegate validation work to their validators

This separation of concerns allows for:
- Reusing validation logic across different rules
- Testing validation logic independently
- Extending the framework with custom validators

## Directory Structure

- `base.py` - Core classes and protocols for the rules system
- `adapters/` - Adapters for using other components (like classifiers) as rules
- `formatting/` - Rules for validating text formatting
- `content/` - Rules for validating content

## Using Rules

The recommended way to create rules is through factory functions:

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a rule using the factory function
rule = create_length_rule(min_chars=10, max_chars=100)

# Validate text
result = rule.validate("This is a test")
```

Each rule type provides two factory functions:
1. `create_X_validator()` - Creates a standalone validator
2. `create_X_rule()` - Creates a rule with a validator

## Rule vs. Validator Relationship

Rules delegate validation work to validators:

```
┌─────────┐         ┌────────────┐
│         │ uses    │            │
│  Rule   ├────────►│ Validator  │
│         │         │            │
└─────────┘         └────────────┘
```

When you call `rule.validate(text)`:
1. The rule receives the text
2. The rule delegates to its validator
3. The validator performs the validation
4. The validator returns a result
5. The rule may process the result further

## Creating Custom Rules

To create a custom rule:

1. Create a validator class that extends `BaseValidator`
2. Implement the validation logic in the validator
3. Create a rule class that extends `Rule`
4. Implement `_create_default_validator()` to create your validator
5. Create factory functions for both the validator and rule

See the [Rules and Validators documentation](../../docs/rules_and_validators.md) for more details on creating custom rules and implementation patterns.

## Best Practices

1. **Always use factory functions** to create rules and validators
2. **Don't instantiate rule or validator classes directly** unless you have a specific reason
3. **Pass configuration as direct parameters** to factory functions, not as dictionaries
4. **Implement both validator and rule factory functions** for all rule types
5. **Document the relationship** between rules and validators in module docstrings
6. **Handle empty text consistently** using the BaseValidator.handle_empty_text method
7. **Use type-safe config classes** that extend RuleConfig
8. **Extract rule-specific parameters** in factory functions
9. **Delegate validation logic** from rules to validators
10. **Add rule_id to metadata** in rule.validate method

For a complete guide on implementing rules and validators, see the [Rules and Validators documentation](../../docs/rules_and_validators.md).
