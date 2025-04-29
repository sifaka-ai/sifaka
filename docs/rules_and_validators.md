# Rules and Validators in Sifaka

This document explains the relationship between Rules and Validators in the Sifaka framework and provides guidance on consistent implementation patterns.

## Relationship Between Rules and Validators

### Core Concepts

- **Validators**: Perform the actual validation logic on text inputs
- **Rules**: Serve as high-level containers that delegate validation to validators
- **Factory Functions**: Provide a consistent way to create both validators and rules
- **Config Classes**: Extend RuleConfig to provide type-safe access to parameters

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
from sifaka.rules.base import Rule, RuleResult, RuleConfig, BaseValidator
from typing import Optional

# Custom config class
@dataclass(frozen=True)
class MyCustomConfig(RuleConfig):
    min_length: int = 10
    max_length: int = 100
    check_spelling: bool = False

# Custom validator
class MyCustomValidator(BaseValidator[str]):
    def __init__(self, config: MyCustomConfig):
        super().__init__()
        self.config = config

    def validate(self, text: str, **kwargs) -> RuleResult:
        # Handle empty text
        empty_result = self.handle_empty_text(text)
        if empty_result:
            return empty_result

        # Implement validation logic
        errors = []
        if len(text) < self.config.min_length:
            errors.append(f"Text too short: {len(text)} chars (min {self.config.min_length})")

        return RuleResult(
            passed=not errors,
            message=errors[0] if errors else "Validation passed",
            metadata={"errors": errors}
        )

# Custom rule
class MyCustomRule(Rule):
    def __init__(
        self,
        name: str = "my_custom_rule",
        description: str = "Custom validation rule",
        config: Optional[RuleConfig] = None,
        validator: Optional[MyCustomValidator] = None,
        **kwargs,
    ):
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> MyCustomValidator:
        """Create a default validator from config."""
        custom_config = MyCustomConfig(**self._rule_params)
        return MyCustomValidator(custom_config)

# Factory functions
def create_my_custom_validator(
    min_length: int = 10,
    max_length: int = 100,
    check_spelling: bool = False,
    **kwargs,
) -> MyCustomValidator:
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with parameters
    config_params = {
        "min_length": min_length,
        "max_length": max_length,
        "check_spelling": check_spelling,
        **rule_config_params,
    }

    # Create config and validator
    config = MyCustomConfig(**config_params)
    return MyCustomValidator(config)

def create_my_custom_rule(
    name: str = "my_custom_rule",
    description: str = "Custom validation rule",
    min_length: int = 10,
    max_length: int = 100,
    check_spelling: bool = False,
    **kwargs,
) -> MyCustomRule:
    # Create validator using the validator factory
    validator = create_my_custom_validator(
        min_length=min_length,
        max_length=max_length,
        check_spelling=check_spelling,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return MyCustomRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
```

## Implementation Patterns

### Rule and Validator Relationship

```
┌─────────┐         ┌────────────┐
│         │ uses    │            │
│  Rule   ├────────►│ Validator  │
│         │         │            │
└─────────┘         └────────────┘
```

### Standard Implementation Pattern

1. **Config Class**: Extend RuleConfig for type-safe parameter access
   ```python
   @dataclass(frozen=True)
   class StyleConfig(RuleConfig):
       capitalization: Optional[CapitalizationStyle] = None
       require_end_punctuation: bool = False
       # ...other parameters
   ```

2. **Validator Class**: Implement validation logic
   ```python
   class DefaultStyleValidator(BaseValidator[str]):
       def __init__(self, config: StyleConfig):
           self.config = config

       def validate(self, text: str, **kwargs) -> RuleResult:
           # Handle empty text
           empty_result = self.handle_empty_text(text)
           if empty_result:
               return empty_result

           # Implement validation logic
           # ...
   ```

3. **Rule Class**: Delegate to validator
   ```python
   class StyleRule(Rule):
       def __init__(self, validator: StyleValidator, **kwargs):
           super().__init__(**kwargs)
           self.validator = validator

       def validate(self, text: str, **kwargs) -> RuleResult:
           result = self.validator.validate(text, **kwargs)
           return result.with_metadata(rule_id=self.id)
   ```

4. **Factory Functions**: Create both validators and rules
   ```python
   def create_style_validator(**kwargs) -> StyleValidator:
       # Extract config parameters
       # Create and return validator

   def create_style_rule(**kwargs) -> StyleRule:
       # Create validator using validator factory
       validator = create_style_validator(**kwargs)
       # Create and return rule with validator
       return StyleRule(validator=validator, **kwargs)
   ```

### Empty Text Handling

All validators should handle empty text consistently:

```python
def validate(self, text: str, **kwargs) -> RuleResult:
    # Handle empty text
    empty_result = self.handle_empty_text(text)
    if empty_result:
        return empty_result

    # Continue with validation logic
    # ...
```

The `handle_empty_text` method is provided by BaseValidator and returns a standard result for empty text:

```python
def handle_empty_text(self, text: str) -> Optional[RuleResult]:
    if not text.strip():
        return RuleResult(
            passed=True,
            message="Empty text validation skipped",
            metadata={"reason": "empty_input"},
        )
    return None
```

### Configuration Pattern

All rule-specific configuration should be stored in RuleConfig.params:

1. **Direct Parameters**: Factory functions should accept direct parameters
   ```python
   def create_style_rule(
       capitalization: Optional[CapitalizationStyle] = None,
       require_end_punctuation: bool = False,
       # ...other parameters
   )
   ```

2. **Config Classes**: Extend RuleConfig for type-safe access
   ```python
   @dataclass(frozen=True)
   class StyleConfig(RuleConfig):
       capitalization: Optional[CapitalizationStyle] = None
       require_end_punctuation: bool = False
   ```

3. **Parameter Extraction**: Factory functions should extract parameters
   ```python
   # Extract RuleConfig parameters from kwargs
   rule_config_params = {}
   for param in ["priority", "cache_size", "cost", "params"]:
       if param in kwargs:
           rule_config_params[param] = kwargs.pop(param)
   ```

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

## Example Module Documentation

```python
"""
Style validation rules for text.

This module provides validators and rules for checking text styling constraints
such as capitalization, punctuation, and other formatting standards.

This module follows the standard Sifaka delegation pattern:
- Rules delegate validation work to validators
- Validators implement the actual validation logic
- Factory functions provide a consistent way to create both

Usage Example:
    from sifaka.rules.formatting.style import create_style_rule, CapitalizationStyle

    # Create a style rule using the factory function
    rule = create_style_rule(
        capitalization=CapitalizationStyle.SENTENCE_CASE,
        require_end_punctuation=True,
        rule_id="sentence_style"
    )

    # Validate text
    result = rule.validate("This is a test.")
"""
```
