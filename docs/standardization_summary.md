# Rules and Validators Standardization

This document summarizes the standardization patterns implemented for rules and validators in the Sifaka project.

## Delegation Pattern

All rules now follow the delegation pattern where:

1. Rules delegate validation work to validators
2. Validators implement the actual validation logic
3. Factory functions provide a consistent way to create both
4. Empty text is handled consistently using `BaseValidator.handle_empty_text`

### Example

```python
class MyRule(Rule[str, RuleResult, MyValidator, RuleResultHandler[RuleResult]]):
    """Rule for validating something."""

    def __init__(
        self,
        name: str = "my_rule",
        description: str = "Validates something",
        config: Optional[RuleConfig] = None,
        validator: Optional[MyValidator] = None,
    ) -> None:
        """Initialize with configuration."""
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
        )

    def _create_default_validator(self) -> MyValidator:
        """Create a default validator from config."""
        # Extract specific params
        params = self.config.params
        config = MyConfig(
            param1=params.get("param1", default_value),
            param2=params.get("param2", default_value),
            # Include standard RuleConfig parameters
            cache_size=self.config.cache_size,
            priority=self.config.priority,
            cost=self.config.cost,
        )
        return MyValidator(config)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate the given text."""
        # Delegate to validator
        result = self._validator.validate(text, **kwargs)
        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._name)
```

## Factory Function Pattern

All rule modules now provide two factory functions:

1. `create_X_validator()` - Creates a standalone validator
2. `create_X_rule()` - Creates a rule with a validator

### Example

```python
def create_my_validator(
    param1: Optional[Type1] = None,
    param2: Optional[Type2] = None,
    **kwargs: Any,
) -> MyValidator:
    """Create a validator."""
    # Create config with default or provided values
    config_params = {}
    if param1 is not None:
        config_params["param1"] = param1
    if param2 is not None:
        config_params["param2"] = param2

    # Add any remaining config parameters
    config_params.update(kwargs)

    # Create config
    config = MyConfig(**config_params)

    # Create validator
    return MyValidator(config)


def create_my_rule(
    name: str = "my_rule",
    description: str = "Validates something",
    param1: Optional[Type1] = None,
    param2: Optional[Type2] = None,
    **kwargs: Any,
) -> MyRule:
    """Create a rule."""
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)
    
    # Create validator using the validator factory
    validator = create_my_validator(
        param1=param1,
        param2=param2,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )
    
    # Create params dictionary for RuleConfig
    params = {}
    if param1 is not None:
        params["param1"] = param1
    if param2 is not None:
        params["param2"] = param2
    
    # Create RuleConfig
    config = RuleConfig(params=params, **rule_config_params)
    
    # Create rule
    return MyRule(
        name=name,
        description=description,
        config=config,
        validator=validator,
    )
```

## Empty Text Handling

All validators now handle empty text consistently:

```python
def validate(self, text: str, **kwargs) -> RuleResult:
    """Validate text."""
    # Handle empty text
    empty_result = self.handle_empty_text(text)
    if empty_result:
        return empty_result
    
    # Continue with validation logic
    # ...
```

## Configuration Handling

All rule-specific configuration is now stored in `RuleConfig.params`:

```python
# Create a rule with configuration
rule = create_my_rule(
    name="my_rule",
    param1="value1",
    param2="value2",
    priority=RulePriority.HIGH,
    cost=1.0,
)
```

## Module Documentation

All rule modules now include standardized documentation:

```python
"""
Module description.

## Rule and Validator Relationship

This module follows the standard Sifaka delegation pattern:
- Rules delegate validation work to validators
- Validators implement the actual validation logic
- Factory functions provide a consistent way to create both
- Empty text is handled consistently using BaseValidator.handle_empty_text

## Configuration Pattern

This module follows the standard Sifaka configuration pattern:
- All rule-specific configuration is stored in RuleConfig.params
- Factory functions handle configuration extraction
- Validator factory functions create standalone validators
- Rule factory functions use validator factory functions internally

## Usage Example

```python
from sifaka.rules.module import create_my_rule

# Create a rule using the factory function
rule = create_my_rule(
    param1="value1",
    param2="value2"
)

# Validate text
result = rule.validate("This is a test.")
```
"""
```

## Testing

All standardized rules and validators are tested to ensure:

1. Rules properly delegate to validators
2. Factory functions create properly configured rules and validators
3. Empty text is handled consistently across all validators
4. Configuration is properly handled with `RuleConfig` and params
