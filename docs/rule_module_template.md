# Rule Module Docstring Template

When creating a new rule module, use this template for the module docstring to ensure consistent documentation of the relationship between rules and validators.

```python
"""
[Rule type] validation rules for Sifaka.

This module provides validators and rules for checking [what the rule validates].

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The [RuleType]Config class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_[rule_type]_rule) handle configuration
    - Validator factory functions (create_[rule_type]_validator) create standalone validators

## Rule and Validator Relationship

This module follows the standard Sifaka delegation pattern:
- Rules delegate validation work to validators
- Validators implement the actual validation logic
- Factory functions provide a consistent way to create both
- Empty text is handled consistently using BaseValidator.handle_empty_text

## Usage Example

```python
from sifaka.rules.[category].[module] import create_[rule_type]_rule

# Create a rule using the factory function
rule = create_[rule_type]_rule(
    param1=value1,
    param2=value2
)

# Validate text
result = rule.validate("Text to validate")
```

## Factory Functions

This module provides two factory functions:

1. `create_[rule_type]_validator()` - Creates a standalone validator
   ```python
   validator = create_[rule_type]_validator(param1=value1, param2=value2)
   ```

2. `create_[rule_type]_rule()` - Creates a rule with a validator
   ```python
   rule = create_[rule_type]_rule(param1=value1, param2=value2)
   ```

## Configuration

The following configuration options are available:

- `param1` (type, optional): Description of parameter 1
- `param2` (type, optional): Description of parameter 2
- `rule_id` (str, optional): Custom identifier for the rule

For more details on implementing rules and validators, see:
- [Rules and Validators Documentation](../../docs/rules_and_validators.md)
"""
```

## Key Elements to Include

1. **Brief description** of what the rule validates
2. **Configuration Pattern** section explaining how configuration is handled
3. **Rule and Validator Relationship** section explaining the delegation pattern
4. **Usage Example** showing how to use the factory function
5. **Factory Functions** section explaining both factory functions
6. **Configuration** section documenting available parameters
7. **Reference to documentation** for more details

## Example Implementation

```python
"""
Length validation rules for Sifaka.

This module provides validators and rules for checking text length constraints.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The LengthConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_length_rule) handle configuration
    - Validator factory functions (create_length_validator) create standalone validators

## Rule and Validator Relationship

This module follows the standard Sifaka delegation pattern:
- Rules delegate validation work to validators
- Validators implement the actual validation logic
- Factory functions provide a consistent way to create both
- Empty text is handled consistently using BaseValidator.handle_empty_text

## Usage Example

```python
from sifaka.rules.formatting.length import create_length_rule

# Create a rule using the factory function
rule = create_length_rule(
    min_chars=10,
    max_chars=100
)

# Validate text
result = rule.validate("This is a test")
```

## Factory Functions

This module provides two factory functions:

1. `create_length_validator()` - Creates a standalone validator
   ```python
   validator = create_length_validator(min_chars=10, max_chars=100)
   ```

2. `create_length_rule()` - Creates a rule with a validator
   ```python
   rule = create_length_rule(min_chars=10, max_chars=100)
   ```

## Configuration

The following configuration options are available:

- `min_chars` (int, optional): Minimum number of characters allowed
- `max_chars` (int, optional): Maximum number of characters allowed
- `rule_id` (str, optional): Custom identifier for the rule

For more details on implementing rules and validators, see:
- [Rules and Validators Documentation](../../docs/rules_and_validators.md)
"""
```

## Implementation Checklist

When implementing a new rule module, ensure it includes:

1. **Module Docstring**:
   - Description of the rule type
   - Explanation of the delegation pattern
   - Usage example
   - Factory function documentation
   - Configuration options

2. **Config Class**:
   - Extends RuleConfig
   - Type-safe parameter access
   - Default values for parameters

3. **Validator Class**:
   - Extends BaseValidator
   - Implements validate() method
   - Handles empty text consistently

4. **Rule Class**:
   - Implements _create_default_validator()
   - Delegates to validator
   - Adds rule_id to metadata

5. **Factory Functions**:
   - create_X_validator()
   - create_X_rule()
   - Proper parameter extraction

6. **Exports**:
   - Config classes
   - Validator classes
   - Rule classes
   - Factory functions
