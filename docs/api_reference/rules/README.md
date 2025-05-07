# Rules API Reference

This document provides detailed API reference for all Rules in Sifaka.

## Overview

Rules are the validation components in Sifaka that perform binary pass/fail checks on text. They are used to enforce constraints on generated content and provide feedback when those constraints are not met. Rules serve as the primary validation mechanism in the Sifaka framework.

Rules implement these main interfaces:
- **RuleProtocol**: The core interface for all rule-like objects
- **RuleValidator**: The interface for validation logic that can be used by rules
- **Rule**: The base class for all validation rules

## Core Interfaces

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

Sifaka provides factory functions for creating rules. Always use these factory functions instead of instantiating rule classes directly.

### create_rule

```python
def create_rule(
    name: str,
    description: str,
    validator: RuleValidator[T],
    config: Optional[RuleConfig] = None,
    **kwargs
) -> Rule[T]:
    """
    Create a rule with a custom validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        validator: Validator to use for validation
        config: Configuration for the rule
        **kwargs: Additional keyword arguments for the rule configuration

    Returns:
        A rule instance

    Raises:
        ValueError: If parameters are invalid
    """
```

### create_function_rule

```python
def create_function_rule(
    func: Callable[[str], bool],
    name: str,
    description: str,
    failure_message: str = "Validation failed",
    success_message: str = "Validation passed",
    config: Optional[RuleConfig] = None,
    **kwargs
) -> Rule[str]:
    """
    Create a rule from a validation function.

    Args:
        func: Function that takes a string and returns a boolean
        name: Name of the rule
        description: Description of the rule
        failure_message: Message to include when validation fails
        success_message: Message to include when validation passes
        config: Configuration for the rule
        **kwargs: Additional keyword arguments for the rule configuration

    Returns:
        A rule instance

    Raises:
        ValueError: If parameters are invalid
    """
```

### create_regex_rule

```python
def create_regex_rule(
    pattern: str,
    name: str = "regex_rule",
    description: str = "Validates text against a regex pattern",
    match_mode: str = "contains",
    failure_message: Optional[str] = None,
    success_message: Optional[str] = None,
    config: Optional[RuleConfig] = None,
    **kwargs
) -> Rule[str]:
    """
    Create a rule that validates text against a regex pattern.

    Args:
        pattern: Regex pattern to match
        name: Name of the rule
        description: Description of the rule
        match_mode: Mode for matching ('contains', 'matches', 'not_contains', 'not_matches')
        failure_message: Message to include when validation fails
        success_message: Message to include when validation passes
        config: Configuration for the rule
        **kwargs: Additional keyword arguments for the rule configuration

    Returns:
        A rule instance

    Raises:
        ValueError: If parameters are invalid
    """
```

### create_and_rule

```python
def create_and_rule(
    rules: List[Rule[T]],
    name: str = "and_rule",
    description: str = "Validates that all rules pass",
    config: Optional[RuleConfig] = None,
    **kwargs
) -> Rule[T]:
    """
    Create a rule that passes only if all component rules pass.

    Args:
        rules: List of rules to combine
        name: Name of the rule
        description: Description of the rule
        config: Configuration for the rule
        **kwargs: Additional keyword arguments for the rule configuration

    Returns:
        A rule instance

    Raises:
        ValueError: If parameters are invalid
    """
```

### create_or_rule

```python
def create_or_rule(
    rules: List[Rule[T]],
    name: str = "or_rule",
    description: str = "Validates that at least one rule passes",
    config: Optional[RuleConfig] = None,
    **kwargs
) -> Rule[T]:
    """
    Create a rule that passes if any component rule passes.

    Args:
        rules: List of rules to combine
        name: Name of the rule
        description: Description of the rule
        config: Configuration for the rule
        **kwargs: Additional keyword arguments for the rule configuration

    Returns:
        A rule instance

    Raises:
        ValueError: If parameters are invalid
    """
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

## Implementation Details

Rules in Sifaka follow a standardized implementation pattern:

1. **State Management**: Rules use the `_state_manager` pattern for managing state
2. **Configuration**: Rules use `RuleConfig` for configuration
3. **Factory Functions**: Rules provide factory functions for easy instantiation
4. **Delegation**: Rules delegate validation logic to validators

### State Management

Rules use the `_state_manager` pattern for managing state:

```python
from pydantic import PrivateAttr
from sifaka.rules.base import Rule, RuleConfig, RuleValidator, create_rule_state

class MyRule(Rule):
    """Custom rule implementation."""

    _state_manager = PrivateAttr(default_factory=create_rule_state)

    def __init__(self, name: str, description: str, config: RuleConfig):
        super().__init__(name=name, description=description, config=config)
        # Initialize any rule-specific attributes

    def warm_up(self):
        """Initialize expensive resources."""
        state = self._state_manager.get_state()
        if not state.initialized:
            # Initialize state
            state.initialized = True

    def validate(self, text: str) -> RuleResult:
        """Validate the text."""
        state = self._state_manager.get_state()

        # Check cache
        if text in state.cache:
            return state.cache[text]

        # Perform validation
        result = self._validate_internal(text)

        # Cache result
        state.cache[text] = result

        return result
```

### Validator Delegation

Rules typically delegate validation logic to validators:

```python
from sifaka.rules.base import Rule, RuleConfig, RuleValidator, RuleResult

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

class MyRule(Rule):
    """Custom rule implementation."""

    def __init__(self, name: str, description: str, config: RuleConfig, validator: RuleValidator[str]):
        super().__init__(name=name, description=description, config=config)
        self.validator = validator

    def validate(self, text: str) -> RuleResult:
        """Validate the text."""
        # Delegate to validator
        result = self.validator.validate(text)

        # Update rule-specific information
        result.rule_name = self.name
        result.metadata["rule_id"] = id(self)

        return result
```

## Best Practices

1. **Use factory functions** for creating rules and validators
2. **Don't instantiate rule or validator classes directly** unless you have a specific reason
3. **Pass configuration as direct parameters** to factory functions, not as dictionaries
4. **Implement both validator and rule factory functions** for all rule types
5. **Document the relationship** between rules and validators in module docstrings
6. **Handle empty text consistently** using the BaseValidator.handle_empty_text method
7. **Use type-safe config classes** that extend RuleConfig
8. **Extract rule-specific parameters** in factory functions
9. **Delegate validation logic** from rules to validators
10. **Add rule_id to metadata** in rule.validate method

## Error Handling

Rules implement several error handling patterns:

### Handling Empty Text

```python
def validate(self, text: str) -> RuleResult:
    """Validate the text."""
    if not text:
        return RuleResult(
            passed=True,
            rule_name=self.name,
            message="Empty text is valid",
            metadata={"empty": True}
        )

    # Normal validation logic
    return self._validate_internal(text)
```

### Handling Validation Errors

```python
def validate(self, text: str) -> RuleResult:
    """Validate the text."""
    try:
        # Try to validate
        return self._validate_internal(text)
    except Exception as e:
        # Return failure result
        return RuleResult(
            passed=False,
            rule_name=self.name,
            message=f"Validation error: {str(e)}",
            metadata={"error": str(e), "error_type": type(e).__name__}
        )
```

### Handling Timeouts

```python
def validate(self, text: str) -> RuleResult:
    """Validate the text."""
    try:
        # Try to validate with timeout
        return self._validate_with_timeout(text, timeout=10.0)
    except TimeoutError:
        # Return failure result
        return RuleResult(
            passed=False,
            rule_name=self.name,
            message="Validation timed out",
            metadata={"timeout": True}
        )
```

## See Also

- [Rules Component Documentation](../../components/rules.md)
- [Implementation Notes for Rules](../../implementation_notes/rules.md)
- [Classifiers API Reference](../classifiers/README.md)
- [Critics API Reference](../critics/README.md)
