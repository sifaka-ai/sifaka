# Composition Over Inheritance in Sifaka

This document explains the use of composition over inheritance in Sifaka, which reduces complexity and improves maintainability.

## Overview

Sifaka uses composition over inheritance to create a more maintainable and flexible codebase. This approach:

- Reduces complexity by avoiding deep inheritance hierarchies
- Makes code easier to understand and maintain
- Improves flexibility by allowing components to be combined in different ways
- Reduces coupling between components

## Core Components

### Rule

The `Rule` class is a key example of composition over inheritance in Sifaka. Instead of using a deep inheritance hierarchy, the `Rule` class delegates validation to a validator object:

```python
class Rule(BaseModel):
    name: str
    description: str
    config: RuleConfig
    
    def __init__(
        self,
        name: str,
        description: str,
        validator: RuleValidator,
        config: Optional[RuleConfig] = None,
        result_handler: Optional[RuleResultHandler] = None,
        **kwargs: Any
    ):
        super().__init__(
            name=name,
            description=description,
            config=config or RuleConfig(),
            **kwargs
        )
        self._validator = validator
        self._result_handler = result_handler
    
    def validate(self, output: Any, **kwargs: Any) -> RuleResult:
        # Delegate to the validator
        result = self._validator.validate(output, **kwargs)
        
        # Add rule metadata
        result = result.with_metadata(rule_name=self.name)
        
        # Handle result if handler is provided
        if self._result_handler is not None:
            self._result_handler.handle_result(result)
            if not self._result_handler.should_continue(result):
                return result
        
        return result
```

### Validators

Validators implement the actual validation logic and are composed with rules:

```python
class LengthValidator(BaseValidator[str]):
    def __init__(self, config: LengthConfig):
        self.config = config
    
    def validate(self, text: str, **kwargs) -> RuleResult:
        # Validation logic
        char_count = len(text)
        if self.config.min_chars is not None and char_count < self.config.min_chars:
            return RuleResult(
                passed=False,
                message=f"Text is too short: {char_count} characters (minimum {self.config.min_chars})",
                metadata={"char_count": char_count}
            )
        return RuleResult(
            passed=True,
            message="Text length validation successful",
            metadata={"char_count": char_count}
        )
```

### Factory Functions

Factory functions are used to create rules with their validators:

```python
def create_length_rule(
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    **kwargs
) -> Rule:
    # Create validator
    validator = create_length_validator(
        min_chars=min_chars,
        max_chars=max_chars,
        min_words=min_words,
        max_words=max_words
    )
    
    # Create rule
    return Rule(
        name=kwargs.get("name", "length_rule"),
        description=kwargs.get("description", "Validates text length"),
        validator=validator,
        config=kwargs.get("config")
    )
```

## Benefits

### 1. Simplified Code Structure

Composition over inheritance creates a flatter, more understandable code structure:

- **Before**: Deep inheritance hierarchies with multiple levels of abstraction
- **After**: Flat structure with clear delegation of responsibilities

### 2. Improved Flexibility

Components can be combined in different ways without changing their implementation:

- Different validators can be used with the same rule
- The same validator can be used with different rules
- New validators can be added without changing existing rules

### 3. Better Testability

Components can be tested in isolation:

- Validators can be tested without rules
- Rules can be tested with mock validators
- Integration tests can verify that components work together correctly

### 4. Reduced Coupling

Components are less tightly coupled:

- Changes to validators don't require changes to rules
- Rules don't need to know the implementation details of validators
- New components can be added without changing existing ones

## Implementation Patterns

### 1. Delegation Pattern

Rules delegate validation to validators:

```python
def validate(self, output: Any, **kwargs: Any) -> RuleResult:
    return self._validator.validate(output, **kwargs)
```

### 2. Factory Functions

Factory functions create rules with their validators:

```python
def create_rule(validator, name, description, **kwargs):
    return Rule(
        name=name,
        description=description,
        validator=validator,
        **kwargs
    )
```

### 3. Protocol-Based Interfaces

Protocols define the interfaces that components must implement:

```python
@runtime_checkable
class RuleValidator(Protocol[T_contra]):
    def validate(self, output: T_contra, **kwargs: Any) -> "RuleResult":
        ...
    
    def can_validate(self, output: T_contra) -> bool:
        ...
    
    @property
    def validation_type(self) -> type:
        ...
```

## Best Practices

1. Use factory functions to create components
2. Keep validators focused on a single responsibility
3. Use composition to combine validators for complex rules
4. Document the relationships between components
5. Use protocols to define interfaces
6. Test components in isolation and in combination
