# Error Handling in Sifaka

This document describes the standardized error handling patterns and exception hierarchy in the Sifaka framework.

## Exception Hierarchy

Sifaka uses a standardized exception hierarchy to provide clear categorization of errors and consistent error handling patterns. The hierarchy is as follows:

```
SifakaError (base class for all Sifaka errors)
├── ValidationError (validation issues)
├── ConfigurationError (configuration issues)
├── RuntimeError (runtime issues)
│   ├── ModelError (model-related issues)
│   ├── ClassifierError (classifier-related issues)
│   ├── CriticError (critic-related issues)
│   └── ChainError (chain-related issues)
└── TimeoutError (timeout issues)
```

## Error Handling Patterns

### Component-Specific Error Handling

Each component type should follow these error handling patterns:

#### Models

Models should propagate errors with context:

```python
try:
    # Core generation logic
    return response_text
except Exception as e:
    # Log the error
    logger.error(f"Generation error: {e}")
    
    # Re-raise with context
    raise ModelError(f"Error generating text: {e}", cause=e)
```

#### Classifiers

Classifiers should return fallback results:

```python
try:
    # Core classification logic
    return ClassificationResult(
        label="some_label",
        confidence=0.8,
        metadata={"processed_successfully": True}
    )
except Exception as e:
    # Log the error
    logger.error(f"Classification error: {e}")
    
    # Return a fallback result
    return ClassificationResult(
        label="unknown",
        confidence=0.0,
        metadata={
            "error": str(e),
            "error_type": type(e).__name__,
            "reason": "classification_error"
        }
    )
```

#### Critics

Critics should return fallback feedback:

```python
try:
    # Core critique logic
    return CriticMetadata(
        score=0.8,
        feedback="Good quality text",
        issues=[],
        suggestions=[]
    )
except Exception as e:
    # Log the error
    logger.error(f"Critique error: {e}")
    
    # Return a fallback result
    return CriticMetadata(
        score=0.0,
        feedback=f"Error during critique: {str(e)}",
        issues=["Critique process failed"],
        suggestions=[]
    )
```

#### Rules

Rules should return failure results:

```python
try:
    # Core validation logic
    return RuleResult(
        passed=True,
        message="Validation passed",
        metadata={}
    )
except Exception as e:
    # Log the error
    logger.error(f"Validation error: {e}")
    
    # Return a failure result
    return RuleResult(
        passed=False,
        message=f"Error during validation: {str(e)}",
        metadata={
            "error": str(e),
            "error_type": type(e).__name__,
            "reason": "validation_error"
        }
    )
```

#### Chains

Chains should propagate errors with context:

```python
try:
    # Core chain logic
    return result
except Exception as e:
    # Log the error
    logger.error(f"Chain error: {e}")
    
    # Re-raise with context
    raise ChainError(f"Error running chain: {e}", cause=e)
```

### Using Error Handling Utilities

Sifaka provides several utilities for standardized error handling:

#### Error Metadata Formatting

```python
from sifaka.utils.errors import format_error_metadata

try:
    # Some operation
    pass
except Exception as e:
    metadata = format_error_metadata(e)
    return Result(success=False, metadata=metadata)
```

#### Error Handling Decorator

```python
from sifaka.utils.errors import handle_errors

@handle_errors(fallback_value=None, log_errors=True)
def my_function():
    # Function implementation
    pass
```

#### Error Handling Context Manager

```python
from sifaka.utils.errors import with_error_handling

with with_error_handling("data processing", logger=logger):
    # Operation implementation
    pass
```

## Best Practices

### 1. Use Specific Exception Types

Always use the most specific exception type that applies to the situation:

```python
# Good
raise ValidationError("Invalid input: value must be positive")

# Bad
raise SifakaError("Invalid input: value must be positive")
```

### 2. Include Contextual Information

Include relevant context in error messages:

```python
# Good
raise ConfigurationError(f"Invalid configuration for {component_name}: missing required parameter 'max_tokens'")

# Bad
raise ConfigurationError("Invalid configuration")
```

### 3. Preserve Original Exceptions

When catching and re-raising exceptions, preserve the original exception as the cause:

```python
# Good
try:
    # Some operation
    pass
except Exception as e:
    raise RuntimeError(f"Operation failed: {e}", cause=e)

# Bad
try:
    # Some operation
    pass
except Exception as e:
    raise RuntimeError(f"Operation failed: {e}")
```

### 4. Log Errors at the Appropriate Level

Use the appropriate logging level for different types of errors:

- `ERROR`: For errors that prevent normal operation
- `WARNING`: For issues that don't prevent operation but are concerning
- `INFO`: For normal operation events

```python
try:
    # Some operation
    pass
except ValidationError as e:
    # This is expected in normal operation
    logger.warning(f"Validation failed: {e}")
except RuntimeError as e:
    # This prevents normal operation
    logger.error(f"Runtime error: {e}")
```

### 5. Provide Fallback Behavior

When possible, provide fallback behavior rather than failing completely:

```python
try:
    result = complex_operation()
    return result
except Exception as e:
    logger.error(f"Complex operation failed: {e}")
    return fallback_result
```

### 6. Use Error Handling Utilities

Use the provided error handling utilities for consistent error handling:

```python
@handle_errors(fallback_value=None, log_errors=True)
def my_function():
    # Function implementation
    pass

with with_error_handling("data processing", logger=logger):
    # Operation implementation
    pass
```

## Migration Guide

To migrate existing code to use the new error handling system:

1. Replace existing exception types with the new hierarchy:
   - `ValidationError` → `sifaka.utils.errors.ValidationError`
   - `ConfigurationError` → `sifaka.utils.errors.ConfigurationError`

2. Add error handling utilities to functions and methods:
   - Use `@handle_errors` for functions that should return fallback values
   - Use `with_error_handling` for operations that should be logged

3. Update error metadata formatting:
   - Use `format_error_metadata` to standardize error metadata

4. Update component-specific error handling:
   - Follow the patterns described in this document for each component type
