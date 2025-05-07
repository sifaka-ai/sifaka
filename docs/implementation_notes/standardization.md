# Standardization in Sifaka

This document describes the standardized approaches to common patterns in the Sifaka framework.

## Overview

Standardization is essential for maintaining code quality, consistency, and maintainability. The Sifaka framework provides several standardized utilities for common patterns:

1. **Error Handling**: Standardized error handling with consistent exception classes and error reporting
2. **Configuration Validation**: Standardized configuration validation with consistent validation methods
3. **Pattern Matching**: Standardized pattern matching with regex compilation and caching
4. **Result Creation**: Standardized result creation with consistent metadata structure
5. **Empty Text Handling**: Standardized empty text handling across components

## Error Handling

The `sifaka.utils.errors` module provides standardized error handling:

```python
from sifaka.utils.errors import (
    SifakaError, ValidationError, try_operation, handle_error
)

# Using exception classes
try:
    # Some operation
    if invalid_condition:
        raise ValidationError("Invalid input", metadata={"field": "name"})
except SifakaError as e:
    # Handle Sifaka-specific error
    print(f"Error: {e.message}")
    print(f"Metadata: {e.metadata}")
except Exception as e:
    # Handle other errors
    error_info = handle_error(e, "MyComponent")
    print(f"Unexpected error: {error_info['error_message']}")

# Using try_operation
result = try_operation(
    lambda: process_data(input_data),
    component_name="DataProcessor",
    default_value=None,
    log_level="error"
)
```

### Exception Hierarchy

Sifaka uses a structured exception hierarchy:

1. **SifakaError**: Base class for all Sifaka exceptions
   - **ValidationError**: Raised when validation fails
   - **ConfigurationError**: Raised when configuration is invalid
   - **ProcessingError**: Raised when processing fails
     - **ResourceError**: Raised when a resource is unavailable
     - **TimeoutError**: Raised when an operation times out

2. **Common Error Types**:
   - **InputError**: Raised when input is invalid
   - **StateError**: Raised when state is invalid
   - **DependencyError**: Raised when a dependency fails

### Error Handling Functions

The module provides standardized error handling functions:

1. **try_operation**: Execute an operation with standardized error handling
2. **handle_error**: Process an error and return standardized metadata
3. **log_error**: Log an error with standardized formatting

## Configuration Validation

The `sifaka.utils.config` module provides standardized configuration validation:

```python
from sifaka.utils.config import (
    validate_config, validate_params, ComponentConfig
)
from pydantic import Field
from typing import Dict, Optional

# Define a configuration class
class MyComponentConfig(ComponentConfig):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    patterns: Dict[str, str] = Field(default_factory=dict)
    max_items: Optional[int] = Field(None, gt=0)

# Validate configuration
config_dict = {
    "threshold": 0.7,
    "patterns": {"key": "value"},
    "max_items": 10
}
config = validate_config(config_dict, MyComponentConfig)

# Validate parameters
params = {
    "threshold": 0.7,
    "patterns": {"key": "value"},
    "max_items": 10
}
validated_params = validate_params(
    params,
    {
        "threshold": (float, (0.0, 1.0)),
        "patterns": (dict, None),
        "max_items": (int, (1, None), True)
    }
)
```

### Configuration Classes

The module provides base classes for different configuration types:

1. **BaseConfig**: Base class for all configuration classes
2. **ComponentConfig**: Base class for component configurations
3. **ValidationConfig**: Base class for validation configurations

### Configuration Functions

The module provides standardized configuration functions:

1. **validate_config**: Validate configuration against a Pydantic model
2. **validate_params**: Validate parameters against expected types and constraints
3. **merge_configs**: Merge multiple configurations with precedence

## Pattern Matching

The `sifaka.utils.patterns` module provides standardized pattern matching:

```python
from sifaka.utils.patterns import (
    compile_pattern, match_pattern, find_patterns
)

# Compile a pattern
pattern = compile_pattern(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    case_sensitive=False
)

# Match a pattern
is_match = match_pattern(
    "user@example.com",
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    case_sensitive=False
)

# Find patterns
matches = find_patterns(
    "Contact us at user@example.com or support@example.com",
    {
        "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    },
    case_sensitive=False
)
```

### Pattern Functions

The module provides standardized pattern matching functions:

1. **compile_pattern**: Compile a regex pattern with caching
2. **match_pattern**: Match a pattern against text
3. **find_patterns**: Find all matches of patterns in text
4. **count_patterns**: Count matches of patterns in text
5. **match_glob**: Match a glob pattern against text
6. **match_wildcard**: Match a wildcard pattern against text

## Result Creation

The `sifaka.utils.results` module provides standardized result creation:

```python
from sifaka.utils.results import (
    create_rule_result, create_classification_result, create_critic_result
)

# Create a rule result
result = create_rule_result(
    passed=True,
    message="Validation passed",
    component_name="LengthValidator",
    metadata={"text_length": 100}
)

# Create a classification result
result = create_classification_result(
    label="positive",
    confidence=0.85,
    component_name="SentimentClassifier",
    metadata={"pos_score": 0.85, "neg_score": 0.15}
)

# Create a critic result
result = create_critic_result(
    score=0.75,
    feedback="Good content, but could be more concise",
    component_name="ContentCritic",
    issues=["Too verbose", "Redundant information"]
)
```

### Result Functions

The module provides standardized result creation functions:

1. **create_rule_result**: Create a standardized rule result
2. **create_classification_result**: Create a standardized classification result
3. **create_critic_result**: Create a standardized critic result
4. **create_error_result**: Create a standardized error result
5. **create_unknown_result**: Create a standardized unknown classification result
6. **merge_metadata**: Merge multiple metadata dictionaries

## Empty Text Handling

The `sifaka.utils.text` module provides standardized empty text handling:

```python
from sifaka.utils.text import (
    is_empty_text, handle_empty_text, handle_empty_text_for_classifier
)

# Check if text is empty
if is_empty_text(text):
    # Handle empty text
    pass

# Handle empty text in rules and adapters
empty_result = handle_empty_text(text)
if empty_result:
    return empty_result

# Handle empty text in classifiers
empty_result = handle_empty_text_for_classifier(text)
if empty_result:
    return empty_result
```

### Empty Text Functions

The module provides standardized empty text handling functions:

1. **is_empty_text**: Check if text is empty or contains only whitespace
2. **handle_empty_text**: Standardized handling for empty text in rules and adapters
3. **handle_empty_text_for_classifier**: Standardized handling for empty text in classifiers

## Migration Guide

To migrate existing code to use the standardized utilities:

1. **Error Handling**:
   - Replace custom error handling with `try_operation` and `handle_error`
   - Replace custom exceptions with Sifaka exception classes

2. **Configuration Validation**:
   - Replace custom configuration validation with `validate_config` and `validate_params`
   - Extend `BaseConfig`, `ComponentConfig`, or `ValidationConfig` for configuration classes

3. **Pattern Matching**:
   - Replace direct regex usage with `compile_pattern`, `match_pattern`, and `find_patterns`
   - Use pattern caching for improved performance

4. **Result Creation**:
   - Replace direct result creation with `create_rule_result`, `create_classification_result`, and `create_critic_result`
   - Use consistent metadata structure

5. **Empty Text Handling**:
   - Replace direct empty text checks with `is_empty_text`
   - Replace custom empty text handling with `handle_empty_text` and `handle_empty_text_for_classifier`

## Benefits

Standardization provides several benefits:

1. **Consistency**: All components use consistent patterns and approaches
2. **Maintainability**: Changes can be made in one place
3. **Readability**: Code is more readable and easier to understand
4. **Reliability**: Standardized approaches are more reliable and less error-prone
5. **Performance**: Standardized approaches can be optimized for performance
