# Empty Text Handling in Sifaka

This document describes the standardized approach to handling empty text inputs across the Sifaka framework.

## Overview

Empty text handling is a common requirement across different components in Sifaka. Previously, there were inconsistencies in how empty text was handled:

- In rules: Empty text would PASS validation (`passed=True`)
- In adapters: Empty text would FAIL validation (`passed=False`)
- In classifiers: Empty text would return "unknown" label with 0.0 confidence

To address this inconsistency, we've implemented a standardized approach using a centralized utility.

## Standardized Approach

The standardized approach is implemented in `sifaka.utils.text`:

```python
from sifaka.utils.text import handle_empty_text

# Check if text is empty
empty_result = handle_empty_text(text)
if empty_result:
    return empty_result
```

### Default Behavior

The default behavior for empty text handling is:

- **Rules**: Empty text PASSES validation (`passed=True`)
- **Adapters**: Empty text PASSES validation (`passed=True`) for consistency
- **Classifiers**: Empty text returns "unknown" label with 0.0 confidence

This standardized approach ensures consistent behavior across the framework while maintaining backward compatibility.

## Implementation Details

### Utility Functions

The `sifaka.utils.text` module provides three key functions:

1. `is_empty_text(text: str) -> bool`: Checks if text is empty or contains only whitespace
2. `handle_empty_text(text: str, passed: bool = True, ...) -> Optional[RuleResult]`: Standardized handling for empty text in rules and adapters
3. `handle_empty_text_for_classifier(text: str, ...) -> Optional[ClassificationResult]`: Standardized handling for empty text in classifiers

### Usage in Components

#### In Validators

```python
def validate(self, text: str, **kwargs) -> RuleResult:
    # Handle empty text first
    empty_result = self.handle_empty_text(text)
    if empty_result:
        return empty_result

    # Proceed with normal validation
    # ...
```

#### In Adapters

```python
def validate(self, input_text: str, **kwargs) -> RuleResult:
    # Handle empty text first
    empty_result = self.handle_empty_text(input_text)
    if empty_result:
        return empty_result

    # Proceed with normal validation
    # ...
```

#### In Classifiers

```python
def classify(self, text: str) -> ClassificationResult:
    # Handle empty text
    from sifaka.utils.text import handle_empty_text_for_classifier
    empty_result = handle_empty_text_for_classifier(text)
    if empty_result:
        return empty_result

    # Proceed with normal classification
    # ...
```

## Benefits

This standardized approach provides several benefits:

1. **Consistency**: All components handle empty text in a consistent way
2. **Maintainability**: Changes to empty text handling can be made in one place
3. **Configurability**: The behavior can be configured as needed
4. **Documentation**: The approach is well-documented and easy to understand

## Migration Guide

To migrate existing code to use the standardized approach:

1. Replace direct empty text checks with calls to `handle_empty_text()`
2. Update tests to expect the standardized behavior
3. Remove any component-specific empty text handling logic

## Example

Before:
```python
def validate(self, text: str, **kwargs) -> RuleResult:
    if not text.strip():
        return RuleResult(
            passed=True,
            message="Empty text validation skipped",
            metadata={"reason": "empty_input"}
        )
    # ...
```

After:
```python
def validate(self, text: str, **kwargs) -> RuleResult:
    from sifaka.utils.text import handle_empty_text
    empty_result = handle_empty_text(text)
    if empty_result:
        return empty_result
    # ...
```
