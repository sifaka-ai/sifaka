# Results

This page documents the result types returned by Sifaka operations.

## Overview

Sifaka uses three main result types:

1. `ValidationResult`: Returned by validators to indicate whether text meets specific criteria
2. `ImprovementResult`: Returned by improvers to provide information about improvements made to text
3. `Result`: Returned by the Chain's run method, containing the final text and all validation and improvement results

## ValidationResult

The `ValidationResult` class represents the result of a validation operation.

### Attributes

- `passed` (bool): Whether the validation passed
- `message` (Optional[str]): A human-readable message describing the validation result
- `details` (Optional[Dict[str, Any]]): Additional details about the validation result

### Methods

#### `__bool__`

```python
__bool__() -> bool
```

Allows using the result in a boolean context. Returns `passed`.

### Example

```python
from sifaka.validators import length

validator = length(min_words=50, max_words=200)
result = validator.validate("This is a short text.")

if not result:  # Same as if not result.passed
    print(f"Validation failed: {result.message}")
    
# Access details
if result.details:
    print(f"Word count: {result.details.get('word_count')}")
```

## ImprovementResult

The `ImprovementResult` class represents the result of an improvement operation.

### Attributes

- `original_text` (str): The original text before improvement
- `improved_text` (str): The improved text
- `changes_made` (bool): Whether any changes were made
- `message` (Optional[str]): A human-readable message describing the improvements
- `details` (Optional[Dict[str, Any]]): Additional details about the improvements

### Methods

#### `__bool__`

```python
__bool__() -> bool
```

Allows using the result in a boolean context. Returns `changes_made`.

### Example

```python
from sifaka.validators import clarity

improver = clarity()
improved_text, result = improver.improve("Original text that could be clearer.")

if result:  # Same as if result.changes_made
    print("Text was improved")
    print(f"Original: {result.original_text}")
    print(f"Improved: {result.improved_text}")
    print(f"Message: {result.message}")
```

## Result

The `Result` class represents the result of a chain execution.

### Attributes

- `text` (str): The final text after all validations and improvements
- `passed` (bool): Whether all validations passed
- `validation_results` (List[ValidationResult]): Results of all validations
- `improvement_results` (List[ImprovementResult]): Results of all improvements
- `metadata` (Optional[Dict[str, Any]]): Additional metadata about the result

### Methods

#### `__bool__`

```python
__bool__() -> bool
```

Allows using the result in a boolean context. Returns `passed`.

### Example

```python
from sifaka import Chain
from sifaka.validators import length, clarity

chain = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=50, max_words=200))
    .improve_with(clarity())
    .run())

if chain:  # Same as if chain.passed
    print("Chain execution succeeded")
    print(chain.text)
else:
    print("Chain execution failed validation")
    
# Access validation results
for result in chain.validation_results:
    print(f"Validation: {result.passed} - {result.message}")
    
# Access improvement results
for result in chain.improvement_results:
    print(f"Improvement: {result.changes_made} - {result.message}")
```

## Usage in Chain

When you call `run()` on a Chain, it returns a `Result` object containing:

1. The final text after all validations and improvements
2. Whether all validations passed
3. The results of all validations
4. The results of all improvements

```python
from sifaka import Chain
from sifaka.validators import length, clarity

result = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=50, max_words=200))
    .improve_with(clarity())
    .run())

print(f"Result passed validation: {result.passed}")
print(result.text)

# Print validation results
for i, validation_result in enumerate(result.validation_results):
    print(f"Validation {i+1}: {validation_result.message}")

# Print improvement results
for i, improvement_result in enumerate(result.improvement_results):
    print(f"Improvement {i+1}: {improvement_result.message}")
```
