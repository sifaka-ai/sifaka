# Length Validator

This page documents the length validator in Sifaka.

## Overview

The length validator checks if text meets length requirements in terms of words or characters. It can enforce minimum and maximum constraints for both word count and character count.

## Basic Usage

```python
from sifaka.validators import length

# Create a validator that checks word count
validator = length(min_words=50, max_words=200)

# Validate text
result = validator.validate("This is a short text.")
if not result.passed:
    print(f"Validation failed: {result.message}")
```

## API Reference

### Function

```python
length(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> LengthValidator
```

Create a length validator.

**Parameters:**
- `min_words`: Minimum number of words required.
- `max_words`: Maximum number of words allowed.
- `min_chars`: Minimum number of characters required.
- `max_chars`: Maximum number of characters allowed.

**Returns:**
- A LengthValidator instance.

**Raises:**
- `ValidationError`: If no constraints are provided or if min > max.

### Class

```python
LengthValidator(
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
)
```

Validator that checks if text meets length requirements.

**Parameters:**
- `min_words`: Minimum number of words required.
- `max_words`: Maximum number of words allowed.
- `min_chars`: Minimum number of characters required.
- `max_chars`: Maximum number of characters allowed.

**Raises:**
- `ValidationError`: If no constraints are provided or if min > max.

#### Methods

##### `validate`

```python
validate(self, text: str) -> ValidationResult
```

Validate text against length requirements.

**Parameters:**
- `text`: The text to validate.

**Returns:**
- A ValidationResult indicating whether the text meets the length requirements.

## Examples

### Word Count Constraints

```python
from sifaka.validators import length

# Create a validator with word count constraints
validator = length(min_words=10, max_words=100)

# Validate text that's too short
result = validator.validate("This is short.")
print(result.passed)  # False
print(result.message)  # "Text is too short: 3 words, minimum 10 words required"

# Validate text that's too long
long_text = " ".join(["word"] * 150)
result = validator.validate(long_text)
print(result.passed)  # False
print(result.message)  # "Text is too long: 150 words, maximum 100 words allowed"

# Validate text that meets requirements
good_text = " ".join(["word"] * 50)
result = validator.validate(good_text)
print(result.passed)  # True
print(result.message)  # "Text meets length requirements"
```

### Character Count Constraints

```python
from sifaka.validators import length

# Create a validator with character count constraints
validator = length(min_chars=20, max_chars=500)

# Validate text that's too short
result = validator.validate("Too short")
print(result.passed)  # False
print(result.message)  # "Text is too short: 9 characters, minimum 20 characters required"

# Validate text that meets requirements
good_text = "This text has more than twenty characters."
result = validator.validate(good_text)
print(result.passed)  # True
print(result.message)  # "Text meets length requirements"
```

### Combined Constraints

```python
from sifaka.validators import length

# Create a validator with both word and character count constraints
validator = length(min_words=10, max_words=100, min_chars=50, max_chars=500)

# All constraints must be satisfied
text = "This text has enough characters but not enough words."
result = validator.validate(text)
print(result.passed)  # False
print(result.message)  # "Text is too short: 9 words, minimum 10 words required"
```

### Accessing Details

The validation result includes details about the validation:

```python
from sifaka.validators import length

validator = length(min_words=10, max_words=100)
result = validator.validate("This is a short text.")

print(result.details["word_count"])  # 5
print(result.details["constraint_violated"])  # "min_words"
```

## Using with Chain

The length validator is commonly used with the Chain API:

```python
from sifaka import Chain
from sifaka.validators import length

result = (Chain()
    .with_model("openai:gpt-4")
    .with_prompt("Write a short story about a robot.")
    .validate_with(length(min_words=50, max_words=200))
    .run())

if result.passed:
    print("Text meets length requirements")
    print(result.text)
else:
    print(f"Validation failed: {result.validation_results[0].message}")
```
